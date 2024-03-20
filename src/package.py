# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import argparse
import bisect
import gzip
import io
import itertools
import json
import os
import pathlib
import re
import shutil
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import packaging.version as pv
import spack.version as vn
from packaging.markers import Marker, Op, Value, Variable
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from spack.error import UnsatisfiableSpecError
from spack.parser import SpecSyntaxError
from spack.spec import Spec
from spack.util.naming import mod_to_class
from spack.version.version_types import VersionStrComponent, _prev_version_str_component

# If a marker on python version satisfies this range, we statically evaluate it as true.
UNSUPPORTED_PYTHON = vn.VersionRange(
    vn.StandardVersion.typemin(), vn.StandardVersion.from_string("3.7")
)

# The prefix to use for Pythohn package names in Spack.
SPACK_PREFIX = "py-"

NAME_REGEX = re.compile(r"[-_.]+")

DB_URL = "https://github.com/haampie/pypi-to-spack-package/releases/download/latest/data.db.gz"

MAX_VERSIONS = 10

KNOWN_PYTHON_VERSIONS = ((3, 7, 17), (3, 8, 18), (3, 9, 18), (3, 10, 13), (3, 11, 7), (3, 12, 1))

DepToWhen = Tuple[str, vn.VersionList, Optional[Spec], Optional[Marker], FrozenSet[str]]


class Node:
    __slots__ = ("name", "dep_to_when", "version_info", "ordered_versions")

    def __init__(
        self,
        name: str,
        dep_to_when: Dict[DepToWhen, vn.VersionList],
        version_info: Dict[pv.Version, str],
        ordered_versions: List[pv.Version],
    ):
        self.name = name
        self.dep_to_when = dep_to_when
        self.version_info = version_info
        self.ordered_versions = ordered_versions


class VersionsLookup:
    def __init__(self, cursor: sqlite3.Cursor):
        self.cursor = cursor
        self.cache: Dict[str, List[pv.Version]] = {}

    def _query(self, name: str) -> List[pv.Version]:
        query = self.cursor.execute("SELECT version FROM versions WHERE name = ?", (name,))
        return sorted(vv for v, in query if (vv := _acceptable_version(v)))

    def _python_versions(self) -> List[pv.Version]:
        return [
            pv.Version(f"{major}.{minor}.{p}")
            for major, minor, patch in KNOWN_PYTHON_VERSIONS
            for p in range(patch + 1)
        ]

    def __getitem__(self, name: str) -> List[pv.Version]:
        result = self.cache.get(name)
        if result is not None:
            return result
        result = self._query(name) if name != "python" else self._python_versions()
        self.cache[name] = result
        return result


def prev_version_for_range(v: vn.StandardVersion) -> vn.StandardVersion:
    """Translate Specifier <x into a Spack range upperbound :y"""
    # TODO: <0 is broken.
    if len(v.version) == 0:
        return v
    elif isinstance(v.version[-1], VersionStrComponent):
        prev = _prev_version_str_component(v.version[-1])
    elif v.version[-1] == 0:
        return prev_version_for_range(v.up_to(len(v) - 1))
    else:
        prev = v.version[-1] - 1

    # Construct a string-version for printing
    string_components = []
    for part, sep in zip(v.version[:-1], v.separators):
        string_components.append(str(part))
        string_components.append(str(sep))
    string_components.append(str(prev))

    return vn.StandardVersion("".join(string_components), v.version[:-1] + (prev,), v.separators)


def _eval_python_version_marker(variable: str, op: str, value: str) -> Optional[vn.VersionList]:
    # Do everything in terms of ranges for simplicity.

    # `value` has semver semantics. Literal `3` is really `3.0.0`.
    # `python_version`: major.minor
    # `python_full_version`: major.minor.patch  (TODO: rc/alpha/beta etc)

    # `python_version > "3"` is translated as `@3.1:`
    # `python_version > "3.6"` is translated as `@3.7:`
    # `python_version > "3.6.1"` is translated as `@3.7:`
    # `python_version < "3"` is translated as `@:2`
    # `python_version < "3.6"` is translated as `@:3.5`
    # `python_version < "3.6.1"` is translated as `@:3.6`  # note...
    # `python_version == "3"` is translated as `@3.0`
    # `python_full_version > "3"` is translated as `@3.0.1:`
    # `python_full_version > "3.6"` is translated as `@3.6.1:`
    # `python_full_version > "3.6.1"` is translated as `@3.6.2:`

    # Apparently `in` and `not in` work, and interpret the right hand side as TEXT :sob: not as
    # list of versions they parse.
    if op not in ("==", ">", ">=", "<", "<=", "!="):
        return None

    try:
        vv = pv.Version(value)
    except pv.InvalidVersion:
        print(f"could not parse version: `{variable} {op} {value}`", file=sys.stderr)
        return None

    if vv.is_prerelease or vv.is_postrelease or vv.epoch:
        print(f"cannot deal with version: `{variable} {op} {value}`", file=sys.stderr)
        return None
    if variable == "python_version":
        v = vn.StandardVersion.from_string(f"{vv.major}.{vv.minor}")
    elif variable == "python_full_version":
        v = vn.StandardVersion.from_string(f"{vv.major}.{vv.minor}.{vv.micro}")

    if op == "==":
        return vn.VersionList([vn.VersionRange(v, v)])
    elif op == ">":
        return vn.VersionList([vn.VersionRange(vn._next_version(v), vn.StandardVersion.typemax())])
    elif op == ">=":
        return vn.VersionList([vn.VersionRange(v, vn.StandardVersion.typemax())])
    elif op == "<":
        # TODO: This is currently wrong for python_version < x.y.z.
        return vn.VersionList(
            [vn.VersionRange(vn.StandardVersion.typemin(), prev_version_for_range(v))]
        )
    elif op == "<=":
        return vn.VersionList([vn.VersionRange(vn.StandardVersion.typemin(), v)])
    elif op == "!=":
        return vn.VersionList(
            [
                vn.VersionRange(vn.StandardVersion.typemin(), prev_version_for_range(v)),
                vn.VersionRange(vn._next_version(v), vn.StandardVersion.typemax()),
            ]
        )
    print(f"cannot deal with operator: `{variable} {op} {value}`", file=sys.stderr)
    return None


def _eval_constraint(node: tuple) -> Union[None, bool, List[Spec]]:
    # TODO: os_name, platform_machine, platform_release, platform_version, implementation_version

    # Operator
    variable, op, value = node
    assert isinstance(op, Op)

    # Flip the comparison if the value is on the left-hand side.
    if isinstance(variable, Value) and isinstance(value, Variable):
        flipped_op = {
            ">": "<",
            "<": ">",
            ">=": "<=",
            "<=": ">=",
            "==": "==",
            "!=": "!=",
            "~=": "~=",
        }.get(op.value)
        if flipped_op is None:
            print(f"do not know how to evaluate `{node}`", file=sys.stderr)
            return None
        variable, op, value = value, Op(flipped_op), variable

    assert isinstance(variable, Variable)
    assert isinstance(value, Value)

    # Statically evaluate implementation name, since all we support is cpython
    if variable.value == "implementation_name":
        if op.value == "==":
            return value.value == "cpython"
        elif op.value == "!=":
            return value.value != "cpython"
        return None

    if variable.value == "platform_python_implementation":
        if op.value == "==":
            return value.value.lower() == "cpython"
        elif op.value == "!=":
            return value.value.lower() != "cpython"
        return None

    platforms = ("linux", "cray", "darwin", "windows", "freebsd")

    if variable.value == "platform_system" and op.value in ("==", "!="):
        platform = value.value.lower()
        if platform in platforms:
            return [
                Spec(f"platform={p}")
                for p in platforms
                if p != platform and op.value == "!=" or p == platform and op.value == "=="
            ]
        return op.value == "!="  # we don't support it, so statically true/false.

    if variable.value == "sys_platform" and op.value in ("==", "!="):
        platform = value.value.lower()
        if platform == "win32":
            platform = "windows"
        elif platform == "linux2":
            platform = "linux"
        if platform in platforms:
            return [
                Spec(f"platform={p}")
                for p in platforms
                if p != platform and op.value == "!=" or p == platform and op.value == "=="
            ]
        return op.value == "!="  # we don't support it, so statically true/false.

    try:
        if variable.value == "extra":
            if op.value == "==":
                return [Spec(f"+{value.value}")]
            elif op.value == "!=":
                return [Spec(f"~{value.value}")]
    except SpecSyntaxError as e:
        print(f"could not parse `{value}` as variant: {e}", file=sys.stderr)
        return None

    # Otherwise we only know how to handle constraints on the Python version.
    if variable.value not in ("python_version", "python_full_version"):
        return None

    versions = _eval_python_version_marker(variable.value, op.value, value.value)

    if versions is None:
        return None

    simplify_python_constraint(versions)

    if not versions:
        # No supported versions for python remain, so statically false.
        return False
    elif versions == vn.any_version:
        # No constraints on python, so statically true.
        return True
    else:
        spec = Spec("^python")
        spec.dependencies("python")[0].versions = versions
        return [spec]


def _eval_node(node) -> Union[None, bool, List[Spec]]:
    if isinstance(node, tuple):
        return _eval_constraint(node)
    return _do_evaluate_marker(node)


def _intersection(lhs: List[Spec], rhs: List[Spec]) -> List[Spec]:
    """Expand: (a or b) and (c or d) = (a and c) or (a and d) or (b and c) or (b and d)
    where `and` is spec intersection."""
    specs: List[Spec] = []
    for l in lhs:
        for r in rhs:
            intersection = l.copy()
            try:
                intersection.constrain(r)
            except UnsatisfiableSpecError:
                # empty intersection
                continue
            specs.append(intersection)
    return list(set(specs))


def _union(lhs: List[Spec], rhs: List[Spec]) -> List[Spec]:
    """This case is trivial: (a or b) or (c or d) = a or b or c or d, BUT do a simplification
    in case the rhs only expresses constraints on versions."""
    if len(rhs) == 1 and not rhs[0].variants and not rhs[0].platform:
        python, *_ = rhs[0].dependencies("python")
        for l in lhs:
            l.versions.add(python.versions)
        return lhs

    return list(set(lhs + rhs))


def _do_evaluate_marker(node: list) -> Union[None, bool, List[Spec]]:
    """A marker is an expression tree, that we can sometimes translate to the Spack DSL."""
    # Format is like this.
    # python_version > "3.6" or (python_version == "3.6" and os_name == "unix")
    # parsed to
    # [
    #     (<Variable('python_version')>, <Op('>')>, <Value('3.6')>),
    #     'and',
    #     [
    #         (<Variable('python_version')>, <Op('==')>, <Value('3.6')>),
    #         'or',
    #         (<Variable('os_name')>, <Op('==')>, <Value('unix')>)
    #     ]
    # ]
    # Apparently it's flattened.

    assert isinstance(node, list) and len(node) > 0

    lhs = _eval_node(node[0])

    for i in range(2, len(node), 2):
        # Actually op should be constant: x and y and z. we don't assert it here.
        op = node[i - 1]
        assert op in ("and", "or")
        if op == "and":
            if lhs is False:
                return False
            rhs = _eval_node(node[i])
            if rhs is False:
                return False
            elif lhs is None or rhs is None:
                lhs = None
            elif lhs is True:
                lhs = rhs
            elif rhs is not True:  # Intersection of specs
                lhs = _intersection(lhs, rhs)
                # The intersection can be empty, which means it's statically false.
                if not lhs:
                    return False
        elif op == "or":
            if lhs is True:
                return True
            rhs = _eval_node(node[i])
            if rhs is True:
                return True
            elif lhs is None or rhs is None:
                lhs = None
            elif lhs is False:
                lhs = rhs
            elif rhs is not False:
                lhs = _union(lhs, rhs)
    return lhs


def _evaluate_marker(m: Marker) -> Union[bool, None, List[Spec]]:
    """Evaluate the marker expression tree either (1) as a list of specs that constitute the when
    conditions, (2) statically as True or False given that we only support cpython, (3) None if
    we can't translate it into Spack DSL."""
    return _do_evaluate_marker(m._markers)


def _normalized_name(name):
    return re.sub(NAME_REGEX, "-", name).lower()


def _best_upperbound(curr: vn.StandardVersion, next: vn.StandardVersion) -> vn.StandardVersion:
    """Return the most general upperound that includes curr but not next. Invariant is that
    curr < next."""
    i = 0
    m = min(len(curr), len(next))
    while i < m and curr.version[0][i] == next.version[0][i]:
        i += 1
    return curr if i == m else curr.up_to(i + 1)


def _best_lowerbound(prev: vn.StandardVersion, curr: vn.StandardVersion) -> vn.StandardVersion:
    return _best_upperbound(curr, prev)


def _acceptable_version(version: str) -> Optional[pv.Version]:
    """Maybe parse with packaging"""
    try:
        return pv.parse(version)
    except pv.InvalidVersion:
        return None


def _delete_old_releases(
    possible_versions: Dict[pv.Version, Any], keep: int = MAX_VERSIONS
) -> None:
    """Delete non-latest patch releases, only keep pre-releases if they are the very latest release
    and retain at most `keep` releases overall."""
    if not possible_versions:
        return
    versions_desc = sorted(possible_versions.keys(), reverse=True)
    curr = versions_desc[0]
    for i in range(1, len(versions_desc)):
        prev = versions_desc[i]
        if (
            keep <= 1
            or len(curr.release) > 2
            and curr.release[0:-1] == prev.release[0:-1]
            or prev.is_prerelease
            or prev.is_postrelease
        ):
            del possible_versions[prev]
        else:
            keep -= 1
            curr = prev


def _condensed_version_list(
    _subset_of_versions: List[pv.Version], _all_versions: List[pv.Version]
) -> vn.VersionList:
    subset = sorted(vn.StandardVersion.from_string(str(v)) for v in _subset_of_versions)
    all = sorted(vn.StandardVersion.from_string(str(v)) for v in _all_versions)

    # Find corresponding index
    i, j = all.index(subset[0]) + 1, 1
    new_versions: List[vn.ClosedOpenRange] = []

    # If the first when entry corresponds to the first known version, use (-inf, ..] as lowerbound.
    if i == 1:
        lo = vn.StandardVersion.typemin()
    else:
        lo = _best_lowerbound(all[i - 2], subset[0])

    while j < len(subset):
        if all[i] != subset[j]:
            hi = _best_upperbound(subset[j - 1], all[i])
            new_versions.append(vn.VersionRange(lo, hi))
            i = all.index(subset[j])
            lo = _best_lowerbound(all[i - 1], subset[j])
        i += 1
        j += 1

    # Similarly, if the last entry corresponds to the last known version,
    # assume the dependency continues to be used: [x, inf).
    if i == len(all):
        hi = vn.StandardVersion.typemax()
    else:
        hi = _best_upperbound(subset[j - 1], all[i])

    new_versions.append(vn.VersionRange(lo, hi))
    return vn.VersionList(new_versions)


def simplify_python_constraint(versions: vn.VersionList) -> None:
    """Modifies a version list of python versions in place to remove redundant constraints
    implied by UNSUPPORTED_PYTHON."""
    # First delete everything implied by UNSUPPORTED_PYTHON
    vs = versions.versions
    while vs and vs[0].satisfies(UNSUPPORTED_PYTHON):
        del vs[0]

    if not vs:
        return

    # Remove any redundant lowerbound, e.g. @3.7:3.9 becomes @:3.9 if @:3.6 unsupported.
    union = UNSUPPORTED_PYTHON._union_if_not_disjoint(vs[0])
    if union:
        vs[0] = union


def _populate(name: str, version_lookup: VersionsLookup, sqlite_cursor: sqlite3.Cursor) -> Node:
    dep_to_when: Dict[DepToWhen, Set[pv.Version]] = defaultdict(set)
    version_info: Dict[pv.Version, str] = {}

    query = sqlite_cursor.execute(
        """
        SELECT version, requires_dist, requires_python, sha256, path, is_sdist
        FROM versions
        WHERE name = ?""",
        (name,),
    )

    version_to_data = {
        v: (requires_dist, requires_python, sha256, path, sdist)
        for version, requires_dist, requires_python, sha256, path, sdist in query
        if (v := _acceptable_version(version))
    }

    # _delete_old_releases(version_to_data)

    for version, (
        requires_dist,
        requires_python,
        sha256_blob,
        path,
        sdist,
    ) in version_to_data.items():
        # Database cannot have duplicate versions.
        assert version not in version_info

        to_insert = []
        if requires_python:
            try:
                specifier_set = SpecifierSet(requires_python)
            except InvalidSpecifier:
                print(
                    f"{name}@{version}: invalid python specifier {requires_python}",
                    file=sys.stderr,
                )
                continue

            versions = _pkg_specifier_set_to_version_list("python", specifier_set, version_lookup)

            # First delete everything implied by UNSUPPORTED_PYTHON
            simplify_python_constraint(versions)

            if not versions:
                print(
                    f"{name}@{version}: no supported python versions: {requires_python}",
                    file=sys.stderr,
                )
                continue
            elif versions != vn.any_version:
                # Only emit non-trivial constraints on python.
                to_insert.append((("python", versions, None, None, frozenset()), version))

        for requirement_str in json.loads(requires_dist):
            try:
                r = Requirement(requirement_str)
            except InvalidRequirement:
                print(f"{name}@{version}: invalid requirement {requirement_str}", file=sys.stderr)
                continue

            child = _normalized_name(r.name)

            if r.marker is not None:
                result = _evaluate_marker(r.marker)
                if result is False:  # skip: statically unsatisfiable
                    continue
                elif result is True:  # unconditional depends_on
                    r.marker = None
                    result = None
                elif result is not None:  # list of when conditions: conditional depends_on
                    r.marker = None
            else:
                result = None

            # Emit an unconditional depends_on, or one or more conditional depends_on statements.
            for when in result or [None]:
                try:
                    versions = _pkg_specifier_set_to_version_list(
                        child, r.specifier, version_lookup
                    )
                except ValueError:
                    print(f"{name}@{version}: invalid specifier {r.specifier}", file=sys.stderr)
                    continue
                data = (child, versions, when, r.marker, frozenset(r.extras))
                to_insert.append((data, version))

        # Delay registering a version until we know that it's valid.
        for k, v in to_insert:
            dep_to_when[k].add(v)
        version_info[version] = ("".join(f"{x:02x}" for x in sha256_blob), path, sdist)

    # Next, simplify a list of specific version to a range if they are consecutive.
    ordered_versions = sorted(version_info.keys())

    # Translate the list of packaging versions to a list of Spack ranges.
    return Node(
        name,
        dep_to_when={
            k: _condensed_version_list(dep_to_when[k], ordered_versions) for k in dep_to_when
        },
        version_info=version_info,
        ordered_versions=ordered_versions,
    )


def _parse_without_trailing_zeros(version: str) -> vn.StandardVersion:
    """Parse as Spack version without trailing zeros, so "1.2.0" becomes "1.2"."""
    v = vn.StandardVersion.from_string(version)
    i = len(v)
    while i > 0 and v.version[i - 1] == 0:
        i -= 1
    return v if i == len(v) else v.up_to(i)


def _pkg_specifier_set_to_version_list(
    pkg: str, specifier_set: SpecifierSet, version_lookup: VersionsLookup
) -> vn.VersionList:
    # Turns out translating a specifier set to a version list is non-trivial and cannot be done
    # statically. Just >= and < are fine, and ~= which can be expressed in those terms:
    # ">=1.2" is "@1.2:"       note: ">=1.2.0" is "@1.2:" drop trailing zeros
    # "<1.2" is "@:1.1"        note: "<1.2.0" is "@:1.1" drop trailing zeros
    # "~1.2" is ">=1.2,<1.3"   note: "~1.2.0" is ">=1.2.0,<1.2.1" don't drop trailing zeros
    # "==1.2.*" is @1.2        note: "==1.2.0.*" is "@1.2.0.0" don't drop trailing zeros
    # "!=1.2.*" is @1.2        note: "!=1.2.0.*" is "@:1.1,1.2.1:" don't drop trailing zeros
    # The rest of the operators is problematic:
    # ">1.2"  isn't "@1.3:"    since "1.2.0.0.0.0.1" in Specifier(">1.2").
    # "<=1.2" isn't "@:1.2.0"  since "1.2.0.1" not in Specifier("<=1.2")
    # "==1.2" isn't "@1.2"     since "1.2.3" not in Specifier("==1.2")
    # "==1.2" isn't "@=1.2"    since "1.2.0" in Specifier("==1.2")
    # "!=1.2" means "<1.2,>1.2" so has the same issues as ">1.2".
    # For the latter, look up the matching version and rewrite in terms of ">=" and "<" relative
    # to that specific version.

    # TODO: what the hell is "===1.2" triple equals.

    out = vn.VersionList([":"])

    for specifier in specifier_set:
        # First the trivial specifiers
        if specifier.operator == ">=":
            v = _parse_without_trailing_zeros(specifier.version)
            new = [vn.VersionRange(v, vn.StandardVersion.typemax())]
        elif specifier.operator == "<":
            v = _parse_without_trailing_zeros(specifier.version)
            new = [vn.VersionRange(vn.StandardVersion.typemin(), prev_version_for_range(v))]
        elif specifier.operator == "~=":
            v = vn.StandardVersion.from_string(specifier.version)
            new = [vn.VersionRange(v, v.up_to(len(v) - 1))]
        elif specifier.version.endswith(".*") and specifier.operator == "==":
            v = vn.StandardVersion.from_string(specifier.version[:-2])
            new = [vn.VersionRange(v, v)]
        elif specifier.version.endswith(".*") and specifier.operator == "!=":
            v = vn.StandardVersion.from_string(specifier.version[:-2])
            new = [
                vn.VersionRange(vn.StandardVersion.typemin(), prev_version_for_range(v)),
                vn.VersionRange(vn._next_version(v), vn.StandardVersion.typemax()),
            ]
        else:
            # Then the specifiers that require a lookup.
            known_versions = version_lookup[pkg]
            v = pv.parse(specifier.version)

            if specifier.operator == ">" or specifier.operator == "<=":
                # Get the first index greater than v
                idx = bisect.bisect_right(known_versions, v)

                if specifier.operator == ">":
                    if idx < len(known_versions):
                        prev = vn.StandardVersion.from_string(str(known_versions[idx - 1]))
                        curr = vn.StandardVersion.from_string(str(known_versions[idx]))
                        lo = _best_lowerbound(prev, curr)
                        new = [vn.VersionRange(lo, vn.StandardVersion.typemax())]
                    else:
                        v = vn.StandardVersion.from_string(specifier.version)
                        new = [vn.VersionRange(vn._next_version(v), vn.StandardVersion.typemax())]
                else:
                    if 0 < idx < len(known_versions):
                        prev = vn.StandardVersion.from_string(str(known_versions[idx - 1]))
                        curr = vn.StandardVersion.from_string(str(known_versions[idx]))
                        hi = _best_upperbound(prev, curr)
                        new = [vn.VersionRange(vn.StandardVersion.typemin(), hi)]
                    else:
                        v = vn.StandardVersion.from_string(specifier.version)
                        new = [
                            vn.VersionRange(
                                vn.StandardVersion.typemin(), prev_version_for_range(v)
                            )
                        ]

            elif specifier.operator == "==" or specifier.operator == "!=":
                # Get the index where equality may hold.
                idx = bisect.bisect_left(known_versions, v)
                if idx != len(known_versions) and v == known_versions[idx]:
                    # If there is an exact match, use that (so ==3 matches 3.0, use 3.0)
                    spack_v = vn.StandardVersion.from_string(str(known_versions[idx]))
                else:
                    # Otherwise, stick to what was given literally
                    spack_v = vn.StandardVersion.from_string(specifier.version)

                if specifier.operator == "==":
                    new = [vn.VersionRange(spack_v, spack_v)]
                else:
                    new = [
                        vn.VersionRange(
                            vn.StandardVersion.typemin(), prev_version_for_range(spack_v)
                        ),
                        vn.VersionRange(vn._next_version(spack_v), vn.StandardVersion.typemax()),
                    ]

            else:
                raise ValueError(f"Not implemented: {specifier}")

        out = out.intersection(vn.VersionList(new))

    return out


def _make_depends_on_spec(name: str, version_list: vn.VersionList, extras: FrozenSet[str]) -> Spec:
    pkg_name = "python" if name == "python" else f"{SPACK_PREFIX}{name}"
    extras_variants = "".join(f"+{v}" for v in sorted(extras))
    spec = Spec(f"{pkg_name} {extras_variants}")
    spec.versions = version_list
    return spec


def _make_when_spec(spec: Optional[Spec], when_versions: vn.VersionList) -> Spec:
    spec = Spec() if spec is None else spec
    spec.versions.intersect(when_versions)
    return spec


def _format_when_spec(spec: Spec) -> str:
    parts = [spec.format("{name}{@versions}{variants}")]
    if spec.architecture:
        parts.append(f"platform={spec.platform}")
    for dep in spec.dependencies():
        parts.append(dep.format("^{name}{@versions}"))
    return " ".join(p for p in parts if p)


def _print_package(
    node: Node, defined_variants: Dict[str, Set[str]], f: io.StringIO = sys.stdout
) -> None:
    if not node.version_info:
        print("    # No versions available", file=f)
        print("    pass", file=f)
        print(file=f)
        return

    has_prereleases = any(v.is_prerelease for v in node.ordered_versions)

    for v in reversed(node.ordered_versions):
        sha256, path, sdist = node.version_info[v]
        preferred = "" if not has_prereleases or v.is_prerelease else ", preferred=True"
        print(
            f'    version("{v}", sha256="{sha256}", url="https://pypi.org/packages/{path}"{preferred})',
            file=f,
        )
    print(file=f)

    for variant in sorted(defined_variants.get(node.name, ())):
        print(f'    variant("{variant}", default=False)', file=f)
    print(file=f)

    # Then the depends_on bits.
    uncommented_lines: List[str] = []
    commented_lines: List[Tuple[str, str]] = []

    children = [
        (
            _make_depends_on_spec(name, version_list, extras),
            _make_when_spec(when_spec, when_versions),
            name,
            marker,
            extras,
        )
        for (
            name,
            version_list,
            when_spec,
            marker,
            extras,
        ), when_versions in node.dep_to_when.items()
    ]

    # Order by (python / not python, name ASC, when spec DESC, spec DESC)
    children.sort(key=lambda x: (x[0]), reverse=True)
    children.sort(key=lambda x: (x[1]), reverse=True)
    children.sort(key=lambda x: (x[0].name != "python", x[0].name))

    for child_spec, when_spec, name, marker, extras in children:

        if marker is not None:
            comment = f"marker: {marker}"
        else:
            comment = False

        if name == node.name:
            # TODO: could turn this into a requirement: requires("+x", when="@y")
            comment = "self-dependency"

        # Comment out a depends_on statement if the variants do not exist, or if there are
        # markers that we could not evaluate.
        if comment is False and defined_variants and name != "python":
            if (
                when_spec
                and when_spec.variants
                and not all(v in defined_variants[node.name] for v in when_spec.variants)
            ):
                comment = "variants statically unused"
            elif name not in defined_variants or not extras.issubset(defined_variants[name]):
                comment = "variants statically unused"

        when_str = _format_when_spec(when_spec)
        if when_str:
            line = f'depends_on("{child_spec}", when="{when_str}")'
        else:
            line = f'depends_on("{child_spec}")'
        if comment:
            commented_lines.append((line, comment))
        else:
            uncommented_lines.append(line)

    if uncommented_lines:
        print('    with default_args(type="run"):', file=f)
    elif commented_lines:
        print('    # with default_args(type="run"):', file=f)

    for line in uncommented_lines:
        print(f"        {line}", file=f)

    # Group commented lines by comment
    commented_lines.sort(key=lambda x: x[1])
    for comment, group in itertools.groupby(commented_lines, key=lambda x: x[1]):
        print(f"\n        # {comment}", file=f)
        for line, _ in group:
            print(f"        # {line}", file=f)

    print(file=f)


def _generate(pkg_name: str, extras: List[str], directory: Optional[str]) -> None:
    # Maps package name to (Node, seen_variants) tuples. The set of variants is those
    # variants that can possibly be turned on. It's intended to list a subset of the
    # variants defined by the package, as a means to omit variants like +test, +dev, and
    # +doc etc (or whatever the package author decided to call them) that are not required
    # by any of its dependents.
    packages: Dict[str, Tuple[Node, Set[str]]] = {}
    version_lookup = VersionsLookup(sqlite_cursor)

    # Queue is a list of (package, with_variants, depth) tuples. The set of variants is
    # those that its dependent required (or required from the command line for the root).
    queue: List[Tuple[str, Set[str], int]] = [(pkg_name, {*args.extras}, 0)]
    i = 1
    while queue:
        name, with_variants, depth = queue.pop()

        entry = packages.get(name)
        seen_before = entry is not None

        # Drop if we've already seen this package with the same variants enabled.
        if entry and with_variants.issubset(entry[1]):
            continue
        print(f"{i:4d}: {' ' * depth}{name}", file=sys.stderr)

        if seen_before:
            node, seen_variants = entry
            seen_variants.update(with_variants)
        else:
            node = _populate(name, version_lookup, sqlite_cursor)
            seen_variants = set(with_variants)
            packages[name] = (node, seen_variants)

        # If we have not seen this package before, we follow the unconditional edges
        # (i.e. those with a when clause that does not require any variants) and the
        # conditional ones that are enabled by the required variants.
        for child_name, _, when_spec, _, extras in node.dep_to_when.keys():
            if child_name == "python":
                continue
            if (
                not seen_before
                and (
                    # unconditional edges and conditional edges of all required
                    when_spec is None
                    or not when_spec.variants
                    or all(variant in seen_variants for variant in when_spec.variants)
                )
                or (
                    # conditional edges with new variants only
                    seen_before
                    and when_spec is not None
                    and when_spec.variants
                    and all(variant in seen_variants for variant in when_spec.variants)
                )
            ):
                queue.append((child_name, extras, depth + 1))

        i += 1

    # Simplify to a map from package name to a set of variants that are effectively used.
    defined_variants = {name: variants for name, (_, variants) in packages.items()}

    output_dir = pathlib.Path(directory or "pypi")
    packages_dir = output_dir / "packages"

    if not output_dir.exists():
        packages_dir.mkdir(parents=True)

    if not (output_dir / "repo.yaml").exists():
        with open(output_dir / "repo.yaml", "w") as f:
            f.write("repo:\n  namespace: python\n")

    for name, (node, _) in packages.items():
        spack_name = f"{SPACK_PREFIX}{name}"
        package_dir = packages_dir / spack_name
        package_dir.mkdir(parents=True, exist_ok=True)
        with open(package_dir / "package.py", "w") as f:
            print("from spack.package import *\n\n", file=f)
            print(f"class {mod_to_class(spack_name)}(PythonPackage):", file=f)
            _print_package(node, defined_variants, f)


def download_db():
    print("Downloading latest database (~500MB, may take a while...)", file=sys.stderr)
    with urllib.request.urlopen(DB_URL) as response, open("data.db", "wb") as f:
        with gzip.GzipFile(fileobj=response) as gz:
            shutil.copyfileobj(gz, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PyPI to Spack package.py", description="Convert PyPI data to Spack data"
    )
    parser.add_argument("--db", default="data.db", help="The database file to read from")
    subparsers = parser.add_subparsers(dest="command", help="The command to run")
    p_generate = subparsers.add_parser("generate", help="Generate a package.py file")
    p_generate.add_argument("--directory", "-o", help="Output directory")
    p_generate.add_argument("package", help="The package name on PyPI")
    p_generate.add_argument(
        "extras", nargs="*", help="Extras / variants to define on given package"
    )
    p_info = subparsers.add_parser("info", help="Show basic info about database or package")
    p_info.add_argument("package", nargs="?", help="package name on PyPI")
    p_update = subparsers.add_parser("update", help="Download the latest database")

    args = parser.parse_args()

    if args.command == "update":
        download_db()
        sys.exit(0)

    elif not os.path.exists(args.db):
        if input("Database does not exist, download? (y/n) ") not in ("y", "Y", "yes"):
            sys.exit(1)
        download_db()

    sqlite_connection = sqlite3.connect(args.db)
    sqlite_cursor = sqlite_connection.cursor()

    if args.command == "info":
        if args.package:
            node = _populate(_normalized_name(args.package), sqlite_cursor)
            variants = set(
                variant
                for _, _, when_spec, _, _ in node.dep_to_when.keys()
                if when_spec
                for variant in when_spec.variants
            )
            print("Normalized name:", node.name)
            print("Variants:", " ".join(sorted(variants)) if variants else "none")
            print("Total versions:", len(node.version_to_shasum))
        else:
            print(
                "Total packages:",
                sqlite_cursor.execute("SELECT COUNT(DISTINCT name) FROM versions").fetchone()[0],
            )
            print(
                "Total versions:",
                sqlite_cursor.execute("SELECT COUNT(*) FROM versions").fetchone()[0],
            )

    elif args.command == "generate":
        _generate(_normalized_name(args.package), args.extras, args.directory)
