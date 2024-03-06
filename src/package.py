# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import argparse
import bisect
import io
import itertools
import json
import os
import pathlib
import re
import sqlite3
import sys
from collections import defaultdict
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import packaging.version as pv
import spack.version as vn
from packaging.markers import Marker, Op, Value, Variable
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from spack.error import UnsatisfiableSpecError
from spack.parser import SpecSyntaxError
from spack.spec import Spec
from spack.util.naming import mod_to_class
from spack.version.version_types import VersionStrComponent, prev_version_str_component

# If a marker on python version satisfies this range, we statically evaluate it as true.
UNSUPPORTED_PYTHON = vn.from_string(":3.6")

# The prefix to use for Pythohn package names in Spack.
SPACK_PREFIX = "pypi-"


class VersionsLookup:
    def __init__(self, cursor: sqlite3.Cursor):
        self.cursor = cursor
        self.cache: Dict[str, List[pv.Version]] = {}

    def __getitem__(self, name: str) -> List[pv.Version]:
        result = self.cache.get(name)
        if result is not None:
            return result
        query = self.cursor.execute(
            """
            SELECT version
            FROM versions
            WHERE name = ?""",
            (name,),
        )
        result = sorted(vv for v, in query if (vv := acceptable_version(v)))
        self.cache[name] = result
        return result


def prev_version_for_range(v: vn.StandardVersion) -> vn.StandardVersion:
    """Translate Specifier <x into a Spack range upperbound :y"""
    # TODO: <0 is broken.
    if len(v.version) == 0:
        return v
    elif isinstance(v.version[-1], VersionStrComponent):
        prev = prev_version_str_component(v.version[-1])
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


def specifier_to_spack_version(s: Specifier):
    # The "version" 1.2.* is only allowed with operators != and ==, in which case it can follow the
    # same code path. However, the PyPI index is filled with >=1.2.* nonsense -- ignore it, it
    # would error in the else branch anyways as * is not a valid version component in Spack.
    if s.version.endswith(".*") and s.operator in ("!=", "=="):
        v = vn.StandardVersion.from_string(s.version[:-2])
    else:
        v = vn.StandardVersion.from_string(s.version)

    if s.operator == ">=":
        return vn.VersionRange(v, vn.StandardVersion.typemax())
    elif s.operator == ">":
        return vn.VersionRange(vn.next_version(v), vn.StandardVersion.typemax())
    elif s.operator == "<=":
        return vn.VersionRange(vn.StandardVersion.typemin(), v)
    elif s.operator == "<":
        return vn.VersionRange(vn.StandardVersion.typemin(), prev_version_for_range(v))
    elif s.operator == "~=":
        return vn.VersionRange(v, v.up_to(len(v) - 1))
    elif s.operator == "==":
        return vn.VersionRange(v, v)
    elif s.operator == "!=":
        return vn.VersionList(
            [
                vn.VersionRange(vn.StandardVersion.typemin(), prev_version_for_range(v)),
                vn.VersionRange(vn.next_version(v), vn.StandardVersion.typemax()),
            ]
        )

    return v


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
        return vn.VersionList([vn.VersionRange(vn.next_version(v), vn.StandardVersion.typemax())])
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
                vn.VersionRange(vn.next_version(v), vn.StandardVersion.typemax()),
            ]
        )
    print(f"cannot deal with operator: `{variable} {op} {value}`", file=sys.stderr)
    return None


def _eval_constraint(node: tuple) -> Union[None, bool, List[Spec]]:
    # TODO: os_name, sys_platform, platform_machine, platform_release, platform_system,
    # platform_version, implementation_version

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

    if versions.satisfies(UNSUPPORTED_PYTHON):
        return False

    spec = Spec("^python")
    spec.dependencies("python")[0].versions = versions
    return [spec]


def _eval_node(node) -> Union[None, bool, List[Spec]]:
    if isinstance(node, tuple):
        return _eval_constraint(node)
    return _evaluate_marker(node)


def intersection(lhs: List[Spec], rhs: List[Spec]) -> List[Spec]:
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


def union(lhs: List[Spec], rhs: List[Spec]) -> List[Spec]:
    """This case is trivial: (a or b) or (c or d) = a or b or c or d, BUT do a simplification
    in case the rhs only expresses constraints on versions."""
    if len(rhs) == 1 and not rhs[0].variants:
        python, *_ = rhs[0].dependencies("python")
        for l in lhs:
            l.versions.add(python.versions)
        return lhs

    return list(set(lhs + rhs))


def _evaluate_marker(node: list) -> Union[None, bool, List[Spec]]:
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
                lhs = intersection(lhs, rhs)
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
                lhs = union(lhs, rhs)
    return lhs


def evaluate_marker(m: Marker) -> Union[bool, None, List[Spec]]:
    """Evaluate the marker expression tree either (1) as a list of specs that constitute the when
    conditions, (2) statically as True or False given that we only support cpython, (3) None if
    we can't translate it into Spack DSL."""
    return _evaluate_marker(m._markers)


def version_list_from_specifier(ss: SpecifierSet) -> vn.VersionList:
    # Note: this is only correct for python versions where we can assume some semver like
    # semantics. Can't use this in requirements in general.
    versions = vn.any_version
    for s in ss:
        versions = versions.intersection(vn.VersionList([specifier_to_spack_version(s)]))
    return versions


def dep_sorting_key(dep):
    """Sensible ordering key when emitting depends_on statements."""
    name, _, when_spec, _, _ = dep
    return (name != "python", name, when_spec)


NAME_REGEX = re.compile(r"[-_.]+")


def normalized_name(name):
    return re.sub(NAME_REGEX, "-", name).lower()


def best_upperbound(curr: vn.StandardVersion, next: vn.StandardVersion) -> vn.StandardVersion:
    """Return the most general upperound that includes curr but not next.
    Invariant is that curr < next."""

    # i is the first index where the two versions differ.
    i = 1
    m = min(len(curr), len(next))
    while i < m and curr.version[i] == next.version[i]:
        i += 1

    # Pad with ".0"
    if i > m:
        return vn.StandardVersion.from_string(f"{curr}{(len(next) - len(curr)) * '.0'}")

    # Truncate if necessary
    return curr if i == m else curr.up_to(i + 1)


def best_lowerbound(prev: vn.StandardVersion, curr: vn.StandardVersion) -> vn.StandardVersion:
    i = 1
    m = min(len(prev), len(curr))
    while i < m and prev.version[i] == curr.version[i]:
        i += 1

    # if prev is a prefix of curr, curr must have an additional component.
    # if not a prefix, truncate on the first differing component.
    return curr if i == m else curr.up_to(i + 1)


DepToWhen = Tuple[str, SpecifierSet, Optional[Spec], Optional[Marker], FrozenSet[str]]


class Node:
    __slots__ = ("name", "dep_to_when", "version_to_shasum", "ordered_versions")

    def __init__(
        self,
        name: str,
        dep_to_when: Dict[DepToWhen, vn.VersionList],
        version_to_shasum: Dict[pv.Version, str],
        ordered_versions: List[pv.Version],
    ):
        self.name = name
        self.dep_to_when = dep_to_when
        self.version_to_shasum = version_to_shasum
        self.ordered_versions = ordered_versions


def acceptable_version(version: str) -> Optional[pv.Version]:
    """Maybe parse with packaging"""
    try:
        v = pv.parse(version)
    except pv.InvalidVersion:
        return None
    if v.is_prerelease or v.is_postrelease:
        return None
    return v


def delete_old_patch_releases(
    defined_versions: List[pv.Version], possible_versions: Dict[pv.Version, Tuple[str, str, bytes]]
) -> None:
    """Reduce the number of version definitions by just considering the latest patch release."""
    if not possible_versions:
        return
    prev = defined_versions[0]
    for i in range(1, len(defined_versions)):
        curr = defined_versions[i]
        if len(curr.release) == 3 and curr.release[0:2] == prev.release[0:2]:
            del possible_versions[prev]
        prev = curr


def condensed_version_list(
    _version_list: List[pv.Version], _ordered_versions: List[pv.Version]
) -> vn.VersionList:
    version_list = [vn.StandardVersion.from_string(str(v)) for v in _version_list]
    ordered_versions = [vn.StandardVersion.from_string(str(v)) for v in _ordered_versions]

    version_list.sort()

    # Find corresponding index
    i, j = ordered_versions.index(version_list[0]) + 1, 1
    new_versions: List[vn.ClosedOpenRange] = []

    # If the first when entry corresponds to the first known version, use (-inf, ..] as lowerbound.
    if i == 1:
        lo = vn.StandardVersion.typemin()
    else:
        lo = best_lowerbound(ordered_versions[i - 2], version_list[0])

    while j < len(version_list):
        if ordered_versions[i] != version_list[j]:
            hi = best_upperbound(version_list[j - 1], ordered_versions[i])
            new_versions.append(vn.VersionRange(lo, hi))
            i = ordered_versions.index(version_list[j])
            lo = best_lowerbound(ordered_versions[i - 1], version_list[j])
        i += 1
        j += 1

    # Similarly, if the last entry corresponds to the last known version,
    # assume the dependency continues to be used: [x, inf).
    if i == len(ordered_versions):
        hi = vn.StandardVersion.typemax()
    else:
        hi = best_upperbound(version_list[j - 1], ordered_versions[i])

    new_versions.append(vn.VersionRange(lo, hi))
    return vn.VersionList(new_versions)


def populate(name: str, sqlite_cursor: sqlite3.Cursor) -> Node:
    dep_to_when: Dict[DepToWhen, List[pv.Version]] = defaultdict(list)
    version_to_shasum: Dict[pv.Version, str] = {}

    query = sqlite_cursor.execute(
        """
        SELECT version, requires_dist, requires_python, sha256 
        FROM versions
        WHERE name = ?""",
        (name,),
    )

    version_to_data = {
        v: (requires_dist, requires_python, sha256)
        for version, requires_dist, requires_python, sha256 in query
        if (v := acceptable_version(version))
    }

    all_versions = sorted(version_to_data.keys())

    delete_old_patch_releases(all_versions, version_to_data)

    for version, (requires_dist, requires_python, sha256_blob) in version_to_data.items():
        # Database cannot have duplicate versions.
        assert version not in version_to_shasum

        to_insert = []
        if requires_python:
            try:
                specifier_set = SpecifierSet(requires_python)
            except InvalidSpecifier:
                print(f"{name}: invalid python specifier {requires_python}", file=sys.stderr)
                continue

            to_insert.append((("python", specifier_set, None, None, frozenset()), version))

            # python_ver = version_list_from_specifier(specifier_set)

            # # First drop any unsupported versions.
            # python_ver.versions = [v for v in python_ver if not v.satisfies(UNSUPPORTED_PYTHON)]

            # # If there is at least one condition and the union is not the entire version space,
            # # then there is a non-trivial constraint on python we need to emit.
            # if python_ver.versions:
            #     # Finally,
            #     union_with_unsupported = vn.VersionList()
            #     union_with_unsupported.versions[:] = python_ver.versions
            #     union_with_unsupported.add(UNSUPPORTED_PYTHON)
            #     if union_with_unsupported != vn.any_version:

        for requirement_str in json.loads(requires_dist):
            r = Requirement(requirement_str)

            if r.marker is not None:
                result = evaluate_marker(r.marker)
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
                data = (normalized_name(r.name), r.specifier, when, r.marker, frozenset(r.extras))
                to_insert.append((data, version))

        # Delay registering a version until we know that it's valid.
        for k, v in to_insert:
            dep_to_when[k].append(v)
        version_to_shasum[version] = "".join(f"{x:02x}" for x in sha256_blob)

    # Next, simplify a list of specific version to a range if they are consecutive.
    ordered_versions = sorted(version_to_shasum.keys())

    # Translate the list of packaging versions to a list of Spack ranges.
    return Node(
        name,
        dep_to_when={
            k: condensed_version_list(dep_to_when[k], ordered_versions) for k in dep_to_when
        },
        version_to_shasum=version_to_shasum,
        ordered_versions=ordered_versions,
    )


def parse_without_trailing_zeros(version: str) -> vn.StandardVersion:
    """Parse as Spack version without trailing zeros, so "1.2.0" becomes "1.2"."""
    v = vn.StandardVersion.from_string(version)
    i = len(v)
    while i > 0 and v.version[i - 1] == 0:
        i -= 1
    return v if i == len(v) else v.up_to(i)


def pkg_specifier_set_to_version_list(
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
            v = parse_without_trailing_zeros(specifier.version)
            new = [vn.VersionRange(v, vn.StandardVersion.typemax())]
        elif specifier.operator == "<":
            v = parse_without_trailing_zeros(specifier.version)
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
                vn.VersionRange(vn.next_version(v), vn.StandardVersion.typemax()),
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
                        lo = best_lowerbound(prev, curr)
                        new = [vn.VersionRange(lo, vn.StandardVersion.typemax())]
                    else:
                        v = vn.StandardVersion.from_string(specifier.version)
                        new = [vn.VersionRange(vn.next_version(v), vn.StandardVersion.typemax())]
                else:
                    if 0 < idx < len(known_versions):
                        prev = vn.StandardVersion.from_string(str(known_versions[idx - 1]))
                        curr = vn.StandardVersion.from_string(str(known_versions[idx]))
                        hi = best_upperbound(prev, curr)
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
                        vn.VersionRange(vn.next_version(spack_v), vn.StandardVersion.typemax()),
                    ]

            else:
                raise ValueError(f"Not implemented: {specifier}")

        out = out.intersection(vn.VersionList(new))

    return out


def print_package(
    node: Node,
    defined_variants: Dict[str, Set[str]],
    version_lookup: VersionsLookup,
    f: io.StringIO = sys.stdout,
) -> None:
    if not node.version_to_shasum:
        print("    # No sdist available", file=f)
        print("    pass", file=f)
        print(file=f)
        return

    for v in reversed(node.ordered_versions):
        print(f'    version("{v}", sha256="{node.version_to_shasum[v]}")', file=f)
    print(file=f)

    for variant in sorted(defined_variants.get(node.name, ())):
        print(f'    variant("{variant}", default=False)', file=f)
    print(file=f)

    # Then the depends_on bits.
    uncommented_lines: List[str] = []
    commented_lines: List[Tuple[str, str]] = []
    for k in sorted(node.dep_to_when.keys(), key=dep_sorting_key):
        child, specifierset, when_spec, marker, extras = k
        when = node.dep_to_when[k]

        if marker is not None:
            comment = f"marker: {marker}"
        else:
            comment = False

        when_spec = Spec() if when_spec is None else when_spec
        when_spec.versions.intersect(when)

        if when_spec == Spec("@:"):
            when_str = ""
        else:
            when_str = f', when="{when_spec}"'

        # Comment out a depends_on statement if the variants do not exist, or if there are
        # markers that we could not evaluate.
        if comment is False and defined_variants and child != "python":
            if (
                when_spec
                and when_spec.variants
                and not all(v in defined_variants[node.name] for v in when_spec.variants)
            ):
                comment = "variants statically unused"
            elif child not in defined_variants or not extras.issubset(defined_variants[child]):
                comment = "variants statically unused"

        pkg_name = "python" if child == "python" else f"{SPACK_PREFIX}{child}"
        extras_variants = "".join(f"+{v}" for v in sorted(extras))
        dep_spec = Spec(f"{pkg_name} {extras_variants}")

        if child != "python":
            dep_spec.versions = pkg_specifier_set_to_version_list(
                child, specifierset, version_lookup
            )
        line = f'depends_on("{dep_spec}"{when_str})'
        if comment:
            commented_lines.append((line, comment))
        else:
            uncommented_lines.append(line)

    if uncommented_lines:
        print('    with default_args(type=("build", "run")):', file=f)
        for line in uncommented_lines:
            print(f"        {line}", file=f)

    # Group commented lines by comment
    commented_lines.sort(key=lambda x: x[1])
    for comment, group in itertools.groupby(commented_lines, key=lambda x: x[1]):
        print(f"\n        # {comment}", file=f)
        for line, _ in group:
            print(f"        # {line}", file=f)

    print(file=f)


def generate(pkg_name: str, extras: List[str]) -> None:
    # Maps package name to (Node, seen_variants) tuples. The set of variants is those
    # variants that can possibly be turned on. It's intended to list a subset of the
    # variants defined by the package, as a means to omit variants like +test, +dev, and
    # +doc etc (or whatever the package author decided to call them) that are not required
    # by any of its dependents.
    packages: Dict[str, Tuple[Node, Set[str]]] = {}

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
            node = populate(name, sqlite_cursor)
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

    packages_dir = pathlib.Path("pypi", "packages")
    packages_dir.mkdir(parents=True, exist_ok=True)

    with open(packages_dir / ".." / "repo.yaml", "w") as f:
        f.write("repo:\n  namespace: python\n")

    version_lookup = VersionsLookup(sqlite_cursor)

    for name, (node, _) in packages.items():
        spack_name = f"{SPACK_PREFIX}{name}"
        package_dir = packages_dir / spack_name
        package_dir.mkdir(parents=True, exist_ok=True)
        with open(package_dir / "package.py", "w") as f:
            print("from spack.package import *\n\n", file=f)
            print(f"class {mod_to_class(spack_name)}(PythonPackage):", file=f)
            print('    url = "https://www.example.com/file.tar.gz"\n', file=f)
            print_package(node, defined_variants, version_lookup, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PyPI to Spack package.py", description="Convert PyPI data to Spack data"
    )
    parser.add_argument("--db", default="data.db", help="The database file to read from")
    subparsers = parser.add_subparsers(dest="command", help="The command to run")
    p_generate = subparsers.add_parser("generate", help="Generate a package.py file")
    p_generate.add_argument("package", help="The package name on PyPI")
    p_generate.add_argument(
        "extras", nargs="*", help="Extras / variants to define on given package"
    )
    p_info = subparsers.add_parser("info", help="Show basic info about database or package")
    p_info.add_argument("package", nargs="?", help="package name on PyPI")

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database file {args.db} does not exist", file=sys.stderr)
        sys.exit(1)

    sqlite_connection = sqlite3.connect(args.db)
    sqlite_cursor = sqlite_connection.cursor()

    if args.command == "info":
        if args.package:
            node = populate(normalized_name(args.package), sqlite_cursor)
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
        generate(normalized_name(args.package), args.extras)
