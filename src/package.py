# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import argparse
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
SPACK_PREFIX = "new-py-"


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
    # `python_version < "3"` is translated as `@:2`
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
        print(f"could not parse `{value}` as version", file=sys.stderr)
        return None

    if vv.is_prerelease or vv.is_postrelease or vv.is_devrelease or vv.epoch:
        print(f"dunno about: `{variable} {op} {value}`", file=sys.stderr)
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
    else:
        # We don't support this comparison.
        return None


def _eval_constraint(node: tuple, accept_extra: Callable[[str], bool]) -> Union[None, bool, Spec]:
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
                return Spec(f"+{value.value}") if accept_extra(value.value) else False
            elif op.value == "!=":
                return Spec(f"~{value.value}") if accept_extra(value.value) else True
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
    return spec


def _eval_node(node, accept_extra: Callable[[str], bool]) -> Union[None, bool, Spec]:
    if isinstance(node, tuple):
        return _eval_constraint(node, accept_extra)
    return _marker_to_spec(node, accept_extra)


def _marker_to_spec(node: list, accept_extra: Callable[[str], bool]) -> Union[None, bool, Spec]:
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

    lhs = _eval_node(node[0], accept_extra)

    for i in range(2, len(node), 2):
        # Actually op should be constant: x and y and z. we don't assert it here.
        op = node[i - 1]
        assert op in ("and", "or")
        if op == "and":
            if lhs is False:
                return False
            rhs = _eval_node(node[i], accept_extra)
            if rhs is False:
                return False
            elif lhs is None or rhs is None:
                lhs = None
            elif lhs is True:
                lhs = rhs
            elif rhs is not True:  # Intersection of specs
                try:
                    lhs.constrain(rhs)
                except UnsatisfiableSpecError:
                    # This happens when people have no clue what they're doing, and such people
                    # exist. E.g. python_version > "3" and python_version < "3.11" is
                    # unsatisfiable.
                    return False
        elif op == "or":
            if lhs is True:
                return True
            rhs = _eval_node(node[i], accept_extra)
            if rhs is True:
                return True
            elif lhs is None or rhs is None:
                lhs = None
            elif lhs is False:
                lhs = rhs
            elif rhs is not False:
                # Union: currently only python versions can be unioned. The rest would need
                # multiple depends_on statements -- not supported yet.
                if lhs.variants or rhs.variants:
                    return None
                p_lhs, p_rhs = lhs.dependencies("python"), rhs.dependencies("python")
                if not (p_lhs and p_rhs):
                    return None
                p_lhs[0].versions.add(p_rhs[0].versions)
    return lhs


def marker_to_spec(m: Marker, accept_extra: Callable[[str], bool]) -> Union[bool, None, Spec]:
    """Evaluate the marker expression tree either (1) as a Spack spec if possible, (2) statically
    as True or False given that we only support cpython, (3) None if we can't translate it into
    Spack DSL."""
    # TODO: simplify expression we can evaluate statically partially.
    return _marker_to_spec(m._markers, accept_extra)


def version_list_from_specifier(ss: SpecifierSet) -> vn.VersionList:
    versions = vn.any_version
    for s in ss:
        versions = versions.intersection(vn.VersionList([specifier_to_spack_version(s)]))
    return versions


def dep_sorting_key(dep):
    """Sensible ordering key when emitting depends_on statements."""
    name, version_list, when_spec, marker, extras = dep
    return (name != "python", name, version_list, when_spec)


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


DepToWhen = Tuple[str, vn.VersionList, Optional[Spec], Optional[Marker], FrozenSet[str]]


class Node:
    __slots__ = ("name", "dep_to_when", "version_to_shasum")

    def __init__(
        self,
        name: str,
        dep_to_when: Dict[DepToWhen, vn.VersionList],
        version_to_shasum: Dict[vn.StandardVersion, str],
    ):
        self.name = name
        self.dep_to_when = dep_to_when
        self.version_to_shasum = version_to_shasum


def acceptable_version(version: str) -> Optional[vn.StandardVersion]:
    """Attempt to parse as a version (using packaging and Spack), and if valid, return it."""
    try:
        v = pv.parse(version)
        # Todo: epoch?
        if v.is_prerelease or v.is_postrelease:
            return None
    except pv.InvalidVersion:
        return None

    try:
        return vn.StandardVersion.from_string(version)
    except ValueError:
        return None


def populate(name: str, sqlite_cursor: sqlite3.Cursor) -> Node:
    dep_to_when: Dict[DepToWhen, vn.VersionList] = defaultdict(vn.VersionList)
    version_to_shasum: Dict[vn.StandardVersion, str] = {}

    query = sqlite_cursor.execute(
        """
        SELECT version, requires_dist, requires_python, sha256 
        FROM versions
        WHERE name = ?""",
        (name,),
    )

    possible_versions = {
        spack_version: (requires_dist, requires_python, sha256)
        for version, requires_dist, requires_python, sha256 in query
        if (spack_version := acceptable_version(version))
    }

    # Now drop old patch version numbers
    if possible_versions:
        sorted_versions = sorted(possible_versions.keys())
        prev = sorted_versions[0]
        for i in range(1, len(sorted_versions)):
            curr = sorted_versions[i]
            if len(curr) == 3 and curr.version[0:2] == prev.version[0:2]:
                del possible_versions[prev]
            prev = curr

    for spack_version, (requires_dist, requires_python, sha256_blob) in possible_versions.items():
        # Database should only contain the latest version of a package.
        assert spack_version not in version_to_shasum

        to_insert = []
        if requires_python:
            # This is the "raw" version list.
            try:
                specifier_set = SpecifierSet(requires_python)
            except InvalidSpecifier:
                print(f"{name}: invalid python specifier {requires_python}", file=sys.stderr)
                continue

            python_ver = version_list_from_specifier(specifier_set)

            # First drop any unsupported versions.
            python_ver.versions = [v for v in python_ver if not v.satisfies(UNSUPPORTED_PYTHON)]

            # If there is at least one condition and the union is not the entire version space,
            # then there is a non-trivial constraint on python we need to emit.
            if python_ver.versions:
                # Finally,
                union_with_unsupported = vn.VersionList()
                union_with_unsupported.versions[:] = python_ver.versions
                union_with_unsupported.add(UNSUPPORTED_PYTHON)
                if union_with_unsupported != vn.any_version:
                    to_insert.append(
                        (("python", python_ver, None, None, frozenset()), spack_version)
                    )

        for requirement_str in json.loads(requires_dist):
            r = Requirement(requirement_str)

            # Translate markers to ^python@ constraints if possible.
            if r.marker is not None:
                try:
                    marker_when_spec = marker_to_spec(r.marker, lambda variant: True)
                except Exception as e:
                    print(
                        f"{name}: broken marker {r.marker}: {e.__class__.__name__}: {e}",
                        file=sys.stderr,
                    )
                    raise
                if marker_when_spec is False:
                    # Statically evaluate to False: do not emit depends_on.
                    continue
                elif marker_when_spec is True:
                    # Statically evaluated to True: emit unconditional depends_on.
                    r.marker = None
                    marker_when_spec = None
                if marker_when_spec is not None:
                    # Translated to a Spec: conditional depends_on.
                    r.marker = None
            else:
                marker_when_spec = None

            to_insert.append(
                (
                    (
                        normalized_name(r.name),
                        version_list_from_specifier(r.specifier),
                        marker_when_spec,
                        r.marker,
                        frozenset(r.extras),
                    ),
                    spack_version,
                )
            )

        # Delay registering a version until we know that it's valid.
        for k, v in to_insert:
            dep_to_when[k].add(v)
        version_to_shasum[spack_version] = "".join(f"{x:02x}" for x in sha256_blob)

    # Next, simplify a list of specific version to a range if they are consecutive.
    known_versions = sorted(version_to_shasum.keys())

    for when in dep_to_when.values():
        if when == vn.any_version:
            continue

        # Find corresponding index
        i, j = known_versions.index(when[0]) + 1, 1
        new_versions: List[vn.ClosedOpenRange] = []

        # If the first when entry corresponds to the first known version,
        # use (-inf, ..] as lowerbound.
        if i == 1:
            lo = vn.StandardVersion.typemin()
        else:
            lo = best_lowerbound(known_versions[i - 2], when[0])

        while j < len(when):
            if known_versions[i] != when[j]:
                # Not consecutive: emit a range.
                new_versions.append(
                    vn.VersionRange(lo, best_upperbound(when[j - 1], known_versions[i]))
                )
                i = known_versions.index(when[j])
                lo = best_lowerbound(known_versions[i - 1], when[j])
            i += 1
            j += 1

        # Similarly, if the last entry corresponds to the last known version,
        # assume the dependency continues to be used: [x, inf).
        if i == len(known_versions):
            hi = vn.StandardVersion.typemax()
        else:
            hi = best_upperbound(when[j - 1], known_versions[i])

        new_versions.append(vn.VersionRange(lo, hi))
        when.versions = new_versions
    return Node(name, dep_to_when=dep_to_when, version_to_shasum=version_to_shasum)


def print_package(
    node: Node, defined_variants: Optional[Dict[str, Set[str]]] = None, f: io.StringIO = sys.stdout
) -> None:
    """
    Arguments:
        node: package to print
        defined_variants: a mapping from package name to a set of variants that are effectively
            used. If provided, this function will emit only variant(...) statements for those, and
            omit any depends_on statements that are statically unsatisfiable.
    """
    if not node.version_to_shasum:
        print("    # No sdist available", file=f)
        print("    pass", file=f)
        print(file=f)
        return

    known_versions = sorted(node.version_to_shasum.keys())

    for v in sorted(known_versions, reverse=True):
        print(f'    version("{v}", sha256="{node.version_to_shasum[v]}")', file=f)
    print(file=f)

    # TODO: if defined_variants is not provided, infer from node.dep_to_when.keys().
    if defined_variants:
        for variant in sorted(defined_variants.get(node.name, ())):
            print(f'    variant("{variant}", default=False)', file=f)
        print(file=f)

    # Then the depends_on bits.
    uncommented_lines: List[str] = []
    commented_lines: List[Tuple[str, str]] = []
    for k in sorted(node.dep_to_when.keys(), key=dep_sorting_key):
        child, version_list, when_spec, marker, extras = k
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
        dep_spec.versions = version_list
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

    for name, (node, _) in packages.items():
        spack_name = f"{SPACK_PREFIX}{name}"
        package_dir = packages_dir / spack_name
        package_dir.mkdir(parents=True, exist_ok=True)
        with open(package_dir / "package.py", "w") as f:
            print("from spack.package import *\n\n", file=f)
            print(f"class {mod_to_class(spack_name)}(PythonPackage):", file=f)
            print_package(node, defined_variants, f)


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
