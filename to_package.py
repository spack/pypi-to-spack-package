# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import sqlite3
import sys
import json
from packaging.requirements import Requirement
from packaging.specifiers import Specifier, SpecifierSet
from packaging.markers import Marker, Variable, Op, Value
from typing import Optional, FrozenSet
import packaging.version as pv
from spack.version.version_types import VersionStrComponent, prev_version_str_component
from typing import Dict, Tuple, List
from collections import defaultdict

import spack.version as vn

conn = sqlite3.connect("data.db")

c = conn.cursor()

# TODO: deal with re-uploads: should use upload_time. Now we just use GROUP BY.
result = c.execute(
    """
SELECT name, version, requires_dist, requires_python 
FROM packages
WHERE name LIKE ?
GROUP BY version""",
    (sys.argv[1],),
)


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

    return vn.StandardVersion(
        "".join(string_components), v.version[:-1] + (prev,), v.separators
    )


def specifier_to_spack_version(s: Specifier):
    # I think 1.2.* is only allowed with operators != and ==, in which case it can follow the
    # same code path.
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
                vn.VersionRange(
                    vn.StandardVersion.typemin(), prev_version_for_range(v)
                ),
                vn.VersionRange(vn.next_version(v), vn.StandardVersion.typemax()),
            ]
        )

    return v


def _eval_constraint(node: tuple):
    # Operator
    variable, comparison, value = node
    assert isinstance(variable, Variable)
    assert isinstance(comparison, Op)
    assert isinstance(value, Value)

    if (
        variable.value == "implementation_name"
        and comparison.value == "=="
        and value.value == "cpython"
    ):
        return vn.any_version

    if variable.value not in ("python_version", "python_full_version"):
        return None

    # Do everything in terms of ranges for simplicity.
    if comparison.value == "==":
        v = vn.StandardVersion.from_string(value.value)
        return vn.VersionList([vn.VersionRange(v, v)])
    elif comparison.value == ">":
        v = vn.StandardVersion.from_string(value.value)
        return vn.VersionList(
            [vn.VersionRange(vn.next_version(v), vn.StandardVersion.typemax())]
        )
    elif comparison.value == ">=":
        v = vn.StandardVersion.from_string(value.value)
        return vn.VersionList([vn.VersionRange(v, vn.StandardVersion.typemax())])
    elif comparison.value == "<":
        v = vn.StandardVersion.from_string(value.value)
        return vn.VersionList(
            [vn.VersionRange(vn.StandardVersion.typemin(), prev_version_for_range(v))]
        )
    elif comparison.value == "<=":
        v = vn.StandardVersion.from_string(value.value)
        return vn.VersionList([vn.VersionRange(vn.StandardVersion.typemin(), v)])
    elif comparison.value == "!=":
        v = vn.StandardVersion.from_string(value.value)
        return vn.VersionList(
            [
                vn.VersionRange(
                    vn.StandardVersion.typemin(), prev_version_for_range(v)
                ),
                vn.VersionRange(vn.next_version(v), vn.StandardVersion.typemax()),
            ]
        )
    else:
        # We don't support this comparison.
        return None


def _eval_node(node):
    if isinstance(node, tuple):
        return _eval_constraint(node)
    return _marker_to_python_constraint(node)


def _marker_to_python_constraint(node) -> Optional[vn.VersionList]:
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
    if lhs is None:
        return None

    # reduce
    for i in range(2, len(node), 2):
        op = node[i - 1]
        assert op in ("and", "or")
        rhs = _eval_node(node[i])
        if rhs is None:
            return None
        if op == "and":
            lhs.intersect(rhs)
        else:
            lhs.add(rhs)
    return lhs


def marker_to_python_constraint(m: Marker) -> Optional[vn.VersionList]:
    return _marker_to_python_constraint(m._markers)


def version_list_from_specifier(ss: SpecifierSet) -> vn.VersionList:
    versions = vn.any_version
    for s in ss:
        versions = versions.intersection(
            vn.VersionList([specifier_to_spack_version(s)])
        )
    return versions


dep_to_when: Dict[
    Tuple[str, vn.VersionList, Optional[Marker], FrozenSet[str]], vn.VersionList
] = defaultdict(vn.VersionList)
known_versions: List[vn.StandardVersion] = []

for name, version, requires_dist, requires_python in result:
    # We skip alpha/beta/rc etc releases, cause Spack's version ordering for them is wrong.
    packaging_version = pv.parse(version)
    if (
        packaging_version.pre is not None
        or packaging_version.dev is not None
        or packaging_version.post is not None
    ):
        continue

    spack_version = vn.StandardVersion.from_string(version)
    known_versions.append(spack_version)

    if requires_python:
        # Add the python dependency separately
        key = (
            "python",
            version_list_from_specifier(SpecifierSet(requires_python)),
            None,
            None,
            frozenset(),
        )
        dep_to_when[key].add(spack_version)

    for requirement_str in json.loads(requires_dist):
        r = Requirement(requirement_str)

        # Translate markers to ^python@ constraints if possible.
        if r.marker is not None:
            python_version = marker_to_python_constraint(r.marker)
            if python_version is not None:
                r.marker = None
        else:
            python_version = vn.any_version

        key = (
            r.name,
            version_list_from_specifier(r.specifier),
            python_version,
            r.marker,
            frozenset(r.extras),
        )

        dep_to_when[key].add(spack_version)

# Next, simplify a list of specific version to a range if they are consecutive.
known_versions.sort()

for key, when in dep_to_when.items():
    if when == vn.any_version:
        continue

    # It's guaranteed to be a sorted list of StandardVersion now.
    lo = when[0]

    # Find corresponding index
    i, j = known_versions.index(lo) + 1, 1
    new_list = []

    # If the first when entry corresponds to the first known version,
    # use (-inf, ..] as lowerbound.
    if i == 0:
        lo = vn.StandardVersion.typemin()

    while j < len(when):
        if known_versions[i] != when[j]:
            new_list.append(vn.VersionRange(lo, when[j - 1]))
            lo = when[j]
            i = known_versions.index(lo) + 1
        else:
            i += 1

        j += 1

    # Similarly, if the last entry corresponds to the last known version,
    # assume the dependency continues to be used: [x, inf).
    hi = vn.StandardVersion.typemax() if i == len(known_versions) else when[j - 1]
    new_list.append(vn.VersionRange(lo, hi))
    when.versions = new_list

if dep_to_when:
    print('with default_args(deptype=("build", "run")):')
    for k in sorted(
        dep_to_when.keys(),
        key=lambda x: (bool(x[3]), bool(x[4]), x[0] != "python", x[0], x[1], x[2]),
    ):
        name, version_list, python_version, marker, extras = k
        when = dep_to_when[k]
        version_list_str = "" if version_list == vn.any_version else f"@{version_list}"

        if marker is not None or extras:
            if marker is not None:
                print(f"\n    # marker: {marker}")
            if extras:
                print(f"\n    # extras: {','.join(extras)}")

        if python_version and python_version != vn.any_version:
            when_str = f"@{when} ^python@{python_version}"
        else:
            when_str = f"@{when}"

        variants = f' {" ".join(f"+{x}" for x in extras)}' if extras else ""
        comment = "# " if marker else ""
        spack_name = f"py-{name}" if name != "python" else "python"
        print(
            f'    {comment}depends_on("{spack_name}{version_list_str}{variants}", when="{when_str}")'
        )
