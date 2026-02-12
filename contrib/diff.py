#!/usr/bin/env spack-python

import argparse
import json
from collections import defaultdict
from typing import Dict, List

import spack.paths
import spack.repo
from spack.dependency import Dependency
from spack.deptypes import LINK, RUN
from spack.spec import Spec
from spack.version import StandardVersion, VersionList, VersionRange, infinity_versions

MOVE_UP = "\033[1A"
CLEAR_LINE = "\x1b[2K"


def to_version_list(versions, all_versions):
    vlist = []
    i = 1
    j = all_versions.index(versions[0]) + 1
    start = StandardVersion.typemin() if j == 1 else versions[0]

    while i < len(versions):
        if versions[i] != all_versions[j]:
            vlist.append(VersionRange(start, versions[i - 1]))
            start = versions[i]
            j = all_versions.index(versions[i])
        i += 1
        j += 1

    last = StandardVersion.typemax() if j == len(all_versions) else versions[-1]
    vlist.append(VersionRange(start, last))

    return VersionList(vlist)


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(required=True, dest="command")
sub.add_parser(
    "before", help="Generate before.json (checkout commit before changes yourself)"
)
sub.add_parser(
    "after", help="Generate after.json (checkout commit after changes yourself)"
)
sub.add_parser("diff", help="Diff before.json and after.json")

args = parser.parse_args()

if args.command in ("before", "after"):
    possible_versions: Dict[
        str, Dict[StandardVersion, Dict[str, List[StandardVersion]]]
    ] = {}
    pkgs = spack.repo.PATH.all_package_names()
    print()
    for i, name in enumerate(pkgs):
        percent = int((i + 1) / len(pkgs) * 100)
        print(f"{MOVE_UP}{CLEAR_LINE}[{percent:3}%] {name}")
        s = Spec(name)
        s._mark_concrete()
        pkg = s.package
        v: StandardVersion
        when: Spec
        edge: Dict[str, Dependency]

        possible_versions[name] = {}

        for v in pkg.versions:
            deps_for_version: Dict[str, List[StandardVersion]] = {}

            for when, edge in pkg.dependencies.items():
                if not v.satisfies(when.versions):
                    continue
                for dep_name, dep in edge.items():
                    if dep.depflag & (LINK | RUN) == 0:
                        continue
                    if spack.repo.PATH.is_virtual(dep_name):
                        continue
                    child = Spec(dep_name)
                    child._mark_concrete()
                    if dep_name not in deps_for_version:
                        deps_for_version[dep_name] = list(child.package.versions.keys())
                    # filter matching versions
                    deps_for_version[dep_name] = [
                        v
                        for v in deps_for_version[dep_name]
                        if v.satisfies(dep.spec.versions)
                    ]

            possible_versions[name][str(v)] = {
                dep_name: [str(w) for w in versions]
                for dep_name, versions in deps_for_version.items()
            }

    print(f"{MOVE_UP}{CLEAR_LINE}[100%] done.")

    with open(f"{args.command}.json", "w") as f:
        json.dump(possible_versions, f, sort_keys=True)

elif args.command == "diff":
    a = json.load(open("before.json"))
    b = json.load(open("after.json"))

    # inf_versions = {StandardVersion.from_string(x) for x in infinity_versions}
    inf_versions = set()

    for name in spack.repo.PATH.all_package_names():
        if name not in a or name not in b:
            continue
        a_pkg = a[name]
        b_pkg = b[name]

        versions_in_a = (
            set(StandardVersion.from_string(x) for x in a_pkg.keys()) - inf_versions
        )
        versions_in_b = (
            set(StandardVersion.from_string(x) for x in b_pkg.keys()) - inf_versions
        )

        change_to_version = defaultdict(list)

        if versions_in_a != versions_in_b:
            a_min_b = versions_in_a - versions_in_b
            b_min_a = versions_in_b - versions_in_a
            if a_min_b:
                change_to_version[
                    f"removed versions {', '.join(f'`{x}`' for x in sorted(a_min_b))}"
                ] = True
            if b_min_a:
                change_to_version[
                    f"added versions {', '.join(f'`{x}`' for x in sorted(b_min_a))}"
                ] = True

        for v in sorted(versions_in_a & versions_in_b):
            deps_a = next(
                val
                for key, val in a_pkg.items()
                if StandardVersion.from_string(key) == v
            )
            deps_b = next(
                val
                for key, val in b_pkg.items()
                if StandardVersion.from_string(key) == v
            )

            deps_in_a = set(deps_a.keys())
            deps_in_b = set(deps_b.keys())

            changes_for_version = []

            if deps_in_a != deps_in_b:
                if deps_in_a - deps_in_b - {"py-setuptools"}:
                    changes_for_version.append(
                        f"deleted {', '.join(f'^{d}' for d in sorted(deps_in_a - deps_in_b))}"
                    )
                if deps_in_b - deps_in_a:
                    changes_for_version.append(
                        f"added {', '.join(f'^{d}' for d in sorted(deps_in_b - deps_in_a))}"
                    )

            for dep in sorted(deps_in_a & deps_in_b):
                a_v = set(StandardVersion.from_string(v) for v in deps_a[dep])
                b_v = set(StandardVersion.from_string(v) for v in deps_b[dep])

                if a_v != b_v:
                    all_versions = sorted(
                        spack.repo.PATH.get_pkg_class(dep).versions.keys()
                    )
                    a_min_b = sorted(a_v - b_v)
                    if a_min_b:
                        try:
                            version_list = to_version_list(a_min_b, all_versions)
                        except Exception:
                            version_list = ",".join(str(x) for x in a_min_b)
                        version_specifier = (
                            "" if str(version_list) == ":" else f"@{version_list}"
                        )
                        changes_for_version.append(
                            f"disallowed `^{dep}{version_specifier}`"
                        )
                    b_min_a = sorted(b_v - a_v)
                    if b_min_a:
                        try:
                            version_list = to_version_list(b_min_a, all_versions)
                        except Exception:
                            version_list = ",".join(str(x) for x in b_min_a)
                        version_specifier = (
                            "" if str(version_list) == ":" else f"@{version_list}"
                        )
                        changes_for_version.append(
                            f"allowed `^{dep}{version_specifier}`"
                        )

            if changes_for_version:
                change_to_version["\n".join(changes_for_version)].append(v)

        if change_to_version:
            print(f"@@ {name} @@")
        for changes_for_version, versions in change_to_version.items():
            if versions is True:
                diff = "-" if changes_for_version.startswith("removed") else "+"
                print(f"{diff} `{name}` {changes_for_version}")
                continue
            try:
                version_list = to_version_list(
                    versions,
                    sorted(spack.repo.PATH.get_pkg_class(name).versions.keys()),
                )
            except Exception as e:
                version_list = ",".join(str(v) for v in versions)
            for line in changes_for_version.split("\n"):
                diff = (
                    "-"
                    if line.startswith("deleted") or line.startswith("allowed")
                    else "+"
                )
                version_specifier = (
                    "" if str(version_list) == ":" else f"@{version_list}"
                )
                print(f"{diff} `{name}{version_specifier}` {line}")
        if change_to_version:
            print()
