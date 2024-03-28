# move fully universal wheel packages to spack repo.

import os
import sys
import re

repo_in = os.path.join(sys.argv[1], "packages")
repo_out = os.path.join(sys.argv[2], "packages")

begin_versions = "    # BEGIN VERSIONS [WHEEL ONLY]"
end_versions = "    # END VERSIONS"
begin_variants = "    # BEGIN VARIANTS"
end_variants = "    # END VARIANTS"
begin_deps = "    # BEGIN DEPENDENCIES"
end_deps = "    # END DEPENDENCIES"

directive_regex = re.compile(r"((?:    (?:version|depends_on|variant))\(|\(|\)\n?)")

for dir in sorted(os.listdir(repo_in)):
    in_package = os.path.join(repo_in, dir, "package.py")
    out_package = os.path.join(repo_out, dir, "package.py")

    if not os.path.exists(out_package):
        print(f"not in spack: {dir}", file=sys.stderr)
        continue

    try:
        with open(in_package, "r") as f:
            contents = f.read()
    except OSError:
        print(f"failed to read {in_package}", file=sys.stderr)
        continue

    try:
        versions = contents[
            contents.index(begin_versions) + len(begin_versions) + 1 : contents.index(end_versions)
        ]
        variants = contents[
            contents.index(begin_variants) + len(begin_variants) + 1 : contents.index(end_variants)
        ]
        deps = contents[
            contents.index(begin_deps) + len(begin_deps) + 1 : contents.index(end_deps)
        ]
        assert versions
    except (ValueError, AssertionError):
        continue

    with open(out_package, "r") as f:
        out_contents = f.read()

    count = 0
    piece = ""
    mode = None
    stuff_added = False
    output_file = ""
    for part in directive_regex.split(out_contents):
        if mode is None and ("version(" in part or "depends_on(" in part or "variant(" in part):
            mode = "directive"
            count = 1
            piece = part
            if not stuff_added and ("version(" in part or "depends_on(" in part):
                output_file += "\n".join(p for p in (versions, variants, deps) if p)
                stuff_added = True

        elif mode == "directive":
            piece += part
            if part == "(":
                count += 1
            if part.startswith(")"):
                count -= 1
                if count == 0:
                    mode = None
                    print(f"removing {piece!r}")
                    piece = ""
        else:
            output_file += part

    with open(out_package, "w") as f:
        f.write(output_file)
