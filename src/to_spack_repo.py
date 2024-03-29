# move fully universal wheel packages to spack repo.

import ast
import os
import re
import sys
from typing import List, Tuple

import spack.util.naming as nm

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
        ].split("\n")
        variants = contents[
            contents.index(begin_variants) + len(begin_variants) + 1 : contents.index(end_variants)
        ].split("\n")
        deps = contents[
            contents.index(begin_deps) + len(begin_deps) + 1 : contents.index(end_deps)
        ].split("\n")
        assert versions
    except (ValueError, AssertionError):
        continue

    with open(out_package, "r") as f:
        src = f.read()

    lines = src.split("\n")
    tree = ast.parse(src)
    clasname = nm.mod_to_class(dir)

    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == clasname:
            class_node = n
            break
    else:
        print(f"failed to find class {clasname} in {out_package}", file=sys.stderr)
        continue

    lines_to_delete = set()

    for node in n.body:
        # delete with expressions and loops
        if isinstance(node, (ast.With, ast.For)):
            for i in range(node.lineno - 1, node.end_lineno):
                lines_to_delete.add(i)
            continue

        elif not isinstance(node, ast.Expr):
            continue

        expr = node.value

        # and delete any version, depends_on, or variant calls
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id in ("version", "depends_on", "variant")
        ):
            for i in range(expr.lineno - 1, expr.end_lineno):
                lines_to_delete.add(i)
            if expr.func.id == "variant":
                pattern = f'variant("{expr.args[0].s}"'

                # Preserve variants from the original package as they contain a description.
                for i, line in enumerate(variants):
                    if pattern in line:
                        variants[i] = "\n".join(lines[expr.lineno - 1 : expr.end_lineno])

    delete = sorted(lines_to_delete, reverse=True)

    for i in delete:
        del lines[i]

    lines.insert(
        delete[-1],
        "\n".join(l for l in ("\n".join(versions), "\n".join(variants), "\n".join(deps)) if l),
    )

    with open(out_package, "w") as f:
        f.write("\n".join(lines))
