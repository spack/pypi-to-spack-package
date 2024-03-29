import io
import sqlite3
import sys

import spack.repo
from packaging.version import Version
from spack.build_systems.python import PythonExtension, PythonPackage

# db
conn = sqlite3.connect("data.db")
c = conn.cursor()


def is_pypi(pkg):
    if PythonPackage not in pkg.__bases__ and PythonExtension not in pkg.__bases__:
        return False
    name = pkg.name[3:] if pkg.name.startswith("py-") else pkg.name
    return c.execute("SELECT * FROM versions WHERE name = ?", (name,)).fetchone() is not None


def dump(f: io.StringIO = sys.stdout):
    count = 0
    for pkg in spack.repo.PATH.all_package_classes():
        if not is_pypi(pkg):
            continue
        count += 1
        name = pkg.name[3:] if pkg.name.startswith("py-") else pkg.name

        variants = ",".join(s for s in pkg.variants if s != "build_system")
        variants = variants if not variants else f"[{variants}]"

        for version in pkg.versions:
            try:
                Version(str(version))
            except:
                print(f"skipping {name}=={version}")
                continue
            print(f"{name}{variants} =={version}", file=f)
    print(f"total: {count} pypi packages")


with open("spack_requirements.txt", "w") as f:
    dump(f)
