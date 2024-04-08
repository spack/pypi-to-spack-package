# PyPI to [Spack](https://www.github.com/spack/spack) `package.py`

## Usage

There are two inputs in this tool:

1. [`spack_requirements.txt`](spack_requirements.txt): a set of requirements we want to generate
   `package.py` files for. This tool will generate at least one matching version for each
    requirement listed in this file.
2. `data.db`: a SQLite database containing most PyPI entries. The database is downloaded
    automatically from the GitHub releases page when running the script for the first time.

Let's first see what's in the database:

```console
$ ./src/package.py info
Total packages: 617672
Total versions: 6059310
```

Then let's export Spack's builtin packages to a `spack_requirements.txt` file:

```console
$ ./src/package.py update-requirements --new
```

This command populates [spack_requirements.txt](spack_requirements.txt). The `--new` flag ensures
that we will later generate new versions not yet in Spack available in PyPI.

Next, we generate `package.py` files for the top N versions:

```console
$ ./src/package.py generate --clean spack_requirements.txt
```

This generates a new repo in the `./repo` dir. Have a look at `black`'s generated `package.py`
file:

```console
$ cat repo/packages/py-black/package.py
```

```python
from spack.package import *


class PyBlack(PythonPackage):
    version("24.3.0", sha256="a0c9c4a0771afc6919578cec71ce82a3e31e054904e7197deacbc9382671c41f", url="https://pypi.org/packages/8f/5f/bac24a952668c7482cfdb4ebf91ba57a796c9da8829363a772040c1a3312/black-24.3.0.tar.gz")
    version("24.2.0", sha256="bce4f25c27c3435e4dace4815bcb2008b87e167e3bf4ee47ccdc5ce906eb4894", url="https://pypi.org/packages/29/69/f3ab49cdb938b3eecb048fa64f86bdadb1fac26e92c435d287181d543b0a/black-24.2.0.tar.gz")
    version("24.1.1", sha256="48b5760dcbfe5cf97fd4fba23946681f3a81514c6ab8a45b50da67ac8fbc6c7b", url="https://pypi.org/packages/77/ec/a429d15d2e7f996203bff98e2b2e84ad4cb3de318de147b0038dc93fbc71/black-24.1.1.tar.gz")
    version("24.1.0", sha256="30fbf768cd4f4576598b1db0202413fafea9a227ef808d1a12230c643cefe9fc", url="https://pypi.org/packages/ea/19/33d4f2f0babcbc07d3e2c058a64c76606cf19884a600536c837aaf4e4f2d/black-24.1.0.tar.gz")
    version("23.12.1", sha256="4ce3ef14ebe8d9509188014d96af1c456a910d5b5cbf434a09fef7e024b3d0d5", url="https://pypi.org/packages/fd/f4/a57cde4b60da0e249073009f4a9087e9e0a955deae78d3c2a493208d0c5c/black-23.12.1.tar.gz")
    version("23.12.0", sha256="330a327b422aca0634ecd115985c1c7fd7bdb5b5a2ef8aa9888a82e2ebe9437a", url="https://pypi.org/packages/5a/73/618bcfd4a4868d52c02ff7136ec60e9d63bc83911d3d8b4998e42acf9557/black-23.12.0.tar.gz")
    version("23.11.0", sha256="4c68855825ff432d197229846f971bc4d6666ce90492e5b02013bcaca4d9ab05", url="https://pypi.org/packages/ef/21/c2d38c7c98a089fd0f7e1a8be16c07f141ed57339b3082737de90db0ca59/black-23.11.0.tar.gz")
    version("23.10.1", sha256="1f8ce316753428ff68749c65a5f7844631aa18c8679dfd3ca9dc1a289979c258", url="https://pypi.org/packages/36/bf/a462f36723824c60dc3db10528c95656755964279a6a5c287b4f9fd0fa84/black-23.10.1.tar.gz")
    version("23.10.0", sha256="31b9f87b277a68d0e99d2905edae08807c007973eaa609da5f0c62def6b7c0bd", url="https://pypi.org/packages/2d/e0/8433441b0236b9d795ffbf5750f98144e0378b6e20401ba4d2db30b99a5c/black-23.10.0.tar.gz")
    version("23.9.1", sha256="24b6b3ff5c6d9ea08a8888f6977eae858e1f340d7260cf56d70a49823236b62d", url="https://pypi.org/packages/12/c3/257adbdbf2cc60bf844b5c0e3791a9d49e4fb4f7bcd8a2e875824ca0b7bc/black-23.9.1.tar.gz")

    variant("colorama", default=False)
    variant("d", default=False)
    variant("jupyter", default=False)
    variant("uvloop", default=False)

    with default_args(type="run"):
        depends_on("py-aiohttp@3.7.4:", when="@23.12:23.12.0,24:24.1.0 platform=linux")
        depends_on("py-aiohttp@3.7.4:", when="@23.12:23.12.0,24:24.1.0 platform=freebsd")
        depends_on("py-aiohttp@3.7.4:", when="@23.12:23.12.0,24:24.1.0 platform=darwin")
        depends_on("py-aiohttp@3.7.4:", when="@23.12:23.12.0,24:24.1.0 platform=cray")
        depends_on("py-aiohttp@3.7.4:", when="@21.10-beta0:21,22.10:+d")
        depends_on("py-click@8.0.0:", when="@22.10:")
        depends_on("py-colorama@0.4.3:", when="@20:21,22.10:+colorama")
        depends_on("py-ipython@7.8:", when="@21.8-beta0:21,22.10:+jupyter")
        depends_on("py-mypy-extensions@0.4.3:", when="@20:21,22.10:")
        depends_on("py-packaging@22:", when="@23.1.0:")
        depends_on("py-pathspec@0.9:", when="@22.10:")
        depends_on("py-platformdirs@2.0.0:", when="@21.8-beta0:21,22.10:")
        depends_on("py-tokenize-rt@3.2:", when="@21.8-beta0:21,22.10:+jupyter")
        depends_on("py-tomli@1.1:", when="@22.10: ^python@:3.10")
        depends_on("py-typing-extensions@4.0.1:", when="@23.9: ^python@:3.10")
        depends_on("py-uvloop@0.15.2:", when="@21.5-beta2:21,22.10:+uvloop")


```

When you're happy, you can automatically export all the packages to the Spack repository:

```console
$ ./src/package.py export
```

> [!TIP]
> Some packages have further runtime dependencies from the system that cannot be articulated in
> `Requires-Dist` fields (e.g. `depends_on("git", type="run")`). Furthermore, Spack maintainers
> sometimes add forward compat bounds that were not anticipated at the time of release of a version
> (e.g. `depends_on("py-cython@:0", when="@:3")` or `depends_on("python@:3.10", when="@:4")`). To
> preserve these extra constraints, you can wrap them in the builtin repo's `package.py`:
> ```python
> # <<< extra constraints
> depends_on(...)
> # extra constraints >>>
> ```
> Those lines will not be deleted by `./src/package.py export`.

## What versions are selected?

For every requirement / constraint that may apply, we take at most the best 10 matching versions.
"Best" in the sense that we prefer final releases with largest version number.

(If you don't want to generate *new* versions, but just clean up existing Spack packages with the
exact same versions, use `./src/package.py generate --no-new-versions`)

## Support for specifiers / markers
- ✅ `extra == "a" and extra == "b" or extra == "c"`: gets translated into
  `depends_on(..., when="+a +b")` `depends_on(..., when="+c")` and operators compose fine.
- ✅ `python_version <= "3.8" or python_version >= "3.10` statements are simplified further
  to a single constraint `when="^python@:3.8,3.10:"`.
- ✅ The variables `sys_platform` and `platform_system` with `==` and `!=` operators are
  translated to one or more `platform=...` for Linux (+cray), Darwin, Windows and FreeBSD.
- ✅ Support for correctly ordered pre-releases through https://github.com/spack/spack/pull/43140
- ✅ Specifiers for dependencies (`>=3.7`, `~=3.7.1` etc) are not directly translated, but
  evaluated on all known versions of the dependency, and then simplified into a Spack range. This
  ensures that we don't have to deal with edge cases (of which there are many). The downside of
  this approach is that it's rather slow, since it does not perform binary search.


## Issues

- **PKG-INFO**: some packages do not provide the `Requires-Dist` fields in the `PKG-INFO` file,
  meaning that this project cannot know their dependencies. This is a shortcoming of the PyPI
  database.
- **Build dependencies**: currently it cannot generate build dependencies, as they are lacking
  from the PyPI database.
- **Markers**: not all can be directly translated to a single `when` condition:
  - ❌ `python_version in "3.7,3.8,3.9"`: could be translated into `^python@3.7:3.9`, but is not,
    because the `in` and `not in` operators use the right-hand side as literal string, instead of
    a version list. So, I have not implemented this.
  - ❌ The variables `os_name`, `platform_machine`, `platform_release`, `platform_version`,
  `implementation_version` are still not implemented (some cannot be?).

## Importing the PyPI database

> [!NOTE]  
> This is only necessary if you want to populate a database from scratch, or update the database
> to the very latest. By default, a (slightly outdated) copy of the database is fetched
> automatically.

1. Go to https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=pypi&page=dataset
2. Run the following query

   ```sql
   EXPORT DATA OPTIONS(
     compression="GZIP",
     uri="gs://<bucket>/pypi-distributions/pypi-*.json.gz",
     format="JSON",
     overwrite=true
   )

   AS

   SELECT
     -- https://packaging.python.org/en/latest/specifications/name-normalization/
     REGEXP_REPLACE(LOWER(x.name), "[-_.]+", "-") AS normalized_name,
     x.version,
     x.requires_dist,
     x.requires_python,
     x.sha256_digest,
     x.path

   FROM `bigquery-public-data.pypi.distribution_metadata` AS x

   -- Do not use a universal wheel if there are platform specific wheels (e.g. black can be built
   -- both binary and pure python, in that case prefer sdist)
   LEFT JOIN `bigquery-public-data.pypi.distribution_metadata` AS y ON (
     REGEXP_REPLACE(LOWER(x.name), "[-_.]+", "-") = REGEXP_REPLACE(LOWER(y.name), "[-_.]+", "-")
     AND x.version = y.version
     AND x.packagetype = "bdist_wheel"
     AND y.packagetype = "bdist_wheel"
     AND y.path NOT LIKE "%-none-any.whl"
   )

   -- Select sdist and universal wheels
   WHERE (
     x.packagetype = "sdist"
     OR x.path LIKE "%py3-none-any.whl"
   ) AND y.name IS NULL
   -- AND x.upload_time >= "2024-03-01" -- If you already have a db, set this to last time fetched

   -- Only pick the last (re)upload of (name, version, packagetype) tuples
   QUALIFY ROW_NUMBER() OVER (
     PARTITION BY normalized_name, x.version, x.packagetype
     ORDER BY x.upload_time DESC
   ) = 1

   -- If there are both universal wheels and sdist, pick the wheel
   AND ROW_NUMBER() OVER (
     PARTITION BY normalized_name, x.version
     ORDER BY CASE WHEN x.packagetype = 'bdist_wheel' THEN 0 ELSE 1 END
   ) = 1
   ```
   which should say something like "Successfully exported 5804957 rows into 101 files".

3. Also make an export of all known versions:

   ```sql
   EXPORT DATA OPTIONS(
     compression="GZIP",
     uri="gs://<bucket>/pypi-versions/pypi-*.json.gz",
     format="JSON",
     overwrite=true
   )

   AS

   SELECT
     REGEXP_REPLACE(LOWER(name), "[-_.]+", "-") AS normalized_name,
     version
   FROM `bigquery-public-data.pypi.distribution_metadata`
   GROUP BY normalized_name, version
   ```
4. Download the files using:
   ```console
   $ gsutil -m cp -r gs://<bucket>/pypi-distributions .
   $ gsutil -m cp -r gs://<bucket>/pypi-versions .
   ```
4. Run `python3 src/import.py --versions --distributions ` to import as sqlite.


## License

This project is part of Spack. Spack is distributed under the terms of both the
MIT license and the Apache License (Version 2.0). Users may choose either
license, at their option.

All new contributions must be made under both the MIT and Apache-2.0 licenses.

See LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, and NOTICE for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
