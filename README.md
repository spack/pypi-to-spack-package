# PyPI to [Spack](https://www.github.com/spack/spack) `package.py`

Note: this is work in progress.

## Usage

First checkout a Spack version that supports correct pre-release ordering.

```console
$ cd spack
$ git fetch origin pull/43140/head:pr-43140
$ git checkout pr-43140
```

Let's fetch the PyPI database and have a look what's in there.

```console
$ spack-python src/package.py info
Total packages: 592120
Total versions: 5651880

$ spack-python src/package.py info black
Normalized name: black
Variants: colorama d jupyter uvloop
Total versions: 19
```

Then run

```console
$ spack-python src/package.py generate black
```

to generate `package.py` files for `black` and its dependencies in the `pypi/` directory.

Alternatively, use

```
$ spack-python src/package.py generate black jupyter colorama d uvloop
```

to include optional dependencies as well.

Then have a look at `black`'s generated `package.py` file:

```console
$ cat pypi/packages/py-black/package.py
```

```python
from spack.package import *


class PyBlack(PythonPackage):
    url = "https://www.example.com/file.tar.gz"

    version("24.2.0", sha256="bce4f25c27c3435e4dace4815bcb2008b87e167e3bf4ee47ccdc5ce906eb4894")  # sdist
    version("24.1.1", sha256="48b5760dcbfe5cf97fd4fba23946681f3a81514c6ab8a45b50da67ac8fbc6c7b")  # sdist
    version("23.12.1", sha256="4ce3ef14ebe8d9509188014d96af1c456a910d5b5cbf434a09fef7e024b3d0d5")  # sdist
    version("23.11.0", sha256="4c68855825ff432d197229846f971bc4d6666ce90492e5b02013bcaca4d9ab05")  # sdist
    version("23.10.1", sha256="1f8ce316753428ff68749c65a5f7844631aa18c8679dfd3ca9dc1a289979c258")  # sdist
    version("23.9.1", sha256="24b6b3ff5c6d9ea08a8888f6977eae858e1f340d7260cf56d70a49823236b62d")  # sdist
    version("23.7.0", sha256="022a582720b0d9480ed82576c920a8c1dde97cc38ff11d8d8859b3bd6ca9eedb")  # sdist
    version("23.3.0", sha256="1c7b8d606e728a41ea1ccbd7264677e494e87cf630e399262ced92d4a8dac940")  # sdist
    version("23.1.0", sha256="b0bd97bea8903f5a2ba7219257a44e3f1f9d00073d6cc1add68f0beec69692ac")  # sdist
    version("22.12.0", sha256="229351e5a18ca30f447bf724d007f890f97e13af070bb6ad4c0a441cd7596a2f")  # sdist
    version("22.10.0", sha256="f513588da599943e0cde4e32cc9879e825d58720d6557062d1098c5ad80080e1")  # sdist
    version("22.8.0", sha256="792f7eb540ba9a17e8656538701d3eb1afcb134e3b45b71f20b25c77a8db7e6e")  # sdist
    version("22.6.0", sha256="6c6d39e28aed379aec40da1c65434c77d75e65bb59a1e1c283de545fb4e7c6c9")  # sdist
    version("22.3.0", sha256="35020b8886c022ced9282b51b5a875b6d1ab0c387b31a065b84db7c33085ca79")  # sdist
    version("22.1.0", sha256="a7c0192d35635f6fc1174be575cb7915e92e5dd629ee79fdaf0dcfa41a80afb5")  # sdist

    variant("colorama", default=False)
    variant("d", default=False)
    variant("jupyter", default=False)
    variant("uvloop", default=False)

    with default_args(type=("build", "run")):
        depends_on("py-aiohttp@3.7.4:", when="@22.10:+d")
        depends_on("py-click@8:", when="@22.10:")
        depends_on("py-colorama@0.4.3:", when="@22.10:+colorama")
        depends_on("py-ipython@7.8:", when="@22.10:+jupyter")
        depends_on("py-mypy-extensions@0.4.3:", when="@22.10:")
        depends_on("py-packaging@22:", when="@23.1:")
        depends_on("py-pathspec@0.9:", when="@22.10:")
        depends_on("py-platformdirs@2:", when="@22.10:")
        depends_on("py-tokenize-rt@3.2:", when="@22.10:+jupyter")
        depends_on("py-tomli@1.1:", when="@23.1: ^python@:3.10")
        depends_on("py-typing-extensions@4.0.1:", when="@23.9: ^python@:3.10")
        depends_on("py-typing-extensions@3.10:", when="@22.10:23.7 ^python@:3.9")
        depends_on("py-uvloop@0.15.2:", when="@22.10:+uvloop")

        # marker: python_full_version < "3.11.0a7"
        # depends_on("py-tomli@1.1:", when="@22.10:22.12")


```

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

## TODO

- [ ] Update spack package.py files instead of creating new ones.

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
     uri="gs://<bucket>/pypi-export/pypi-*.json.gz",
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
     x.path,
     x.packagetype = "sdist" AS is_sdist

   FROM `bigquery-public-data.pypi.distribution_metadata` AS x

   -- Do not use a universal wheel if there are platform specific wheels (e.g. black can be built
   -- both binary and pure python, in that case prefer sdist)
   LEFT JOIN `bigquery-public-data.pypi.distribution_metadata` AS y ON (
    REGEXP_REPLACE(LOWER(x.name), "[-_.]+", "-") = REGEXP_REPLACE(LOWER(y.name), "[-_.]+", "-")
    AND x.version = y.version
    AND x.packagetype = "bdist_wheel"
    AND y.packagetype = "bdist_wheel"
    AND y.python_version NOT IN ("py2.py3", "py3")
   )

   -- Select sdist and universal wheels
   WHERE (
     x.packagetype = "sdist"
     OR x.packagetype = "bdist_wheel" AND x.python_version IN ("py2.py3", "py3")
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
3. Download the files using:
   ```console
   $ gsutil -m cp -r gs://<bucket>/pypi-export .
   ```
4. Run `python3 src/import.py pypi-export/` to import as sqlite.

## License

This project is part of Spack. Spack is distributed under the terms of both the
MIT license and the Apache License (Version 2.0). Users may choose either
license, at their option.

All new contributions must be made under both the MIT and Apache-2.0 licenses.

See LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, and NOTICE for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
