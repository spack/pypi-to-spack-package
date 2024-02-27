# PyPI to [Spack](https://www.github.com/spack/spack) `package.py`

Note: this is work in progress.

Convert Python PyPI entries to Spack `package.py`

1. Go to https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=pypi&page=dataset
2. Run the following query

   ```sql
    SELECT name, version, requires_dist, requires_python, upload_time, sha256_digest
    FROM `bigquery-public-data.pypi.distribution_metadata`
    WHERE packagetype = "sdist"
    ORDER BY upload_time DESC
   ```

   should produce about 5.5M rows.
3. Hit "Save Results" > "JSONL" and download from Google drive.
4. Run `python3 to_sqlite.py <path to json file>` to convert to sqlite.
5. Install `packaging` if not installed already
5. Run `spack-python to_package.py <pkg>` to convert to `package.py`.


## Example

```console
$ spack-python to_package.py describe black
Normalized name: black
Variants: jupyter colorama d uvloop
Total versions: 19
```

```console
$ spack-python to_package.py generate black jupyter colorama d uvloop > out.py
```

outputs a large file

```console
$ wc -l out.py
8896 out.py
```

with `package.py` definitions for `black` and all its dependencies

```console
$ head -n50 out.py 
```

```python
from spack.package import *

class PyBlack:
    version("24.2.0", sha256="bce4f25c27c3435e4dace4815bcb2008b87e167e3bf4ee47ccdc5ce906eb4894")
    version("24.1.1", sha256="48b5760dcbfe5cf97fd4fba23946681f3a81514c6ab8a45b50da67ac8fbc6c7b")
    version("24.1.0", sha256="30fbf768cd4f4576598b1db0202413fafea9a227ef808d1a12230c643cefe9fc")
    version("23.12.1", sha256="4ce3ef14ebe8d9509188014d96af1c456a910d5b5cbf434a09fef7e024b3d0d5")
    version("23.12.0", sha256="330a327b422aca0634ecd115985c1c7fd7bdb5b5a2ef8aa9888a82e2ebe9437a")
    version("23.11.0", sha256="4c68855825ff432d197229846f971bc4d6666ce90492e5b02013bcaca4d9ab05")
    version("23.10.1", sha256="1f8ce316753428ff68749c65a5f7844631aa18c8679dfd3ca9dc1a289979c258")
    version("23.10.0", sha256="31b9f87b277a68d0e99d2905edae08807c007973eaa609da5f0c62def6b7c0bd")
    version("23.9.1", sha256="24b6b3ff5c6d9ea08a8888f6977eae858e1f340d7260cf56d70a49823236b62d")
    version("23.9.0", sha256="3511c8a7e22ce653f89ae90dfddaf94f3bb7e2587a245246572d3b9c92adf066")
    version("23.7.0", sha256="022a582720b0d9480ed82576c920a8c1dde97cc38ff11d8d8859b3bd6ca9eedb")
    version("23.3.0", sha256="1c7b8d606e728a41ea1ccbd7264677e494e87cf630e399262ced92d4a8dac940")
    version("23.1.0", sha256="b0bd97bea8903f5a2ba7219257a44e3f1f9d00073d6cc1add68f0beec69692ac")
    version("22.12.0", sha256="229351e5a18ca30f447bf724d007f890f97e13af070bb6ad4c0a441cd7596a2f")
    version("22.10.0", sha256="f513588da599943e0cde4e32cc9879e825d58720d6557062d1098c5ad80080e1")
    version("22.8.0", sha256="792f7eb540ba9a17e8656538701d3eb1afcb134e3b45b71f20b25c77a8db7e6e")
    version("22.6.0", sha256="6c6d39e28aed379aec40da1c65434c77d75e65bb59a1e1c283de545fb4e7c6c9")
    version("22.3.0", sha256="35020b8886c022ced9282b51b5a875b6d1ab0c387b31a065b84db7c33085ca79")
    version("22.1.0", sha256="a7c0192d35635f6fc1174be575cb7915e92e5dd629ee79fdaf0dcfa41a80afb5")

    variant("jupyter", default=False)
    variant("colorama", default=False)
    variant("uvloop", default=False)
    variant("d", default=False)

    with default_args(deptype=("build", "run")):
        depends_on("python@3.7:", when="@22.10.0:23.3")
        depends_on("python@3.8:", when="@23.7.0:")
        depends_on("py-aiohttp@3.7.4:", when="@23.12.0")
        depends_on("py-aiohttp@3.7.4:", when="@22.10.0:23.11,23.12.1:+d")
        depends_on("py-click@8.0.0:", when="@22.10.0:")
        depends_on("py-colorama@0.4.3:", when="@22.10.0:+colorama")
        depends_on("py-ipython@7.8.0:", when="@22.10.0:+jupyter")
        depends_on("py-mypy-extensions@0.4.3:", when="@22.10.0:")
        depends_on("py-packaging@22.0:", when="@23.1.0:")
        depends_on("py-pathspec@0.9.0:", when="@22.10.0:")
        depends_on("py-platformdirs@2:", when="@22.10.0:")
        depends_on("py-tokenize-rt@3.2.0:", when="@22.10.0:+jupyter")
        depends_on("py-tomli@1.1.0:", when="@23.1.0: ^python@:3.10")
        depends_on("py-typed-ast@1.4.2:", when="@22.10.0:23.3 ^python@:3.7")
        depends_on("py-typing-extensions@3.10.0.0:", when="@22.10.0:23.7 ^python@:3.9")
        depends_on("py-typing-extensions@4.0.1:", when="@23.9.0: ^python@:3.10")
        depends_on("py-uvloop@0.15.2:", when="@22.10.0:+uvloop")

        # marker: python_full_version < "3.11.0a7"
        # depends_on("py-tomli@1.1.0:", when="@22.10.0:22.12")
```

## Issues

- **PKG-INFO**: some packages do not provide the `Requires-Dist` fields in the `PKG-INFO` file,
  meaning that we do not know their dependencies. This is a shortcoming of the PyPI database.
- **Build dependencies**: currently we cannot generate build dependencies, as they are lacking
  from the PyPI database.
- **Markers**: not all can be directly translated to a single `when` condition:
  - ✅ `extra == "a" and extra == "b"`: works, and gets translated into `when="+a +b"`
  - ✅ `python_version <= "3.8" or python_version >= "3.10`: works, because the `or` operator
    can be translated into a version list `^python@:3.8,3.10:`.
  - ❌ `extra == "foo" or extra == "bar"`: can only be expressed in Spack through two separate
    `depends_on(..., when="+foo)` and `depends_on(..., when="+bar)` statements, which is not
    yet implemented.
  - ❌ `python_version in "3.7,3.8,3.9"`: could be translated into `^python@3.7:3.9`, but is not,
    because the `in` and `not in` operators use the right-hand side as literal string, instead of
    a version list. So, I have not implemented this.
  - ❌ The variables `os_name`, `sys_platform`, `platform_machine`, `platform_release`,
    `platform_system`, `platform_version`, `implementation_version` are still to-do in as far as
    they can be translated to a `Spec`.

## License

This project is part of Spack. Spack is distributed under the terms of both the
MIT license and the Apache License (Version 2.0). Users may choose either
license, at their option.

All new contributions must be made under both the MIT and Apache-2.0 licenses.

See LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, and NOTICE for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
