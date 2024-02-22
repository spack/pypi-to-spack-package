# PyPI to [Spack](https://www.github.com/spack/spack) `package.py`

Note: this is work in progress.

Convert Python PyPI entries to Spack `package.py`

1. Go to https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=pypi&page=dataset
2. Run the following query

   ```sql
   SELECT name, version, requires_dist, requires_python
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
$ spack-python to_package.py black
```

outputs

```python
version("24.2.0")
version("24.1.1")
version("24.1.0")
version("23.12.1")
version("23.12.0")
version("23.11.0")
version("23.10.1")
version("23.10.0")
version("23.9.1")
version("23.9.0")
version("23.7.0")
version("23.3.0")
version("23.1.0")
version("22.12.0")
version("22.10.0")
version("22.8.0")
version("22.6.0")
version("22.3.0")
version("22.1.0")

with default_args(deptype=("build", "run")):
    depends_on("python@3.6.2:", when="@22.1.0:22.8.0")
    depends_on("python@3.7:", when="@22.10.0:23.3.0")
    depends_on("python@3.8:", when="@23.7.0:")
    depends_on("py-click@8.0.0:", when="@22.10.0:")
    depends_on("py-mypy-extensions@0.4.3:", when="@22.10.0:")
    depends_on("py-packaging@22.0:", when="@23.1.0:")
    depends_on("py-pathspec@0.9.0:", when="@22.10.0:")
    depends_on("py-platformdirs@2:", when="@22.10.0:")
    depends_on("py-tomli@1.1.0:", when="@22.10.0:22.12.0 ^python@:3.11.0a6")
    depends_on("py-tomli@1.1.0:", when="@23.1.0: ^python@:3.10")
    depends_on("py-typed-ast@1.4.2:", when="@22.10.0:23.3.0 ^python@:3.7")
    depends_on("py-typing-extensions@3.10.0.0:", when="@22.10.0:23.7.0 ^python@:3.9")
    depends_on("py-typing-extensions@4.0.1:", when="@23.9.0: ^python@:3.10")

    depends_on("py-colorama@0.4.3:", when="@22.10.0:+colorama")
    depends_on("py-aiohttp@3.7.4:", when="@22.10.0:23.11.0+d")
    depends_on("py-ipython@7.8.0:", when="@22.10.0:+jupyter")
    depends_on("py-tokenize-rt@3.2.0:", when="@22.10.0:+jupyter")
    depends_on("py-uvloop@0.15.2:", when="@22.10.0:+uvloop")

    # marker: sys_platform == "win32" and implementation_name == "pypy" and extra == "d"
    # depends_on("py-aiohttp@3.7.4:3.8,3.9.1:", when="@23.12.0")

    # marker: (sys_platform == "win32" and implementation_name == "pypy") and extra == "d"
    # depends_on("py-aiohttp@3.7.4:3.8,3.9.1:", when="@23.12.1:")

    # marker: sys_platform != "win32" or implementation_name != "pypy" and extra == "d"
    # depends_on("py-aiohttp@3.7.4:", when="@23.12.0")

    # marker: (sys_platform != "win32" or implementation_name != "pypy") and extra == "d"
    # depends_on("py-aiohttp@3.7.4:", when="@23.12.1:")
```

## License

This project is part of Spack. Spack is distributed under the terms of both the
MIT license and the Apache License (Version 2.0). Users may choose either
license, at their option.

All new contributions must be made under both the MIT and Apache-2.0 licenses.

See LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, and NOTICE for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
