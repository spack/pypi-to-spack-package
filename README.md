# PyPI to [Spack](https://www.github.com/spack/spack) `package.py`

This project provides basic tools to do trivial Python package version bumps in Spack.

## Quick start

Install this tool in a virtual environment (currently only available from source):

```console
$ python3 -mvenv .venv
$ source .venv/bin/activate
$ pip install "git+https://github.com/spack/pypi-to-spack-package.git#egg=pypi-to-spack-package"
```

Set up Spack's Python path:

```console
$ export PYTHONPATH=$spack/lib/spack
```

## Fetching the latest PyPI data 

Use the `pypi-to-spack-import` command to fetch the latest PyPI data from the public BigQuery dataset and import it into a local SQLite database.

You need a Google Cloud project and a GCS bucket to store the intermediate BigQuery exports, which you can set up with the following commands:

```console
$ gcloud projects create YOUR_PROJECT_ID
$ gcloud services enable bigquery.googleapis.com --project=YOUR_PROJECT_ID
$ gsutil mb -p YOUR_PROJECT_ID gs://YOUR_BUCKET
```

Then, run the import command:

```console
$ pypi-to-spack-import --bucket YOUR_BUCKET
```

This will create a SQLite database file named `data.db` in the current directory, containing the tables `versions` and `distributions` with the latest PyPI data.

## Updating `package.py` files for existing Spack packages

Use the `pypi-to-spack-bump` command to update the `package.py` files for existing Spack packages based on the latest PyPI data.

Notice that by default, this command will do *trivial* package bumps. This means that the latest version is added, if:

* no changes to runtime dependencies are required (based on `Requires-Dist` fields in the PyPI metadata)
* no changes to the Python version compatibility are required (based on `Requires-Python` fields in the PyPI metadata)

```console
$ pypi-to-spack-bump
```

# Generating a full Spack repository from PyPI

> [!WARNING]
> This feature is currently not as useful, because it's non-trivial to sync a generated repository with the builtin Spack repo, but it can be used to generate a full Spack repository from scratch, or to generate new versions for existing Spack packages.

## Inputs

1. [`spack_requirements.txt`](spack_requirements.txt): a set of requirements we want to generate
   `package.py` files for. This tool will generate at least one matching version for each
   requirement listed in this file.
2. `data.db`: a SQLite database containing most PyPI entries. The database is downloaded
   automatically from the GitHub releases page when running the script for the first time, or you
   can rebuild it with `pypi-to-spack-import`.

## Basic workflow

Show database info (downloads DB if missing):

```console
$ pypi-to-spack info
```

Export *all* Spack's current builtin Python packages and versions to a `spack_requirements.txt` file.

```console
$ pypi-to-spack update-requirements --new
```

Instead of listing *all* Spack's Python packages, you can also manually create a `spack_requirements.txt` file like this:

```
$ cat spack_requirements.txt
black[uvloop,colorama,jupyter,d]
black[uvloop,colorama,jupyter,d] ==25.1.0
black[uvloop,colorama,jupyter,d] ==24.10.0
black[uvloop,colorama,jupyter,d] ==24.8.0
black[uvloop,colorama,jupyter,d] ==24.4.2
black[uvloop,colorama,jupyter,d] ==24.4.1
black[uvloop,colorama,jupyter,d] ==24.4.0
```

Generate `package.py` files for the listed requirements:

```console
$ pypi-to-spack generate --clean --requirements spack_requirements.txt
```

This creates a new repo under `./repo`. Inspect a generated file, e.g. for `black`:

```console
$ cat repo/packages/py-black/package.py
```

When satisfied, export everything to the Spack repository:

```console
$ pypi-to-spack export
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
> Those lines will not be deleted by `pypi-to-spack export`.

## What versions are selected?

For every requirement / constraint that may apply, we take at most the best 10 matching versions.
"Best" in the sense that we prefer final releases with largest version number.

(If you don't want to generate *new* versions, but just clean up existing Spack packages with the
exact same versions, use `pypi-to-spack generate --no-new-versions`)

## Support for specifiers / markers
- ✅ `extra == "a" and extra == "b" or extra == "c"`: gets translated into
  `depends_on(..., when="+a +b")` `depends_on(..., when="+c")` and operators compose fine.
- ✅ `python_version <= "3.8" or python_version >= "3.10"` statements are simplified further
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

## License

This project is part of Spack. Spack is distributed under the terms of both the
MIT license and the Apache License (Version 2.0). Users may choose either
license, at their option.

All new contributions must be made under both the MIT and Apache-2.0 licenses.

See LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, and NOTICE for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
