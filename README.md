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
5. Run `python3 to_package.py black` to convert to `package.py`.

## License

This project is part of Spack. Spack is distributed under the terms of both the
MIT license and the Apache License (Version 2.0). Users may choose either
license, at their option.

All new contributions must be made under both the MIT and Apache-2.0 licenses.

See LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, and NOTICE for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
