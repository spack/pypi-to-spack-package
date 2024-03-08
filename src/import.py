# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import gzip
import json
import os
import sqlite3
import sys

conn = sqlite3.connect("data.db")
c = conn.cursor()

c.execute(
    """
CREATE TABLE IF NOT EXISTS versions
(
name TEXT NOT NULL,
version TEXT NOT NULL,
requires_dist TEXT,
requires_python TEXT,
sha256 BLOB(32) NOT NULL,
path TEXT NOT NULL,
is_sdist INTEGER
)
"""
)

c.execute(
    """
CREATE UNIQUE INDEX IF NOT EXISTS name_index ON versions (name, version)
"""
)


def insert(entries):
    c.executemany(
        """
    INSERT INTO versions (name, version, requires_dist, requires_python, sha256, path, is_sdist)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(name, version) DO UPDATE SET
    requires_dist = excluded.requires_dist,
    requires_python = excluded.requires_python,
    sha256 = excluded.sha256,
    path = excluded.path,
    is_sdist = excluded.is_sdist
    """,
        entries,
    )
    conn.commit()


path = sys.argv[1] if len(sys.argv) > 1 else "pypi-export"

entries = []
i = 0
for file in sorted(os.listdir(path)):
    print(f"importing {file}")
    with gzip.open(os.path.join(path, file), "rb") as f:
        for line in f.readlines():
            i += 1
            data = json.loads(line)

            entries.append(
                (
                    data["normalized_name"],
                    data["version"],
                    json.dumps(data.get("requires_dist", []), separators=(",", ":")),
                    data.get("requires_python", ""),
                    bytearray.fromhex(data.get("sha256_digest", "")),
                    data["path"],
                    data["is_sdist"],
                )
            )

            if i % 5000 == 0:
                insert(entries)
                entries = []
                print(i)
insert(entries)

