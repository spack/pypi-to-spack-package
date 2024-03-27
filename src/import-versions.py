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
CREATE TABLE IF NOT EXISTS version_lookup
(
name TEXT NOT NULL,
version TEXT NOT NULL
)
"""
)

c.execute(
    """
CREATE INDEX IF NOT EXISTS version_lookup_by_name ON version_lookup (name)
"""
)


def insert(entries):
    c.executemany(
        """
    INSERT INTO version_lookup (name, version)
    VALUES (?, ?)
    """,
        entries,
    )
    conn.commit()


path = sys.argv[1] if len(sys.argv) > 1 else "pypi-versions"

entries = []
i = 0
for file in sorted(os.listdir(path)):
    print(f"importing {file}")
    with gzip.open(os.path.join(path, file), "rb") as f:
        for line in f.readlines():
            i += 1
            data = json.loads(line)

            entries.append((data["normalized_name"], data["version"]))

            if i % 5000 == 0:
                insert(entries)
                entries = []
                print(i)
insert(entries)
