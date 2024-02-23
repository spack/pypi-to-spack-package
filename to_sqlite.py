# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import json
import sqlite3
import sys

from dateutil.parser import parse as dateparse

json_file = sys.argv[1]

conn = sqlite3.connect("data.db")
c = conn.cursor()

c.execute(
    """
CREATE TABLE IF NOT EXISTS packages
(
name TEXT NOT NULL,
version TEXT NOT NULL,
requires_dist TEXT,
requires_python TEXT,
upload_time DATETIME NOT NULL,
sha256 BLOB(32) NOT NULL
)
"""
)

c.execute(
    """
CREATE INDEX IF NOT EXISTS name_index ON packages (name)
"""
)


def insert(entries):
    c.executemany(
        "INSERT INTO packages (name, version, requires_dist, requires_python, upload_time, sha256) VALUES (?, ?, ?, ?, ?, ?)",
        entries,
    )
    conn.commit()


with open(json_file, "rb") as f:
    entries = []
    for i, line in enumerate(f.readlines()):
        data = json.loads(line)

        upload_time = data["upload_time"]

        # should always be UTC
        if upload_time.endswith(" UTC"):
            upload_time = upload_time[:-4]

        entries.append(
            (
                data["name"],
                data["version"],
                json.dumps(data.get("requires_dist", []), separators=(",", ":")),
                data.get("requires_python", ""),
                upload_time,
                bytearray.fromhex(data.get("sha256_digest", "")),
            )
        )

        if i % 5000 == 0:
            insert(entries)
            entries = []
            print(i)
    insert(entries)
