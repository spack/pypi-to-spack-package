# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import sqlite3
import json
import sys

json_file = sys.argv[1]

conn = sqlite3.connect("data.db")
c = conn.cursor()

c.execute(
    """
CREATE TABLE IF NOT EXISTS packages
(name TEXT, version TEXT, requires_dist TEXT, requires_python TEXT)
"""
)

c.execute(
    """
CREATE INDEX IF NOT EXISTS name_index ON packages (name)
"""
)


def insert(entries):
    c.executemany(
        "INSERT INTO packages (name, version, requires_dist, requires_python) VALUES (?, ?, ?, ?)",
        entries,
    )
    conn.commit()


with open(json_file, "rb") as f:
    entries = []
    for i, line in enumerate(f.readlines()):
        data = json.loads(line)

        entries.append(
            (
                data["name"],
                data["version"],
                json.dumps(data.get("requires_dist", []), separators=(",", ":")),
                data.get("requires_python", ""),
            )
        )

        if i % 5000 == 0:
            insert(entries)
            entries = []
            print(i)
    insert(entries)
