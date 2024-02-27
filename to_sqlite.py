# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import json
import re
import sqlite3
import sys
from datetime import datetime

json_file = sys.argv[1]

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
upload_time DOUBLE PRECISION NOT NULL,
sha256 BLOB(32) NOT NULL
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
INSERT INTO versions (name, version, requires_dist, requires_python, upload_time, sha256)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(name, version) DO UPDATE SET
requires_dist = excluded.requires_dist,
requires_python = excluded.requires_python,
upload_time = excluded.upload_time,
sha256 = excluded.sha256
WHERE excluded.upload_time > upload_time""",
        entries,
    )
    conn.commit()


def parse_date(d: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %Z", "%Y-%m-%d %H:%M:%S %Z"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            pass
    raise ValueError("Unable to parse date string")


NAME_REGEX = re.compile(r"[-_.]+")


def normalize(name):
    return re.sub(NAME_REGEX, "-", name).lower()


with open(json_file, "rb") as f:
    entries = []
    newest_time = datetime.min
    newest_line = None
    for i, line in enumerate(f.readlines()):
        data = json.loads(line)

        upload_time = parse_date(data["upload_time"])

        if upload_time > newest_time:
            newest_time = upload_time
            newest_line = line

        entries.append(
            (
                normalize(data["name"]),
                data["version"],
                json.dumps(data.get("requires_dist", []), separators=(",", ":")),
                data.get("requires_python", ""),
                upload_time.timestamp(),
                bytearray.fromhex(data.get("sha256_digest", "")),
            )
        )

        if i % 5000 == 0:
            insert(entries)
            entries = []
            print(i)
insert(entries)

# For future reference. Should probably be stored in the db too.
with open("last_entry.json", "wb") as f:
    f.write(newest_line)

# Delete the upload_time, cause it's further unused; in small deltas we can dedupe ourselves.
c.execute("ALTER TABLE versions DROP COLUMN upload_time")
conn.commit()
c.execute("VACUUM")
conn.commit()
