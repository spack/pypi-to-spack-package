# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import json
import sqlite3
import sys

from datetime import datetime
from packaging.requirements import Requirement, InvalidRequirement

json_file = sys.argv[1]

conn = sqlite3.connect("data.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS nodes (
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT NOT NULL,
version TEXT NOT NULL,
requires_python TEXT,
upload_time DOUBLE PRECISION,
sha256 BLOB(32) NOT NULL
)
""")

c.execute(
    """
CREATE UNIQUE INDEX IF NOT EXISTS name_version ON nodes (name, version)
"""
)

c.execute(
"""
CREATE TABLE IF NOT EXISTS edges (
parent INTEGER,
child TEXT NOT NULL,
extras TEXT,
specifier TEXT,
marker TEXT,
FOREIGN KEY(parent) REFERENCES nodes(id)
)
"""
)


def insert(entries):
    # Insert entries into the database, updating if the upload time is newer.
    c.executemany(
"""
INSERT INTO nodes (name, version, requires_python, upload_time, sha256)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(name, version) DO UPDATE SET
requires_python = excluded.requires_python,
upload_time = excluded.upload_time,
sha256 = excluded.sha256
WHERE excluded.upload_time > upload_time
""",
        entries,
    )
    conn.commit()

def parse_date(date_string):
    formats = ['%Y-%m-%d %H:%M:%S.%f %Z', '%Y-%m-%d %H:%M:%S %Z']
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            pass
    raise ValueError("Unable to parse date string")

def requirement_parts(requirement_string):
    r = Requirement(requirement_string)
    name = r.name
    extras = ",".join(sorted(r.extras)) if r.extras else None
    specifier = str(r.specifier) if r.specifier else None
    marker = str(r.marker) if r.marker else None
    return (name, extras, specifier, marker)

def insert_edges(entries):
    c.executemany(
        """
        INSERT INTO edges (parent, child, extras, specifier, marker)
        VALUES (?, ?, ?, ?, ?)
        """,
        entries,
    )
    conn.commit()


# First insert all nodes.
with open(json_file, "rb") as f:
    # entries = []
    # keys = set()
    # last_date_time = datetime.min
    # last_date_entry = None
    # for i, line in enumerate(f.readlines()):
    #     data = json.loads(line)

    #     # We don't ever store this, to save some bytes.
    #     date_time = parse_date(data["upload_time"])

    #     # Keep track of the last date so that future updates can be incremental from this point.
    #     if date_time > last_date_time:
    #         last_date_time = date_time
    #         last_date_entry = line

    #     entries.append(
    #         (
    #             data["name"],
    #             data["version"],
    #             data.get("requires_python", ""),
    #             date_time.timestamp(),
    #             bytearray.fromhex(data.get("sha256_digest", "")),
    #         )
    #     )

    #     if len(entries) >= 5000:
    #         insert(entries)
    #         entries = []
    #         print(i + 1)
    # insert(entries)
    # print(last_date_entry)

    # conn.commit()

    f.seek(0)
    
    # TODO: badged
    # Delete all edges
    c.execute("DELETE FROM edges")
    conn.commit()
    entries = []

    batch_size = 10000
    count = 0

    while True:
        batch = [
            x
            for _ in range(batch_size)
            if (x := json.loads(f.readline()))
        ]

        keys = [
            (x["name"], x["version"], parse_date(x["upload_time"]).timestamp())
            for x in batch
        ]

        if not batch:
            break

        query = f"SELECT id, name, version FROM nodes WHERE (name, version, upload_time) IN ({','.join('(?,?,?)' for _ in batch)})"
        result = c.execute(query, list(y for x in keys for y in x)).fetchall()

        name_version_to_id = {
            (name, version): id
            for id, name, version in result
        }

        for data in batch:
            
            id = name_version_to_id.get((data["name"], data["version"]))

            # Apparently a re-upload, skip.
            if not id:
                continue

            try:
                entries.extend((id, *requirement_parts(entry)) for entry in data["requires_dist"])
            except InvalidRequirement as e:
                # Just delete the entry, cause it's broken.
                print(f"Deleting {data['name']} {data['version']} because it is broken: {e}")
                c.execute("DELETE FROM nodes WHERE id = ?", (id,))
                conn.commit()

        count += batch_size

        print(count)

        if len(entries) >= 5000:
            insert_edges(entries)
            entries = []
    
    if entries:
        insert_edges(entries)

# Then some normalization... to do. Have done it by hand only so far.

# CREATE TABLE names (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);
# CREATE UNIQUE INDEX id_name ON names (name);
# INSERT INTO names (name) SELECT DISTINCT name FROM nodes;
        
# ALTER TABLE nodes ADD COLUMN name_id INTEGER;
# UPDATE nodes SET name_id = (SELECT id FROM names WHERE name = nodes.name);
# ALTER TABLE nodes DROP COLUMN name;
        
# ALTER TABLE edges ADD COLUMN name_id INTEGER;
# UPDATE edges SET name_id = (SELECT id FROM names WHERE name = edges.child);
# ALTER TABLE edges DROP COLUMN child;

# VACUUM;