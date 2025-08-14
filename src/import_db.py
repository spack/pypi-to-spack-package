#!/usr/bin/env spack-python

# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import argparse
import gzip
import json
import os
import sqlite3
import sys

MOVE_UP = "\033[1A"
CLEAR_LINE = "\x1b[2K"

conn = sqlite3.connect("data.db")
c = conn.cursor()


c.execute(
    """
CREATE TABLE IF NOT EXISTS distributions
(
name TEXT NOT NULL,
version TEXT NOT NULL,
requires_dist TEXT,
requires_python TEXT,
sha256 BLOB(32) NOT NULL,
path TEXT NOT NULL
)
"""
)

c.execute(
    """
CREATE UNIQUE INDEX IF NOT EXISTS name_index ON distributions (name, version)
"""
)

c.execute(
    """
CREATE TABLE IF NOT EXISTS versions
(
name TEXT NOT NULL,
version TEXT NOT NULL
)
"""
)

c.execute(
    """
CREATE UNIQUE INDEX IF NOT EXISTS versions_by_name ON versions (name, version)
"""
)


def insert_versions(entries):
    c.executemany(
        """
    INSERT INTO versions (name, version)
    VALUES (?, ?)
    ON CONFLICT(name, version) DO NOTHING
    """,
        entries,
    )
    conn.commit()


def insert_distributions(entries):
    c.executemany(
        """
    INSERT INTO distributions (name, version, requires_dist, requires_python, sha256, path)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(name, version) DO UPDATE SET
    requires_dist = excluded.requires_dist,
    requires_python = excluded.requires_python,
    sha256 = excluded.sha256,
    path = excluded.path
    """,
        entries,
    )
    conn.commit()


def import_versions(path="pypi-versions"):
    entries = []
    i = 0
    print("importing versions...")
    print()
    files = sorted(os.listdir(path))
    total_lines = 0
    total_files = len(files)
    for j, file in enumerate(files):
        with gzip.open(os.path.join(path, file), "rb") as f:
            lines = f.readlines()
            total_lines += len(lines)
            lines_estimate = total_lines / (j + 1) * total_files
            for line in lines:
                i += 1
                data = json.loads(line)
                entries.append((data["normalized_name"], data["version"]))

                if i % 10000 == 0:
                    insert_versions(entries)
                    entries = []
                    percent = int(i / lines_estimate * 100)
                    print(f"{MOVE_UP}{CLEAR_LINE}[{percent:3}%] {file}: {i}")
    insert_versions(entries)
    sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}")


def import_distributions(path="pypi-distributions"):
    entries = []
    i = 0
    print("importing metadata...")
    print()
    files = sorted(os.listdir(path))
    total_lines = 0
    total_files = len(files)
    for j, file in enumerate(files):
        with gzip.open(os.path.join(path, file), "rb") as f:
            lines = f.readlines()
            total_lines += len(lines)
            lines_estimate = total_lines / (j + 1) * total_files
            for line in lines:
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
                    )
                )

                if i % 10000 == 0:
                    insert_distributions(entries)
                    entries = []
                    percent = int(i / lines_estimate * 100)
                    print(f"{MOVE_UP}{CLEAR_LINE}[{percent:3}%] {file}: {i}")
    insert_distributions(entries)
    sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}")


def main():
    parser = argparse.ArgumentParser(
        description="Import PyPI BigQuery export JSON (distributions / versions) into local"
        "SQLite data.db"
    )
    parser.add_argument(
        "--distributions",
        action="store_true",
        help="Import distribution metadata (requires_dist, requires_python, sha256, path)",
    )
    parser.add_argument(
        "--versions",
        action="store_true",
        help="Import list of known versions (normalized_name, version)",
    )
    args = parser.parse_args()

    if not args.versions and not args.distributions:
        parser.print_help()
        sys.exit(1)

    if args.distributions:
        import_distributions()

    if args.versions:
        import_versions()


if __name__ == "__main__":
    main()
