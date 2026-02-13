#!/usr/bin/env python3

# Find updates, fetch SHA256 checksums, and add version updates to Spack package.py files

import argparse
import asyncio
import hashlib
import json
import itertools
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp

try:
    from spack.repo import PATH as repo
    from spack.spec import Spec
    import packaging.version as pv
    import spack.repo
except ImportError:
    print(
        "Spack Python modules could not be imported, "
        "make sure $spack/lib/spack is in your PYTHONPATH",
        file=sys.stderr,
    )
    sys.exit(1)


VERY_OLD = datetime.now() - timedelta(days=365 * 3)  # 3 years ago
DISALLOWED_DEPS = {"c", "cxx", "fortran", "rust"}
CACHE_DIR = Path("shasums")
CACHE_DIR.mkdir(exist_ok=True)



def get_pypi_versions(cursor, package_name):
    """Get all versions of a PyPI package from data.db

    Returns: (versions_list, latest_upload_time)
    """
    query = cursor.execute(
        "SELECT version, upload_time FROM versions WHERE name = ?",
        (package_name,),
    )
    versions = []
    version_times = {}  # Map version to upload_time
    for version_str, upload_time in query:
        try:
            v = pv.parse(version_str)
            # Skip pre-releases, dev, post, local versions
            if v.pre is None and v.dev is None and v.post is None and v.local is None:
                versions.append(v)
                version_times[v] = upload_time
        except pv.InvalidVersion:
            continue
    # Sort versions in descending order using semantic versioning
    versions.sort(reverse=True)

    # Get upload_time of latest version
    latest_upload_time = version_times.get(versions[0]) if versions else None

    return (versions, latest_upload_time)


def get_distribution_metadata(cursor, package_name, version_str):
    """Get requires_dist and requires_python for a specific package version"""
    query = cursor.execute(
        "SELECT requires_dist, requires_python FROM distributions WHERE name = ? AND version = ?",
        (package_name, version_str),
    )
    result = query.fetchone()
    if result:
        return result  # (requires_dist, requires_python)
    return None


def find_updates(db_path="data.db", patch_only=False, min_days_since_release=None):
    """Find Python packages that have newer versions available on PyPI

    Args:
        db_path: Path to SQLite database
        patch_only: If True, only allow patch version bumps (e.g., 1.2.3 -> 1.2.7)
        min_days_since_release: If set, only include packages with latest release within this many days

    Returns: list of package update dicts
    """
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"{'=' * 100}")
    print("FINDING PACKAGE UPDATES")
    print(f"{'=' * 100}\n")

    print("Connecting to database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all Python packages from Spack
    print(
        f"Loading Spack Python packages that don't depend on {', '.join(DISALLOWED_DEPS)}..."
    )
    python_packages = [
        pkg.name
        for pkg in spack.repo.PATH.all_package_classes()
        if pkg.name.startswith("py-")
        and DISALLOWED_DEPS.isdisjoint(
            itertools.chain.from_iterable(pkg.dependencies.values())
        )
    ]

    print(f"Found {len(python_packages)} Python packages in Spack\n")

    updates = []  # All updates with flag for Python requirement changes
    errors = []
    not_found = []
    up_to_date = []
    very_old = []
    too_old = []  # Packages filtered out by min_days_since_release

    for i, spack_name in enumerate(sorted(python_packages), 1):
        if i % 100 == 0:
            print(
                f"\rProcessed {i}/{len(python_packages)} packages...",
                end="",
                flush=True,
            )

        # Convert spack name to PyPI name: py-numpy -> numpy
        pypi_name = spack_name[3:]  # Remove "py-"

        try:
            pkg = spack.repo.PATH.get_pkg_class(spack_name)
            if not pkg.versions:
                continue

            # Get latest version in Spack
            spack_latest = max(pkg.versions)
            spack_latest_str = str(spack_latest)

            # Parse with packaging
            try:
                spack_v = pv.parse(spack_latest_str)
            except pv.InvalidVersion:
                continue

            # Get versions from PyPI
            pypi_versions, latest_upload_time = get_pypi_versions(cursor, pypi_name)
            if not pypi_versions:
                not_found.append(spack_name)
                continue

            # Find latest PyPI version
            pypi_latest = pypi_versions[0]

            # Check if package meets minimum recency requirement
            if min_days_since_release and latest_upload_time:
                try:
                    upload_dt = datetime.fromisoformat(
                        latest_upload_time.replace("Z", "+00:00").split(".")[0]
                    )
                    if upload_dt.tzinfo is not None:
                        upload_dt = upload_dt.replace(tzinfo=None)
                    min_date = datetime.now() - timedelta(days=min_days_since_release)
                    if upload_dt < min_date:
                        too_old.append(spack_name)
                        continue
                except:
                    pass

            # Warn if latest version is older than 3 years
            if latest_upload_time:
                try:
                    upload_dt = datetime.fromisoformat(
                        latest_upload_time.replace("Z", "+00:00").split(".")[0]
                    )
                    if upload_dt.tzinfo is not None:
                        upload_dt = upload_dt.replace(tzinfo=None)
                    if upload_dt < VERY_OLD:
                        very_old.append((upload_dt.date(), spack_name, pypi_latest))
                except:
                    pass

            # Only check if newer than Spack version
            if pypi_latest > spack_v:
                # If patch_only mode, ensure major and minor versions match
                if patch_only:
                    if (
                        pypi_latest.major != spack_v.major
                        or pypi_latest.minor != spack_v.minor
                    ):
                        up_to_date.append(spack_name)
                        continue

                # Check if metadata matches
                spack_metadata = get_distribution_metadata(
                    cursor, pypi_name, str(spack_v)
                )
                pypi_metadata = get_distribution_metadata(
                    cursor, pypi_name, str(pypi_latest)
                )

                if spack_metadata and pypi_metadata:
                    spack_requires_dist, spack_requires_python = spack_metadata
                    pypi_requires_dist, pypi_requires_python = pypi_metadata

                    # Only include if requires_dist and requires_python are identical
                    if (
                        spack_requires_dist == pypi_requires_dist
                        and spack_requires_python == pypi_requires_python
                    ):
                        updates.append(
                            {
                                "spack_name": spack_name,
                                "pypi_name": pypi_name,
                                "old_version": str(spack_v),
                                "new_version": str(pypi_latest),
                                "old_python": spack_requires_python or "",
                                "new_python": pypi_requires_python or "",
                                "python_requirements_changed": False,
                                "upload_time": latest_upload_time,
                            }
                        )
                    # Track packages with same requires_dist but different requires_python
                    elif spack_requires_dist == pypi_requires_dist:
                        updates.append(
                            {
                                "spack_name": spack_name,
                                "pypi_name": pypi_name,
                                "old_version": str(spack_v),
                                "new_version": str(pypi_latest),
                                "old_python": spack_requires_python or "",
                                "new_python": pypi_requires_python or "",
                                "python_requirements_changed": True,
                                "upload_time": latest_upload_time,
                            }
                        )
                    else:
                        up_to_date.append(spack_name)
                else:
                    up_to_date.append(spack_name)
            else:
                up_to_date.append(spack_name)

        except Exception as e:
            errors.append((spack_name, str(e)))

    print()  # Clear the progress line
    print(f"\n{'=' * 100}")
    print("VERSION UPDATES AVAILABLE")
    print(f"{'=' * 100}\n")
    print(f"Found {len(updates)} packages with updates\n")

    # Separate for display
    identical_updates = [u for u in updates if not u["python_requirements_changed"]]
    python_changed_updates = [u for u in updates if u["python_requirements_changed"]]

    if identical_updates:
        print(
            f"Identical dependencies and Python requirements: {len(identical_updates)}\n"
        )
        for pkg in sorted(identical_updates, key=lambda x: x["spack_name"])[:20]:
            print(
                f"{pkg['spack_name']:45} {pkg['old_version']} -> {pkg['new_version']}"
            )
        if len(identical_updates) > 20:
            print(f"... and {len(identical_updates) - 20} more")

    if python_changed_updates:
        print(f"\nPython requirements changed: {len(python_changed_updates)}\n")
        for pkg in sorted(python_changed_updates, key=lambda x: x["spack_name"])[:20]:
            print(
                f"{pkg['spack_name']:45} {pkg['old_version']} -> {pkg['new_version']}"
            )
            old_py = pkg["old_python"] or "(none)"
            new_py = pkg["new_python"] or "(none)"
            print(f"  requires_python: {old_py} -> {new_py}")
        if len(python_changed_updates) > 20:
            print(f"... and {len(python_changed_updates) - 20} more")

    for upload_dt, spack_name, pypi_latest in sorted(very_old, reverse=True):
        # Calculate relative age
        days_old = (datetime.now().date() - upload_dt).days
        years = days_old // 365
        months = (days_old % 365) // 30

        if years > 0 and months > 0:
            age_str = f"{years} year{'s' if years > 1 else ''}, {months} month{'s' if months > 1 else ''} old"
        elif years > 0:
            age_str = f"{years} year{'s' if years > 1 else ''} old"
        else:
            age_str = f"{months} month{'s' if months > 1 else ''} old"

        print(
            f"WARNING: {spack_name} latest version {pypi_latest} is {age_str} ({upload_dt})"
        )

    if errors:
        print(f"\n{'=' * 100}")
        print(f"ERRORS: {len(errors)}")
        print(f"{'=' * 100}\n")
        for pkg, error in errors[:10]:
            print(f"{pkg}: {error}")
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more")

    # Summary statistics
    print(f"\n{'=' * 100}")
    print("FIND UPDATES SUMMARY")
    print(f"{'=' * 100}")
    print(f"* {len(updates)} versions can be updated")
    print(
        f"  - {len(identical_updates)} with identical dependencies and Python requirements"
    )
    print(f"  - {len(python_changed_updates)} with Python requirement changes only")
    print(f"* {len(not_found)} packages were not found in the database")
    print(f"* {len(up_to_date)} versions are up to date")
    if min_days_since_release:
        print(
            f"* {len(too_old)} packages filtered (no release within {min_days_since_release} days)"
        )
    print(f"* {len(very_old)} packages have latest version older than 3 years")
    print()

    conn.close()
    return updates


def url_to_cache_filename(url: str) -> Path:
    """Convert URL to a cache filename using its hash"""
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    return CACHE_DIR / url_hash


def get_cached_shasum(url: str) -> str | None:
    """Get cached SHA256 sum if it exists"""
    cache_file = url_to_cache_filename(url)
    if cache_file.exists():
        return cache_file.read_text().strip()
    return None


def cache_shasum(url: str, shasum: str) -> None:
    """Cache a SHA256 sum"""
    cache_file = url_to_cache_filename(url)
    cache_file.write_text(shasum)


async def compute_shasum(
    session: aiohttp.ClientSession, url: str
) -> tuple[str, str, str | None]:
    """Download URL and compute SHA256 sum without storing to disk

    Returns: (url, sha256sum or error, status)
    """
    # Check cache first
    cached = get_cached_shasum(url)
    if cached:
        return (url, cached, "cached")

    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                return (url, f"HTTP {response.status}", "error")

            # Stream and hash the content
            hasher = hashlib.sha256()
            async for chunk in response.content.iter_chunked(8192):
                hasher.update(chunk)

            shasum = hasher.hexdigest()
            cache_shasum(url, shasum)
            return (url, shasum, "downloaded")

    except asyncio.TimeoutError:
        return (url, "timeout", "error")
    except Exception as e:
        return (url, str(e), "error")


async def fetch_all_shasums(packages: list[dict]) -> list[dict]:
    """Fetch SHA256 sums for all packages

    Args:
        packages: List of package dicts with spack_name, pypi_name, new_version, etc.

    Returns:
        List of package dicts with added sha256 and url fields, or error info
    """
    results = []

    # Create semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(20)

    async with aiohttp.ClientSession() as session:
        tasks = []

        for pkg in packages:
            spack_name = pkg["spack_name"]
            new_version = pkg["new_version"]

            try:
                # Get package class and create spec
                pkg_class = repo.get_pkg_class(spack_name)
                spec = Spec(f"{spack_name}@={new_version}")
                spec._mark_concrete(True)
                pkg_instance = pkg_class(spec)

                # Get URL from fetcher
                url = pkg_instance.fetcher.url

                # Skip if URL is None
                if url is None:
                    pkg_result = pkg.copy()
                    pkg_result["error"] = "URL is None"
                    pkg_result["status"] = "error"
                    results.append(pkg_result)
                    continue

                # Create task with semaphore
                async def fetch_with_semaphore(package_dict, url):
                    async with semaphore:
                        url_result, shasum_or_error, status = await compute_shasum(
                            session, url
                        )
                        result = package_dict.copy()
                        result["url"] = url_result
                        if status == "error":
                            result["error"] = shasum_or_error
                            result["status"] = "error"
                        else:
                            result["sha256"] = shasum_or_error
                            result["status"] = status
                        return result

                task = fetch_with_semaphore(pkg, url)
                tasks.append(task)

            except Exception as e:
                pkg_result = pkg.copy()
                pkg_result["error"] = str(e)
                pkg_result["status"] = "error"
                results.append(pkg_result)

        # Wait for all downloads to complete
        if tasks:
            results.extend(await asyncio.gather(*tasks, return_exceptions=False))

    return results


def find_first_version_line(content):
    """Find the first version() line in package.py content

    Returns: (line_number, version_string) or (None, None)
    """
    lines = content.split("\n")
    # Match:     version("3.6.2", sha256="...")
    pattern = re.compile(r'^    version\("([^"]+)",')

    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            return (i, match.group(1))

    return (None, None)


def add_version_line(
    package_py_path, new_version, sha256, current_version, python_req_comment=None
):
    """Add a new version line above the current latest version

    Args:
        python_req_comment: Optional tuple of (old_python, new_python) to add as comment

    Returns: (success: bool, message: str)
    """
    # Read the file
    try:
        content = package_py_path.read_text()
    except Exception as e:
        return (False, f"Failed to read file: {e}")

    # Find first version line
    line_num, found_version = find_first_version_line(content)

    if line_num is None:
        return (False, "No version() line found in package.py")

    # Verify it matches the current version we expect
    try:
        found_v = pv.parse(found_version)
        current_v = pv.parse(current_version)

        # They should match (current version should be the latest in the file)
        if found_v != current_v:
            return (
                False,
                f"Version mismatch: found {found_version} in file, expected {current_version}",
            )
    except Exception as e:
        return (False, f"Version parsing failed: {e}")

    # Parse with version to determine if it's actually newer
    try:
        new_v = pv.parse(new_version)
        if new_v <= current_v:
            return (
                False,
                f"New version {new_version} is not newer than {current_version}",
            )
    except Exception as e:
        return (False, f"Failed to parse new version: {e}")

    # Insert the new version line
    lines = content.split("\n")

    # Add comment if Python requirements changed
    if python_req_comment:
        old_py, new_py = python_req_comment
        old_py_str = old_py if old_py else "(none)"
        new_py_str = new_py if new_py else "(none)"
        comment_line = f"    # requires_python: {old_py_str} -> {new_py_str}"
        lines.insert(line_num, comment_line)
        line_num += 1  # Adjust insertion point for version line

    new_line = f'    version("{new_version}", sha256="{sha256}")'
    lines.insert(line_num, new_line)

    # Write back
    try:
        package_py_path.write_text("\n".join(lines))
        return (True, f"Added version {new_version}")
    except Exception as e:
        return (False, f"Failed to write file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Find updates, fetch SHA256 checksums, and add version updates to Spack package.py files"
    )
    parser.add_argument(
        "updates_file",
        nargs="?",
        help="Input JSON file with package versions (optional - if not provided, will find updates first)",
    )
    parser.add_argument(
        "--find-only",
        action="store_true",
        help="Only find updates and save to JSON, don't fetch checksums or add versions",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch checksums, don't add versions to package.py files (requires input file)",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate JSON with checksums",
    )
    parser.add_argument(
        "--db", default="data.db", help="Path to SQLite database (default: data.db)"
    )
    parser.add_argument(
        "--allow-python-changes",
        action="store_true",
        help="Allow adding versions where Python requirements changed (default: only add versions with identical Python requirements)",
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="Only allow patch version bumps (e.g., 1.2.3 -> 1.2.7, but not 1.2.3 -> 1.3.0 or 2.0.0)",
    )
    parser.add_argument(
        "--min-days-since-release",
        type=int,
        default=365,
        help="Only update packages with a release within this many days (default: 365, set to 0 to disable)",
    )

    args = parser.parse_args()

    # Step 1: Find updates (if no input file provided)
    if args.updates_file:
        # User provided an input file - skip find step
        input_file = args.updates_file
        if not Path(input_file).exists():
            print(f"{input_file} not found.", file=sys.stderr)
            sys.exit(1)

        with open(input_file) as f:
            packages = json.load(f)
    else:
        # No input file - find updates first
        min_days = (
            args.min_days_since_release if args.min_days_since_release > 0 else None
        )
        packages = find_updates(
            args.db, patch_only=args.patch_only, min_days_since_release=min_days
        )

        if not packages:
            print("No updates found.")
            return

        # Save to updates.json
        with open("updates.json", "w") as f:
            json.dump(packages, f, indent=2)
        print("Updates written to: updates.json")

        # Stop here if find-only mode
        if args.find_only:
            return

        input_file = "updates.json"

    print(f"{'=' * 100}")
    print("FETCHING SHA256 CHECKSUMS")
    print(f"{'=' * 100}\n")
    print(f"Processing {len(packages)} packages")
    print(f"Cache directory: {CACHE_DIR.absolute()}")
    print()

    # Fetch checksums
    results = asyncio.run(fetch_all_shasums(packages))

    # Separate successful and failed results
    success_results = [r for r in results if r.get("status") != "error"]
    error_results = [r for r in results if r.get("status") == "error"]

    success_count = len(success_results)
    error_count = len(error_results)
    cached_count = len([r for r in success_results if r.get("status") == "cached"])

    print(f"\n{'=' * 100}")
    print("FETCH RESULTS")
    print(f"{'=' * 100}\n")

    # Print errors
    for result in error_results:
        print(
            f"ERROR: {result['spack_name']:40} {result.get('error', 'unknown error')}"
        )

    # Print new downloads
    for result in success_results:
        if result.get("status") != "cached":
            print(f"✓ {result['spack_name']:40} {result['sha256'][:16]}...")

    print(f"\nFetch summary:")
    print(f"  Total:      {len(packages)}")
    print(
        f"  Successful: {success_count} (cached: {cached_count}, downloaded: {success_count - cached_count})"
    )
    print(f"  Errors:     {error_count}")

    # Save intermediate file if requested
    if args.save_intermediate:
        output_file = (
            input_file[:-5] + "_with_shasums.json"
            if input_file.endswith(".json")
            else input_file + "_with_shasums.json"
        )
        with open(output_file, "w") as f:
            json.dump(success_results, f, indent=2)
        print(f"\nIntermediate file saved: {output_file}")

    # Write errors to separate file
    if error_results:
        error_file = "shasum_errors.json"
        with open(error_file, "w") as f:
            json.dump(error_results, f, indent=2)
        print(f"Errors saved: {error_file}")

    # Stop here if fetch-only mode
    if args.fetch_only:
        return

    # Step 3: Add versions to package.py files
    # Filter packages based on python_requirements_changed flag
    if not args.allow_python_changes:
        original_count = len(success_results)
        success_results = [
            r
            for r in success_results
            if not r.get("python_requirements_changed", False)
        ]
        skipped_count = original_count - len(success_results)
        if skipped_count > 0:
            print(
                f"\nSkipping {skipped_count} packages with Python requirement changes (use --allow-python-changes to include them)"
            )

    if not success_results:
        print("\nNo packages to process after filtering.")
        return

    print(f"\n{'=' * 100}")
    print("ADDING VERSIONS TO PACKAGE.PY FILES")
    print(f"{'=' * 100}\n")
    print(f"Processing {len(success_results)} packages\n")

    add_success_count = 0
    add_error_count = 0
    add_errors = []

    for i, pkg in enumerate(success_results, 1):
        spack_name = pkg["spack_name"]
        new_version = pkg["new_version"]
        old_version = pkg["old_version"]
        sha256 = pkg["sha256"]

        # Check if Python requirements changed
        python_req_comment = None
        if pkg.get("python_requirements_changed"):
            python_req_comment = (
                pkg.get("old_python") or "",
                pkg.get("new_python") or "",
            )

        if i % 50 == 0:
            print(
                f"\rProcessed {i}/{len(success_results)} packages...",
                end="",
                flush=True,
            )

        try:
            # Get current version from Spack
            pkg_class = repo.get_pkg_class(spack_name)
            if not pkg_class.versions:
                add_errors.append((spack_name, "No versions in package"))
                add_error_count += 1
                continue

            # Get package directory
            pkg_dir = Path(repo.dirname_for_package_name(spack_name))
            package_py = pkg_dir / "package.py"

            if not package_py.exists():
                add_errors.append((spack_name, "package.py not found"))
                add_error_count += 1
                continue

            # Add the version
            success, message = add_version_line(
                package_py, new_version, sha256, old_version, python_req_comment
            )

            if success:
                add_success_count += 1
                # Format upload date if available
                upload_info = ""
                if pkg.get("upload_time"):
                    try:
                        upload_dt = datetime.fromisoformat(
                            pkg["upload_time"].replace("Z", "+00:00").split(".")[0]
                        )
                        upload_info = f" (released {upload_dt.date()})"
                    except:
                        pass
                print(
                    f"\r✓ {spack_name:45} {old_version} -> {new_version}{upload_info}"
                )
            else:
                add_error_count += 1
                add_errors.append((spack_name, message))

        except Exception as e:
            add_error_count += 1
            add_errors.append((spack_name, str(e)))

    print(f"\n{'=' * 100}")
    print("FINAL SUMMARY")
    print(f"{'=' * 100}\n")
    print(f"Checksums fetched:      {success_count}")
    print(f"Versions added:         {add_success_count}")
    print(f"Fetch errors:           {error_count}")
    print(f"Add version errors:     {add_error_count}")

    if add_errors:
        print(f"\n{'=' * 100}")
        print("ADD VERSION ERRORS")
        print(f"{'=' * 100}\n")
        for pkg, error in add_errors[:20]:
            print(f"{pkg:45} {error}")
        if len(add_errors) > 20:
            print(f"... and {len(add_errors) - 20} more errors")

        # Write errors to file
        with open("add_version_errors.txt", "w") as f:
            for pkg, error in add_errors:
                f.write(f"{pkg}\t{error}\n")
        print(f"\nFull error list written to: add_version_errors.txt")


if __name__ == "__main__":
    main()
