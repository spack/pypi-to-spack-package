#!/usr/bin/env python3

"""
Automate fetching PyPI distribution metadata from BigQuery public dataset
and importing it into local SQLite database.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

try:
    from google.cloud import bigquery
    from google.cloud import storage
except ImportError:
    print(
        "Google Cloud libraries not found. Install with:\n"
        "  pip install google-cloud-bigquery google-cloud-storage",
        file=sys.stderr,
    )
    sys.exit(1)


DISTRIBUTIONS_QUERY = """
EXPORT DATA OPTIONS(
  compression="GZIP",
  uri="gs://{bucket}/pypi-distributions/pypi-*.json.gz",
  format="JSON",
  overwrite=true
)

AS

SELECT
  -- https://packaging.python.org/en/latest/specifications/name-normalization/
  REGEXP_REPLACE(LOWER(x.name), "[-_.]+", "-") AS normalized_name,
  x.version,
  x.requires_dist,
  x.requires_python,
  x.sha256_digest,
  x.path

FROM `bigquery-public-data.pypi.distribution_metadata` AS x

-- Do not use a universal wheel if there are platform specific wheels (e.g. black can be built
-- both binary and pure python, in that case prefer sdist)
LEFT JOIN `bigquery-public-data.pypi.distribution_metadata` AS y ON (
  REGEXP_REPLACE(LOWER(x.name), "[-_.]+", "-") = REGEXP_REPLACE(LOWER(y.name), "[-_.]+", "-")
  AND x.version = y.version
  AND x.packagetype = "bdist_wheel"
  AND y.packagetype = "bdist_wheel"
  AND y.path NOT LIKE "%-none-any.whl"
)

-- Select sdist and universal wheels
WHERE (
  x.packagetype = "sdist"
  OR x.path LIKE "%py3-none-any.whl"
) AND y.name IS NULL{upload_time_filter}

-- Only pick the last (re)upload of (name, version, packagetype) tuples
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY normalized_name, x.version, x.packagetype
  ORDER BY x.upload_time DESC
) = 1

-- If there are both universal wheels and sdist, pick the wheel
AND ROW_NUMBER() OVER (
  PARTITION BY normalized_name, x.version
  ORDER BY CASE WHEN x.packagetype = 'bdist_wheel' THEN 0 ELSE 1 END
) = 1
"""

VERSIONS_QUERY = """
EXPORT DATA OPTIONS(
  compression="GZIP",
  uri="gs://{bucket}/pypi-versions/pypi-*.json.gz",
  format="JSON",
  overwrite=true
)

AS

SELECT
  REGEXP_REPLACE(LOWER(name), "[-_.]+", "-") AS normalized_name,
  version
FROM `bigquery-public-data.pypi.distribution_metadata`
{upload_time_filter}
GROUP BY normalized_name, version
"""


def run_bigquery_export(
    project_id, bucket, query, query_name, upload_time_filter="", use_where=False
):
    """Run a BigQuery export query and wait for completion."""
    client = bigquery.Client(project=project_id) if project_id else bigquery.Client()

    # Format the query with bucket name and optional time filter
    if upload_time_filter:
        if use_where:
            upload_time_filter = f'\nWHERE upload_time >= "{upload_time_filter}"'
        else:
            upload_time_filter = f'\nAND x.upload_time >= "{upload_time_filter}"'
    else:
        upload_time_filter = ""

    formatted_query = query.format(bucket=bucket, upload_time_filter=upload_time_filter)

    print(f"Running {query_name} export query...")
    print(f"Destination: gs://{bucket}/")

    try:
        query_job = client.query(formatted_query)

        # Wait for the job to complete
        print("Waiting for export to complete...")
        while not query_job.done():
            time.sleep(5)
            print(".", end="", flush=True)

        print("\n")

        # Check for errors
        if query_job.errors:
            print(f"Errors occurred: {query_job.errors}", file=sys.stderr)
            return False

        print(f"✓ {query_name} export completed successfully")
        if query_job.num_dml_affected_rows:
            print(f"  Exported {query_job.num_dml_affected_rows} rows")

        return True

    except Exception as e:
        print(f"Error running query: {e}", file=sys.stderr)
        return False


def download_gcs_files(bucket, prefix, destination, cleanup_gcs=True):
    """Download files from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket_obj = storage_client.bucket(bucket)

    # List all files with the prefix
    blobs = list(bucket_obj.list_blobs(prefix=prefix))

    if not blobs:
        print(f"No files found at gs://{bucket}/{prefix}", file=sys.stderr)
        return False

    print(f"Downloading {len(blobs)} files from gs://{bucket}/{prefix}...")

    # Clear and recreate destination directory
    if os.path.exists(destination):
        print(f"Clearing existing directory: {destination}/")
        shutil.rmtree(destination)
    os.makedirs(destination)

    for i, blob in enumerate(blobs, 1):
        filename = os.path.basename(blob.name)
        dest_path = os.path.join(destination, filename)

        print(f"[{i}/{len(blobs)}] {filename}", end="", flush=True)
        blob.download_to_filename(dest_path)
        print(" ✓")

    print(f"✓ Downloaded all files to {destination}/")

    # Delete files from GCS to save costs
    if cleanup_gcs:
        print(f"\nDeleting files from gs://{bucket}/{prefix}...")
        for i, blob in enumerate(blobs, 1):
            print(f"[{i}/{len(blobs)}] Deleting {blob.name}", end="", flush=True)
            blob.delete()
            print(" ✓")
        print(f"✓ Cleaned up GCS bucket")

    return True


def download_using_gsutil(bucket, prefix, destination, cleanup_gcs=True):
    """Fallback: download files using gsutil command."""
    print(f"Downloading files using gsutil...")

    # Clear and recreate destination directory
    if os.path.exists(destination):
        print(f"Clearing existing directory: {destination}/")
        shutil.rmtree(destination)
    os.makedirs(destination)

    # Use wildcard to copy files, not the directory itself
    cmd = ["gsutil", "-m", "cp", f"gs://{bucket}/{prefix}/*", destination]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"✓ Downloaded files to {destination}/")

        # Delete files from GCS to save costs
        if cleanup_gcs:
            print(f"\nDeleting files from gs://{bucket}/{prefix}...")
            rm_cmd = ["gsutil", "-m", "rm", "-r", f"gs://{bucket}/{prefix}"]
            subprocess.run(rm_cmd, check=True, capture_output=True, text=True)
            print(f"✓ Cleaned up GCS bucket")

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running gsutil: {e}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return False
    except FileNotFoundError:
        print("gsutil not found. Please install Google Cloud SDK.", file=sys.stderr)
        return False


def import_to_database(distributions=True, versions=True):
    """Import downloaded data into SQLite database."""
    print("\nImporting data into SQLite database...")

    # Import the import_db module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    try:
        from pypi_to_spack_package import import_db
    except ImportError:
        print("Could not import import_db module", file=sys.stderr)
        return False

    if distributions:
        import_db.import_distributions()

    if versions:
        import_db.import_versions()

    print("✓ Import completed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch PyPI distribution metadata from BigQuery and import to local database"
    )
    parser.add_argument(
        "--project",
        help="Google Cloud project ID with BigQuery enabled (uses gcloud default if not specified)",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="Google Cloud Storage bucket name for temporary export files",
    )
    parser.add_argument(
        "--no-distributions",
        dest="distributions",
        action="store_false",
        default=True,
        help="Skip exporting distribution metadata",
    )
    parser.add_argument(
        "--no-versions",
        dest="versions",
        action="store_false",
        default=True,
        help="Skip exporting version list",
    )
    parser.add_argument(
        "--since",
        default="2025-01-01",
        help="Only export data uploaded since this date (YYYY-MM-DD format, default: 2025-01-01)",
    )
    parser.add_argument(
        "--no-import",
        action="store_true",
        help="Skip importing into SQLite database",
    )

    args = parser.parse_args()

    # Validate date format
    try:
        datetime.strptime(args.since, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Use YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    success = True

    # Run BigQuery exports
    if args.distributions:
        if not run_bigquery_export(
            args.project,
            args.bucket,
            DISTRIBUTIONS_QUERY,
            "distributions",
            args.since,
            use_where=False,
        ):
            success = False

    if args.versions and success:
        if not run_bigquery_export(
            args.project,
            args.bucket,
            VERSIONS_QUERY,
            "versions",
            args.since,
            use_where=True,
        ):
            success = False

    if not success:
        print("\nExport failed", file=sys.stderr)
        sys.exit(1)

    # Download files
    print("\n" + "=" * 80)
    print("DOWNLOADING FILES")
    print("=" * 80 + "\n")

    if args.distributions:
        success = download_using_gsutil(
            args.bucket, "pypi-distributions", "pypi-distributions"
        )
        if not success:
            sys.exit(1)

    if args.versions:
        success = download_using_gsutil(args.bucket, "pypi-versions", "pypi-versions")
        if not success:
            sys.exit(1)

    # Import to database
    if not args.no_import:
        print("\n" + "=" * 80)
        print("IMPORTING TO DATABASE")
        print("=" * 80 + "\n")

        if not import_to_database(args.distributions, args.versions):
            sys.exit(1)

    print("\n" + "=" * 80)
    print("✓ ALL OPERATIONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
