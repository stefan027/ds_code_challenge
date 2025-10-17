"""Download data from S3 bucket."""

import os
import argparse
import json
import boto3


BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUCKET = "cct-ds-code-challenge-input-data"
FILES_TO_DOWNLOAD = [
    "sr.csv.gz",
    "sr_hex.csv.gz",
    "sr_hex_truncated.csv",
    "city-hex-polygons-8.geojson",
    "city-hex-polygons-8-10.geojson"
]
DIRS_TO_DOWNLOAD = [
    "images/swimming-pool",
]
OUTPUT_DIR = os.path.join(BASEDIR, "data")


def create_s3_client(credentials_path=None):
    """Create an S3 client, optionally using credentials from a JSON file."""
    if credentials_path:
        with open(credentials_path, "r") as f:
            creds = json.load(f)["s3"]

        return boto3.client(
            "s3",
            aws_access_key_id=creds["access_key"],
            aws_secret_access_key=creds["secret_key"],
        )

    # Use default credentials
    return boto3.client("s3")


def download_file_from_s3(s3_client, bucket, key, destination, verbose=True):
    """Download a single file from S3."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if verbose:
        print(f"Downloading s3://{bucket}/{key} -> {destination}")
    s3_client.download_file(bucket, key, destination)
    if verbose:
        print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--credentials", type=str, help="Path to AWS credentials JSON file"
    )
    args = parser.parse_args()

    # Create S3 client
    s3_client = create_s3_client(args.credentials)

    # Download the files
    for filename in FILES_TO_DOWNLOAD:
        dest_fp = os.path.join(OUTPUT_DIR, filename)
        download_file_from_s3(s3_client, BUCKET, filename, dest_fp)

    for dir_prefix in DIRS_TO_DOWNLOAD:
        print("Downloading directory:", dir_prefix)
        response = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=dir_prefix)
        for obj in response.get('Contents', []):
            key = obj['Key']
            local_path = os.path.join(OUTPUT_DIR, dir_prefix, os.path.basename(key))
            download_file_from_s3(s3_client, BUCKET, key, local_path, verbose=False)
        print("Download complete.")
