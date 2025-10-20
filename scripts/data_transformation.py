"""Script to perform geospatial join of service request data with hexagonal grid polygons."""

import sys
from pathlib import Path
import logging
import argparse
import time
import pandas as pd
import geopandas as gpd

# Add the repo root to the Python path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", type=str, default="./data",)
# Failure threshold (default=5%)
# A 5% failure rate is a reasonable operational tolerance:
# - It allows for minor coordinate issues (bad GPS data, out-of-bounds requests).
# - But it will catch systemic issues (e.g. wrong CRS or missing polygons).
# In practice, a reasonable tolerance will be set taking into account the specific
# requirements of the request (e.g. the level of accuracy required and the 'cost' of an error).
parser.add_argument("-e", "--error-threshold", type=float, default=0.05)
args = parser.parse_args()


INPUT_PATH = Path(args.data_dir)/"sr.csv.gz"
HEX_POLYGONS_PATH = Path(args.data_dir)/"city-hex-polygons-8.geojson"
TEST_DATA_PATH = Path(args.data_dir)/"sr_hex.csv.gz"

FAILURE_THRESHOLD = args.error_threshold


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

start_time = time.perf_counter()

# Read service requests data
sr = pd.read_csv(INPUT_PATH)
sr_hex = pd.read_csv(TEST_DATA_PATH)

# Read geospatial data
poly8 = gpd.read_file(HEX_POLYGONS_PATH)
poly8 = poly8.rename(columns={"index": "h3_level8_index"})

# Check for uniqueness of notification_number in both datasets
assert sr["notification_number"].is_unique
assert sr_hex["notification_number"].is_unique
assert set(sr["notification_number"]) == set(sr_hex["notification_number"])

# The next step performs the geospatial join. To do the spatial join, we use
# `sjoin_nearest`from GeoPandas. For each service request, the `sjoin_nearest`
# function joins it to the nearest hexagon, based on the centroid of that hexagon.
# This is the most time-consuming step, but should not take longer than 8 minutes,
# even when benchmarked on older hardware. I did consider using the H3 Hexagonal
# Grid System along with Numpy for distance calculations to optimise for speed,
# but ultimately concluded that the potential speed gain was not worth the
# additional complexity for a dataset of this size.
df = gpd.GeoDataFrame(
    sr,
    geometry=gpd.points_from_xy(sr["longitude"], sr["latitude"]),
    crs="EPSG:4326"   # WGS84 latitude/longitude
)
df = df.to_crs("EPSG:32734")  # Convert to UTM Zone 34S

# Ensure both GeoDataFrames have the same coordinate reference system
poly8 = poly8.to_crs(df.crs)

# Perform the spatial join
# df = gpd.sjoin(df, poly8[["h3_level8_index", "geometry"]], how="left")
df = gpd.sjoin_nearest(df, poly8[["h3_level8_index", "geometry"]], how="left")
df = df.drop(columns=["index_right"])
# Set `h3_level8_index` to '0' where coordinates are missing
df.loc[df["longitude"].isna() | df["latitude"].isna(), "h3_level8_index"] = '0'

# Record the number of records that failed to join (excluding those with missing coordinates)
num_failed_joins = df["h3_level8_index"].isna().sum()

# Calculate the failure rate (excluding those with missing coordinates)
num_invalid_coords = (df["longitude"].isna() | df["latitude"].isna()).sum()
num_valid_coords = df.shape[0] - num_invalid_coords
failure_rate = num_failed_joins / num_valid_coords
logging.info(f"Total records: {df.shape[0]}")
logging.info(f"Records with missing coordinates: {num_invalid_coords}")
logging.info(f"Failed joins: {num_failed_joins} ({failure_rate*100:.2f}%)")

# Script errors out if failure rate exceeds threshold
if failure_rate > FAILURE_THRESHOLD:
    logging.error("Join failure rate exceeds threshold (%.2f%%)", failure_rate * 100)
    sys.exit(1)

# End timing
elapsed = time.perf_counter() - start_time
logging.info(f"Join completed in {elapsed:.2f} seconds")

# Compare with `sr_hex`
df_test = (
    df[["notification_number", "h3_level8_index", "longitude", "latitude"]]
    .merge(
        sr_hex[["notification_number", "h3_level8_index", "longitude", "latitude"]],
        on="notification_number", how="left", suffixes=('_calc', '_test')
    )
)
df_test["match"] = df_test["h3_level8_index_calc"] == df_test["h3_level8_index_test"]
print("Mismatched records compared to `sr_hex.csv`:")
print(df_test['match'].value_counts())