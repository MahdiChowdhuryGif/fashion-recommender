from pathlib import Path
import shutil

import pandas as pd

# ----------------------------------
# Configuration
# ----------------------------------

SUBSET_SIZE = 2000       # Change this if you want a different deployment size
RANDOM_SEED = 42         # Keeps the same random sample every run

# ----------------------------------
# Paths
# ----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_IMAGES = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "valid_images.csv"
)

RAW_IMAGES = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "images"
)

DEPLOYMENT_FOLDER = (
    PROJECT_ROOT /
    "data" /
    "deployment"
)

DEPLOYMENT_IMAGES = (
    DEPLOYMENT_FOLDER /
    "images"
)

DEPLOYMENT_CSV = (
    DEPLOYMENT_FOLDER /
    "valid_images.csv"
)

# ----------------------------------
# Create folders
# ----------------------------------

if DEPLOYMENT_IMAGES.exists():

    shutil.rmtree(DEPLOYMENT_IMAGES)

DEPLOYMENT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

DEPLOYMENT_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)
# ----------------------------------
# Load cleaned image list
# ----------------------------------

print("\nLoading cleaned image list...")

df = pd.read_csv(VALID_IMAGES)

print(f"Dataset contains {len(df):,} valid images.")

# ----------------------------------
# Validate subset size
# ----------------------------------

if SUBSET_SIZE > len(df):

    raise ValueError(
        f"Subset size ({SUBSET_SIZE}) is larger than "
        f"the dataset ({len(df)})."
    )

# ----------------------------------
# Select random subset
# ----------------------------------

print(
    f"\nSelecting {SUBSET_SIZE:,} images "
    f"(Random Seed = {RANDOM_SEED})..."
)

subset = df.sample(
    n=SUBSET_SIZE,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# ----------------------------------
# Save deployment CSV
# ----------------------------------

subset.to_csv(
    DEPLOYMENT_CSV,
    index=False
)

print("Deployment CSV created.")

# ----------------------------------
# Copy images
# ----------------------------------

print("\nCopying deployment images...")

copied = 0
missing = 0

for filename in subset["filename"]:

    source = RAW_IMAGES / filename
    destination = DEPLOYMENT_IMAGES / filename

    if source.exists():

        shutil.copy2(
            source,
            destination
        )

        copied += 1

    else:

        missing += 1

        print(f"Missing image: {filename}")

# ----------------------------------
# Summary
# ----------------------------------

print("\n----------------------------------------")
print("Deployment Dataset Created Successfully")
print("----------------------------------------")

print(f"Images selected : {len(subset):,}")
print(f"Images copied   : {copied:,}")
print(f"Missing images  : {missing:,}")

print("\nCreated:")

print(f"CSV    : {DEPLOYMENT_CSV}")
print(f"Images : {DEPLOYMENT_IMAGES}")