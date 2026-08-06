from pathlib import Path
import shutil

import pandas as pd
from PIL import Image

# ----------------------------------
# Configuration
# ----------------------------------

SUBSET_SIZE = 10000
RANDOM_SEED = 42

THUMBNAIL_SIZE = (256, 256)
JPEG_QUALITY = 85

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
# Prepare deployment folder
# ----------------------------------

if DEPLOYMENT_FOLDER.exists():
    shutil.rmtree(DEPLOYMENT_FOLDER)

DEPLOYMENT_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------
# Load cleaned dataset
# ----------------------------------

print("\nLoading cleaned image list...")

df = pd.read_csv(VALID_IMAGES)

print(f"Dataset contains {len(df):,} cleaned images.")

# ----------------------------------
# Validate configuration
# ----------------------------------

if SUBSET_SIZE > len(df):

    raise ValueError(
        f"Subset size ({SUBSET_SIZE}) exceeds "
        f"dataset size ({len(df)})."
    )

# ----------------------------------
# Create reproducible subset
# ----------------------------------

print(
    f"\nSelecting {SUBSET_SIZE:,} deployment images..."
)

subset = (
    df.sample(
        n=SUBSET_SIZE,
        random_state=RANDOM_SEED
    )
    .sort_values("filename")
    .reset_index(drop=True)
)

subset.to_csv(
    DEPLOYMENT_CSV,
    index=False
)

print("Deployment CSV created.")

# ----------------------------------
# Generate deployment thumbnails
# ----------------------------------

print("\nGenerating deployment images...")

processed = 0
missing = 0
failed = 0

for filename in subset["filename"]:

    source = RAW_IMAGES / filename
    destination = DEPLOYMENT_IMAGES / filename

    if not source.exists():

        missing += 1
        print(f"Missing image: {filename}")
        continue

    try:

        with Image.open(source) as image:

            image = image.convert("RGB")

            image.thumbnail(
                THUMBNAIL_SIZE,
                Image.Resampling.LANCZOS
            )

            image.save(
                destination,
                format="JPEG",
                quality=JPEG_QUALITY,
                optimize=True
            )

        processed += 1

    except Exception as e:

        failed += 1
        print(f"Failed: {filename} ({e})")

# ----------------------------------
# Verify deployment dataset
# ----------------------------------

deployment_images = list(
    DEPLOYMENT_IMAGES.glob("*.jpg")
)

# ----------------------------------
# Summary
# ----------------------------------

print("\n========================================")
print("Deployment Dataset Created Successfully")
print("========================================")

print(f"Subset size          : {SUBSET_SIZE:,}")
print(f"Images processed     : {processed:,}")
print(f"Images verified      : {len(deployment_images):,}")
print(f"Missing source files : {missing:,}")
print(f"Failed conversions   : {failed:,}")

print("\nCreated:")

print(f"CSV    : {DEPLOYMENT_CSV}")
print(f"Images : {DEPLOYMENT_IMAGES}")

print("\nNext step:")

print(
    "Run the embedding generator in deployment mode "
    "to create deployment embeddings."
)