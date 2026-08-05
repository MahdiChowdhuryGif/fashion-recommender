from pathlib import Path
import os
import time

import numpy as np
import pandas as pd

from models.feature_extractor import FeatureExtractor


def main():

    # ----------------------------------
    # Configuration
    # ----------------------------------

    DATASET_MODE = os.getenv("DATASET_MODE", "full").lower()

    DEPLOYMENT_MODE = DATASET_MODE == "deployment"

    TEST_MODE = False
    TEST_SIZE = 100

    # ----------------------------------
    # Paths
    # ----------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    if DEPLOYMENT_MODE:

        print("\nRunning in DEPLOYMENT mode")

        VALID_IMAGES = (
            PROJECT_ROOT /
            "data" /
            "deployment" /
            "valid_images.csv"
        )

        IMAGE_FOLDER = (
            PROJECT_ROOT /
            "data" /
            "deployment" /
            "images"
        )

    else:

        print("\nRunning in FULL DATASET mode")

        VALID_IMAGES = (
            PROJECT_ROOT /
            "data" /
            "processed" /
            "valid_images.csv"
        )

        IMAGE_FOLDER = (
            PROJECT_ROOT /
            "data" /
            "raw" /
            "images"
        )

    OUTPUT_DIR = (
        PROJECT_ROOT /
        "data" /
        "processed"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EMBEDDINGS_FILE = (
        OUTPUT_DIR /
        "image_embeddings.npy"
    )

    FILENAMES_FILE = (
        OUTPUT_DIR /
        "image_filenames.csv"
    )

    # ----------------------------------
    # Load Feature Extractor
    # ----------------------------------

    print("\nLoading ResNet50 feature extractor...")

    extractor = FeatureExtractor()

    print(f"Using device: {extractor.device}")

    # ----------------------------------
    # Load image list
    # ----------------------------------

    print("\nLoading image list...")

    df = pd.read_csv(VALID_IMAGES)

    if TEST_MODE:
        filenames = df["filename"].tolist()[:TEST_SIZE]
    else:
        filenames = df["filename"].tolist()

    total_images = len(filenames)

    print(f"Found {total_images:,} images.")

    # ----------------------------------
    # Generate embeddings
    # ----------------------------------

    print("\nGenerating embeddings...\n")

    successful_embeddings = []
    successful_filenames = []
    failed_images = []

    start_time = time.time()

    for i, filename in enumerate(filenames, start=1):

        image_path = IMAGE_FOLDER / filename

        try:

            embedding = extractor.extract(image_path)

            successful_embeddings.append(embedding)
            successful_filenames.append(filename)

        except Exception as e:

            print(f"Warning: Failed to process {filename}")
            print(f"Reason : {e}")

            failed_images.append(filename)
            continue

        if i % 500 == 0 or i == total_images:

            elapsed = time.time() - start_time

            percent = (i / total_images) * 100

            print(
                f"{i:,}/{total_images:,} "
                f"({percent:.1f}%) "
                f"Elapsed: {elapsed:.1f}s"
            )

    # ----------------------------------
    # Convert to NumPy array
    # ----------------------------------

    embeddings = np.array(
        successful_embeddings,
        dtype=np.float32
    )

    # ----------------------------------
    # Save outputs
    # ----------------------------------

    print("\nSaving embeddings...")

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    pd.DataFrame({

        "filename": successful_filenames

    }).to_csv(

        FILENAMES_FILE,

        index=False

    )

    elapsed = time.time() - start_time

    # ----------------------------------
    # Summary
    # ----------------------------------

    print("\n----------------------------------------")
    print("Embedding Generation Complete")
    print("----------------------------------------")

    print(f"Images requested : {total_images:,}")
    print(f"Images embedded  : {len(successful_filenames):,}")
    print(f"Failed images    : {len(failed_images):,}")
    print(f"Embedding shape  : {embeddings.shape}")
    print(f"Time taken       : {elapsed:.2f} seconds")

    if failed_images:

        print("\nFailed Images")

        for image in failed_images[:20]:
            print(f" - {image}")

        if len(failed_images) > 20:
            print(f"...and {len(failed_images) - 20} more.")

    print("\nSaved files:")

    print(f"Embeddings : {EMBEDDINGS_FILE}")
    print(f"Filenames  : {FILENAMES_FILE}")
    print(f"Embeddings saved : {len(successful_filenames):,}")


if __name__ == "__main__":
    main()