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

    num_images = len(filenames)

    print(f"Found {num_images:,} images.")

    # ----------------------------------
    # Create embedding array
    # ----------------------------------

    print("\nGenerating embeddings...\n")

    embeddings = np.zeros(
        (num_images, 2048),
        dtype=np.float32
    )

    start_time = time.time()

    # ----------------------------------
    # Generate embeddings
    # ----------------------------------

    for i, filename in enumerate(filenames):

        image_path = IMAGE_FOLDER / filename

        embeddings[i] = extractor.extract(image_path)

        if (i + 1) % 500 == 0 or (i + 1) == num_images:

            elapsed = time.time() - start_time

            percent = ((i + 1) / num_images) * 100

            print(
                f"{i + 1:,}/{num_images:,} "
                f"({percent:.1f}%) "
                f"Elapsed: {elapsed:.1f}s"
            )

    # ----------------------------------
    # Save embeddings
    # ----------------------------------

    print("\nSaving embeddings...")

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    pd.DataFrame({

        "filename": filenames

    }).to_csv(

        FILENAMES_FILE,

        index=False

    )

    elapsed = time.time() - start_time

    print("\n----------------------------------------")
    print("Embedding Generation Complete")
    print("----------------------------------------")

    print(f"Images processed : {num_images:,}")
    print(f"Embedding shape  : {embeddings.shape}")
    print(f"Time taken       : {elapsed:.2f} seconds")

    print("\nSaved files:")

    print(f"Embeddings : {EMBEDDINGS_FILE}")
    print(f"Filenames  : {FILENAMES_FILE}")


if __name__ == "__main__":
    main()