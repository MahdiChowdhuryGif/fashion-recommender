from pathlib import Path
import time

import numpy as np
import pandas as pd

from models.feature_extractor import FeatureExtractor


def main():

    # ----------------------------------
    # Paths
    # ----------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    VALID_IMAGES = PROJECT_ROOT / "data" / "processed" / "valid_images.csv"
    IMAGE_FOLDER = PROJECT_ROOT / "data" / "raw" / "images"

    OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    EMBEDDINGS_FILE = OUTPUT_DIR / "image_embeddings.npy"
    FILENAMES_FILE = OUTPUT_DIR / "image_filenames.csv"

    # ----------------------------------
    # Settings
    # ----------------------------------

    TEST_MODE = False      # True = process a small subset
    TEST_SIZE = 100        # Number of images in test mode

    # ----------------------------------
    # Load Feature Extractor
    # ----------------------------------

    extractor = FeatureExtractor()

    print(f"\nUsing device: {extractor.device}")

    # ----------------------------------
    # Load image list
    # ----------------------------------

    df = pd.read_csv(VALID_IMAGES)

    if TEST_MODE:
        filenames = df["filename"].tolist()[:TEST_SIZE]
    else:
        filenames = df["filename"].tolist()

    num_images = len(filenames)

    print(f"\nProcessing {num_images:,} images...\n")

    # ----------------------------------
    # Create embedding array
    # ----------------------------------

    embeddings = np.zeros((num_images, 2048), dtype=np.float32)

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
    # Save results
    # ----------------------------------

    print("\nSaving embeddings...")

    np.save(EMBEDDINGS_FILE, embeddings)

    pd.DataFrame({
        "filename": filenames
    }).to_csv(
        FILENAMES_FILE,
        index=False
    )

    elapsed = time.time() - start_time

    print("\nFinished!")

    print(f"Embedding shape: {embeddings.shape}")

    print(f"Total time: {elapsed:.2f} seconds")

    print(f"\nEmbeddings saved to:\n{EMBEDDINGS_FILE}")

    print(f"\nFilenames saved to:\n{FILENAMES_FILE}")


if __name__ == "__main__":
    main()