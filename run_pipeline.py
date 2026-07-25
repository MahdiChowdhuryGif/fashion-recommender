from pathlib import Path
import argparse
import os
import time
import pandas as pd

from scripts.create_embeddings import main as create_embeddings
from scripts.evaluate import main as evaluate


# --------------------------------------------------
# Dataset Mode
# --------------------------------------------------

DATASET_MODE = os.getenv("DATASET_MODE", "full").lower()

PROJECT_ROOT = Path(__file__).resolve().parent

if DATASET_MODE == "deployment":

    IMAGE_FOLDER = (
        PROJECT_ROOT /
        "data" /
        "deployment" /
        "images"
    )

    VALID_IMAGES = (
        PROJECT_ROOT /
        "data" /
        "deployment" /
        "valid_images.csv"
    )

else:

    IMAGE_FOLDER = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "images"
    )

    VALID_IMAGES = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "valid_images.csv"
    )


def print_header():

    print("=" * 60)
    print("        Fashion Recommendation AI Pipeline")
    print("=" * 60)

    print(f"\nRunning in {DATASET_MODE.upper()} mode")


def check_project():

    print("\n[1/4] Checking project structure...\n")

    if not VALID_IMAGES.exists():

        raise FileNotFoundError(
            f"Missing file:\n{VALID_IMAGES}"
        )

    if not IMAGE_FOLDER.exists():

        raise FileNotFoundError(
            f"Missing folder:\n{IMAGE_FOLDER}"
        )

    print("✓ Dataset found")
    print("✓ Project structure verified")


def embeddings_exist():

    embeddings = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "image_embeddings.npy"
    )

    filenames = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "image_filenames.csv"
    )

    return embeddings.exists() and filenames.exists()


def main():

    parser = argparse.ArgumentParser(
        description="Fashion Recommendation AI Pipeline"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate image embeddings even if they already exist."
    )

    args = parser.parse_args()

    start_time = time.time()

    print_header()

    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    check_project()

    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    print(
    f"\n[2/4] Image Embedding Generation "
    f"({DATASET_MODE.title()} Mode)\n"
    )

    if embeddings_exist() and not args.force:

        print("✓ Existing embeddings found")
        print("Skipping embedding generation.")

    else:

        print("Generating image embeddings...\n")

        create_embeddings()

        print("\n✓ Embeddings generated successfully.")

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    print("\n[3/4] Model Evaluation\n")

    evaluate()

    print("\n✓ Evaluation completed successfully.")

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    elapsed = time.time() - start_time

    embedding_file = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "image_embeddings.npy"
    )

    report_file = (
        PROJECT_ROOT /
        "data" /
        "reports" /
        "recommendation_results.png"
    )

    dataset_size = len(pd.read_csv(VALID_IMAGES))

    print("\n[4/4] Pipeline Summary\n")

    print(f"Dataset Mode          : {DATASET_MODE.title()}")
    print(f"Dataset Images        : {dataset_size:,}")
    print("Embedding Size        : 2048")
    print(f"Embeddings File       : {embedding_file}")
    print(f"Evaluation Report     : {report_file}")
    print(f"Total Execution Time  : {elapsed:.2f} seconds")

    print("\n✓ Pipeline completed successfully!")

    print("=" * 60)


if __name__ == "__main__":
    main()