from pathlib import Path
import argparse
import os
import time
import pandas as pd

from scripts.preprocess import verify_images
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

    print("\n[1/5] Checking project structure...\n")

    if not IMAGE_FOLDER.exists():

        raise FileNotFoundError(
            f"Missing folder:\n{IMAGE_FOLDER}"
        )

    image_count = len(list(IMAGE_FOLDER.glob("*.jpg")))

    if image_count == 0:

        raise RuntimeError(
            "No images were found in the dataset folder."
        )

    print("✓ Dataset found")
    print(f"✓ {image_count:,} images detected")


def embeddings_need_regeneration():

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

    if not embeddings.exists() or not filenames.exists():
        return True

    try:
        embedding_files = pd.read_csv(filenames)
        valid_files = pd.read_csv(VALID_IMAGES)

        return len(embedding_files) != len(valid_files)

    except Exception:
        return True


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

    print("\n[2/5] Data Cleaning\n")

    verify_images()

    print("\n✓ Data cleaning completed successfully.")

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    print(
        f"\n[3/5] Image Embedding Generation "
        f"({DATASET_MODE.title()} Mode)\n"
    )

    if not args.force and not embeddings_need_regeneration():

        print("✓ Existing embeddings match the cleaned dataset.")
        print("Skipping embedding generation.")

    else:

        print("Generating image embeddings...\n")

        create_embeddings()

        print("\n✓ Embeddings generated successfully.")

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    print("\n[4/5] Model Evaluation\n")

    evaluate()

    print("\n✓ Evaluation completed successfully.")

    # --------------------------------------------------
    # Step 5
    # --------------------------------------------------

    elapsed = time.time() - start_time

    embedding_file = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "image_embeddings.npy"
    )

    report_folder = (
    PROJECT_ROOT /
    "data" /
    "reports"
    )

    report_files = sorted(
    report_folder.glob("recommendation_*.png")
    )

    dataset_size = len(pd.read_csv(VALID_IMAGES))

    print("\n[5/5] Pipeline Summary\n")

    print(f"Dataset Mode          : {DATASET_MODE.title()}")
    print(f"Dataset Images        : {dataset_size:,}")
    print("Embedding Size        : 2048")
    print(f"Embeddings File       : {embedding_file}")
    print("Evaluation Reports    :")

    if report_files:
        for report in report_files:
         print(f"  {report}")
    else:
        print("  No evaluation reports found.")
    print(f"Total Execution Time  : {elapsed:.2f} seconds")

    print("\n✓ Pipeline completed successfully!")

    print("=" * 60)


if __name__ == "__main__":
    main()