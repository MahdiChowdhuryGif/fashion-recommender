from pathlib import Path
import argparse
import time

from scripts.create_embeddings import main as create_embeddings
from scripts.evaluate import main as evaluate


def print_header():
    print("=" * 60)
    print("        Fashion Recommendation AI Pipeline")
    print("=" * 60)


def check_project():

    print("\n[1/4] Checking project structure...\n")

    project_root = Path(__file__).resolve().parent

    valid_images = (
        project_root /
        "data" /
        "processed" /
        "valid_images.csv"
    )

    image_folder = (
        project_root /
        "data" /
        "raw" /
        "images"
    )

    if not valid_images.exists():
        raise FileNotFoundError(
            f"Missing file:\n{valid_images}"
        )

    if not image_folder.exists():
        raise FileNotFoundError(
            f"Missing folder:\n{image_folder}"
        )

    print("✓ Dataset found")
    print("✓ Project structure verified")


def embeddings_exist():

    project_root = Path(__file__).resolve().parent

    embeddings = (
        project_root /
        "data" /
        "processed" /
        "image_embeddings.npy"
    )

    filenames = (
        project_root /
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

    print("\n[2/4] Image Embedding Generation\n")

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

    project_root = Path(__file__).resolve().parent

    embedding_file = (
        project_root /
        "data" /
        "processed" /
        "image_embeddings.npy"
    )

    report_file = (
        project_root /
        "data" /
        "reports" /
        "recommendation_results.png"
    )

    print("\n[4/4] Pipeline Summary\n")

    print(f"Dataset Images        : 50,293")
    print(f"Embedding Size        : 2048")
    print(f"Embeddings File       : {embedding_file}")
    print(f"Evaluation Report     : {report_file}")
    print(f"Total Execution Time  : {elapsed:.2f} seconds")

    print("\n✓ Pipeline completed successfully!")

    print("=" * 60)


if __name__ == "__main__":
    main()