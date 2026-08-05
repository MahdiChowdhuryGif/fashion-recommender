from pathlib import Path
from PIL import Image
import pandas as pd
import hashlib

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_IMAGES = PROJECT_ROOT / "data" / "raw" / "images"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VALID_IMAGES_CSV = PROCESSED_DIR / "valid_images.csv"
REPORT_FILE = REPORT_DIR / "cleaning_report.txt"


def calculate_sha256(file_path, chunk_size=8192):
    """
    Calculates the SHA-256 hash of a file.
    Used to detect exact duplicate images.
    """
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def verify_images():
    """
    Cleans the raw dataset by:

    1. Removing corrupted images.
    2. Removing exact duplicate images.
    3. Generating valid_images.csv.
    4. Creating a cleaning report.
    """

    valid_images = []
    corrupted_images = []
    duplicate_images = []

    seen_hashes = {}

    image_files = sorted(RAW_IMAGES.glob("*.jpg"))

    print(f"\nFound {len(image_files):,} images.\n")

    for i, image_path in enumerate(image_files, start=1):

        try:
            # Verify image integrity
            with Image.open(image_path) as img:
                img.verify()

            # Calculate SHA-256 hash
            image_hash = calculate_sha256(image_path)

            if image_hash in seen_hashes:
                duplicate_images.append(image_path.name)
            else:
                seen_hashes[image_hash] = image_path.name
                valid_images.append(image_path.name)

        except Exception:
            corrupted_images.append(image_path.name)

        if i % 1000 == 0:
            print(f"Checked {i:,} images...")

    # Save cleaned image list
    df = pd.DataFrame(valid_images, columns=["filename"])
    df.to_csv(VALID_IMAGES_CSV, index=False)

    # Generate cleaning report
    with open(REPORT_FILE, "w") as f:

        f.write("DATA CLEANING REPORT\n")
        f.write("=====================\n\n")

        f.write(f"Total Images      : {len(image_files)}\n")
        f.write(f"Valid Images      : {len(valid_images)}\n")
        f.write(f"Corrupted Images  : {len(corrupted_images)}\n")
        f.write(f"Duplicate Images  : {len(duplicate_images)}\n\n")

        if corrupted_images:
            f.write("Corrupted Files\n")
            f.write("------------------------------\n")
            for image in corrupted_images:
                f.write(image + "\n")
            f.write("\n")

        if duplicate_images:
            f.write("Duplicate Files\n")
            f.write("------------------------------\n")
            for image in duplicate_images:
                f.write(image + "\n")

    print("\n================================")
    print("Data Cleaning Complete")
    print("================================")
    print(f"Total Images      : {len(image_files):,}")
    print(f"Valid Images      : {len(valid_images):,}")
    print(f"Corrupted Images  : {len(corrupted_images):,}")
    print(f"Duplicate Images  : {len(duplicate_images):,}")

    print(f"\nClean dataset saved to:")
    print(VALID_IMAGES_CSV)

    print(f"\nCleaning report saved to:")
    print(REPORT_FILE)


if __name__ == "__main__":
    verify_images()