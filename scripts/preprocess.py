from pathlib import Path
from PIL import Image
import pandas as pd

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


def verify_images():
    """
    Checks every image in the dataset.
    Invalid or corrupted images are skipped.
    """

    valid_images = []
    corrupted = []

    image_files = sorted(RAW_IMAGES.glob("*.jpg"))

    print(f"Found {len(image_files)} images.\n")

    for i, image_path in enumerate(image_files, start=1):

        try:
            with Image.open(image_path) as img:
                img.verify()

            valid_images.append(image_path.name)

        except Exception:
            corrupted.append(image_path.name)

        if i % 1000 == 0:
            print(f"Checked {i:,} images...")

    df = pd.DataFrame(valid_images, columns=["filename"])
    df.to_csv(VALID_IMAGES_CSV, index=False)

    with open(REPORT_FILE, "w") as f:
        f.write("DATA CLEANING REPORT\n")
        f.write("=====================\n\n")
        f.write(f"Total Images: {len(image_files)}\n")
        f.write(f"Valid Images: {len(valid_images)}\n")
        f.write(f"Corrupted Images: {len(corrupted)}\n\n")

        if corrupted:
            f.write("Corrupted Files:\n")
            for img in corrupted:
                f.write(img + "\n")

    print("\nFinished.\n")
    print(f"Valid images : {len(valid_images)}")
    print(f"Corrupted    : {len(corrupted)}")
    print(f"\nCSV saved to:\n{VALID_IMAGES_CSV}")
    print(f"\nReport saved to:\n{REPORT_FILE}")


if __name__ == "__main__":
    verify_images()