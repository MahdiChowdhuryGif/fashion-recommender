from pathlib import Path
import hashlib
from collections import defaultdict

RAW_IMAGES = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "images"
)

def sha256_hash(file_path, chunk_size=8192):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def main():

    image_files = sorted(RAW_IMAGES.glob("*.jpg"))

    print(f"\nScanning {len(image_files):,} images...\n")

    hashes = defaultdict(list)

    for i, image in enumerate(image_files, start=1):

        hashes[sha256_hash(image)].append(image.name)

        if i % 1000 == 0:
            print(f"Checked {i:,} images...")

    duplicate_groups = [
        files
        for files in hashes.values()
        if len(files) > 1
    ]

    total_duplicates = sum(len(group) - 1 for group in duplicate_groups)

    print("\n==============================")
    print("Duplicate Analysis Complete")
    print("==============================")
    print(f"Total Images        : {len(image_files):,}")
    print(f"Duplicate Groups    : {len(duplicate_groups):,}")
    print(f"Duplicate Images    : {total_duplicates:,}")

    if duplicate_groups:
        print("\nExample duplicate groups:\n")

        for group in duplicate_groups[:10]:
            print(group)
            print()


if __name__ == "__main__":
    main()