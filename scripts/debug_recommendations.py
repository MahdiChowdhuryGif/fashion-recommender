from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from recommender.recommender import FashionRecommender


def main():

    project_root = Path(__file__).resolve().parent.parent

    query_image = (
        project_root /
        "data" /
        "raw" /
        "images" /
        "0000cdba64314d84a49ed1c266589cc0.jpg"
    )

    recommender = FashionRecommender()

    results = recommender.recommend(query_image)

    # Create figure
    fig, axes = plt.subplots(1, 6, figsize=(18, 5))

    # -------------------------
    # Query image
    # -------------------------

    query = Image.open(query_image)

    axes[0].imshow(query)
    axes[0].set_title("Query")
    axes[0].axis("off")

    # -------------------------
    # Recommendations
    # -------------------------

    for i, result in enumerate(results):

        image = Image.open(result["image_path"])

        axes[i + 1].imshow(image)

        axes[i + 1].set_title(
            f"{result['similarity']:.4f}"
        )

        axes[i + 1].axis("off")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()