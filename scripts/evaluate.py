from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from recommender.recommender import FashionRecommender


def main():

    project_root = Path(__file__).resolve().parent.parent

    report_folder = (
        project_root /
        "data" /
        "reports"
    )

    report_folder.mkdir(exist_ok=True)

    query_image = (
        project_root /
        "data" /
        "raw" /
        "images" /
        "0000cdba64314d84a49ed1c266589cc0.jpg"
    )

    recommender = FashionRecommender()

    recommendations = recommender.recommend(query_image)

    # ---------------------------------------------------
    # Create figure
    # ---------------------------------------------------

    fig = plt.figure(figsize=(18, 8))

    fig.suptitle(
        "Fashion Recommendation Results",
        fontsize=18,
        fontweight="bold"
    )

    # ---------------------------------------------------
    # Query Image
    # ---------------------------------------------------

    ax = plt.subplot(2, 3, 1)

    ax.imshow(Image.open(query_image))

    ax.set_title(
        "Query Image",
        fontsize=13,
        fontweight="bold"
    )

    ax.axis("off")

    # ---------------------------------------------------
    # Recommendations
    # ---------------------------------------------------

    for i, item in enumerate(recommendations):

        ax = plt.subplot(2, 3, i + 2)

        image = Image.open(item["image_path"])

        ax.imshow(image)

        ax.set_title(
            f"Recommendation {item['rank']}\n"
            f"{item['similarity_percent']}%",
            fontsize=11
        )

        ax.axis("off")

    plt.tight_layout()

    output_file = (
        report_folder /
        "recommendation_results.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nEvaluation image saved to:\n")

    print(output_file)


if __name__ == "__main__":
    main()