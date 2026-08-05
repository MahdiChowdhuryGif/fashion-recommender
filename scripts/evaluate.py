from pathlib import Path
import random

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from recommender.recommender import FashionRecommender


def create_report(query_image, recommendations, output_file, evaluation_number):

    fig = plt.figure(figsize=(18, 8))

    fig.suptitle(
        f"Fashion Recommendation Evaluation {evaluation_number}",
        fontsize=18,
        fontweight="bold"
    )

    # ----------------------------------
    # Query Image
    # ----------------------------------

    ax = plt.subplot(2, 3, 1)

    ax.imshow(Image.open(query_image))

    ax.set_title(
        "Query Image",
        fontsize=13,
        fontweight="bold"
    )

    ax.axis("off")

    # ----------------------------------
    # Recommendations
    # ----------------------------------

    for i, item in enumerate(recommendations):

        ax = plt.subplot(2, 3, i + 2)

        ax.imshow(Image.open(item["image_path"]))

        ax.set_title(
            f"Rank {item['rank']}\n"
            f"Cosine: {item['cosine_similarity']:.4f}\n"
            f"{item['similarity_percent']}%",
            fontsize=10
        )

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    project_root = Path(__file__).resolve().parent.parent

    report_folder = (
        project_root /
        "data" /
        "reports"
    )

    report_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    valid_images = pd.read_csv(

        project_root /
        "data" /
        "processed" /
        "valid_images.csv"

    )["filename"].tolist()

    image_folder = (
        project_root /
        "data" /
        "raw" /
        "images"
    )

    recommender = FashionRecommender()

    sample_size = min(3, len(valid_images))

    random_images = random.sample(
        valid_images,
        sample_size
    )

    print("\n==============================================")
    print("Automatic Model Evaluation")
    print("==============================================")

    for evaluation_number, filename in enumerate(
        random_images,
        start=1
    ):

        query_image = image_folder / filename

        recommendations = recommender.recommend(
            query_image,
            top_n=5
        )

        output_file = (
            report_folder /
            f"recommendation_{evaluation_number}.png"
        )

        create_report(
            query_image,
            recommendations,
            output_file,
            evaluation_number
        )

        print(f"\nEvaluation {evaluation_number}")
        print("-" * 40)

        print(f"Query Image: {filename}")

        for item in recommendations:

            print(
                f"{item['rank']}. "
                f"{item['filename']} | "
                f"Cosine Similarity: "
                f"{item['cosine_similarity']:.4f} | "
                f"{item['similarity_percent']}%"
            )

        print(f"Saved report: {output_file}")

    print("\n==============================================")
    print("Evaluation Complete")
    print("==============================================")
    print(f"Reports saved to: {report_folder}")


if __name__ == "__main__":
    main()