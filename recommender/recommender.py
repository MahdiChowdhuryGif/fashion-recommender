from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from models.feature_extractor import FeatureExtractor


class FashionRecommender:
    """
    Fashion image recommender based on cosine similarity
    between ResNet50 feature embeddings.
    """

    def __init__(self) -> None:

        self.project_root = Path(__file__).resolve().parent.parent

        self.image_folder = (
            self.project_root /
            "data" /
            "raw" /
            "images"
        )

        embeddings_file = (
            self.project_root /
            "data" /
            "processed" /
            "image_embeddings.npy"
        )

        filenames_file = (
            self.project_root /
            "data" /
            "processed" /
            "image_filenames.csv"
        )

        print("Loading embeddings...")

        self.embeddings = np.load(embeddings_file)

        self.filenames = (
            pd.read_csv(filenames_file)["filename"]
            .tolist()
        )

        self.extractor = FeatureExtractor()

        print(
            f"Loaded {len(self.filenames):,} image embeddings."
        )

    def recommend(
        self,
        image_path: Union[str, Path],
        top_n: int = 5,
        exclude_query: bool = True,
        duplicate_threshold: float = 0.9999,
    ) -> List[Dict]:
        """
        Recommend visually similar fashion items.

        Parameters
        ----------
        image_path
            Query image.

        top_n
            Number of recommendations.

        exclude_query
            Exclude the query image if it exists
            in the dataset.

        duplicate_threshold
            Skip recommendations that are almost
            identical to previously selected ones.

        Returns
        -------
        List[Dict]
        """

        image_path = Path(image_path)

        query_embedding = self.extractor.extract(image_path)

        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]

        sorted_indices = np.argsort(similarities)[::-1]

        recommendations = []
        selected_embeddings = []

        query_filename = image_path.name

        for index in sorted_indices:

            filename = self.filenames[index]

            similarity = float(similarities[index])

            # Skip the uploaded image

            if exclude_query and filename == query_filename:
                continue

            candidate_embedding = self.embeddings[index]

            # Check for duplicate recommendations

            duplicate = False

            for embedding in selected_embeddings:

                score = cosine_similarity(
                    candidate_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]

                if score >= duplicate_threshold:
                    duplicate = True
                    break

            if duplicate:
                continue

            selected_embeddings.append(candidate_embedding)

            recommendations.append({

                "rank": len(recommendations) + 1,

                "filename": filename,

                "image_path":
                    self.image_folder / filename,

                "similarity": similarity,

                "similarity_percent":
                    round(similarity * 100, 2)

            })

            if len(recommendations) >= top_n:
                break

        return recommendations


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

    recommendations = recommender.recommend(query_image)

    print("\nTop Recommendations\n")

    for item in recommendations:

        print(
            f"{item['rank']}. "
            f"{item['filename']} "
            f"({item['similarity_percent']}%)"
        )


if __name__ == "__main__":
    main()