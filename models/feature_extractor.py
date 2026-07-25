from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms


class FeatureExtractor:
    """
    Generates 2048-dimensional image embeddings using a pretrained ResNet50.
    """

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        weights = models.ResNet50_Weights.DEFAULT

        model = models.resnet50(weights=weights)

        # Remove the classification layer
        self.model = torch.nn.Sequential(*list(model.children())[:-1])

        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():

            embedding = self.model(image)

        return embedding.squeeze().cpu().numpy().astype(np.float16)