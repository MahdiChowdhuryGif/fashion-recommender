👗 Fashion Recommendation AI

An AI-powered fashion recommendation system that finds visually similar clothing items using deep learning image embeddings.

The application uses a pretrained ResNet-50 convolutional neural network to extract image features and recommends visually similar fashion items using cosine similarity. A FastAPI backend provides the recommendation API, while a responsive HTML, CSS, and JavaScript frontend allows users to upload an image, preview it, and view ranked recommendations.

Live Demo

Frontend

https://fashion-recommender-frontend.onrender.com/

Backend API

https://fashion-recommender-1r1l.onrender.com

API Documentation

https://fashion-recommender-1r1l.onrender.com/docs

Features

Upload a fashion image

Drag-and-drop image upload

Automatic image preview

Display uploaded image information including filename, file size, and resolution

Deep learning feature extraction using pretrained ResNet-50

2048-dimensional image embeddings

Cosine similarity recommendation engine

Top 5 visually similar fashion items

Ranked recommendation cards

Similarity percentages and visual progress bars

Full-size image viewer

AI Processing Summary

Loading messages showing the recommendation stages

FastAPI REST API

Responsive frontend

How the System Works

When a user uploads an image:

The image is received by the FastAPI backend.

ResNet-50 is used as a feature extractor.

The classification layer is removed so the network produces a 2048-dimensional feature embedding.

The uploaded image embedding is compared with stored dataset embeddings.

Cosine similarity measures visual similarity.

Images are ranked from most similar to least similar.

The five highest-scoring images are returned.

The frontend displays the recommendations, similarity scores, filenames, and processing information.

Machine Learning Pipeline

Process the raw fashion image dataset

Validate image files

Detect duplicate images

Remove invalid or corrupted images

Save the cleaned image list

Generate ResNet-50 feature embeddings

Store embeddings as a NumPy array

Store corresponding image filenames

Evaluate the recommendation system

Create a smaller deployment dataset when required

Run the FastAPI recommendation service

Dataset

The project uses the Vibrent Clothes Rental dataset.

Dataset source:

https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental

The original dataset is not included in the repository because of its size.

Place the raw images inside:

data/raw/images/

Dataset Statistics

Statistic

Value

Images checked

50,293

Duplicate groups

8,061

Duplicate images identified

18,915

Valid images after cleaning

31,378

Embedding size

2,048 dimensions

Feature extractor

ResNet-50

Similarity metric

Cosine Similarity

Recommendations returned

5

Dataset Modes

The project supports two operating modes using the DATASET_MODE environment variable.

Full Dataset

Used for local development and evaluation.

DATASET_MODE=full

The cleaned dataset contains 31,378 valid images.

Run the pipeline with:

py run_pipeline.py

Deployment Dataset

Used for cloud deployment to reduce storage, startup time, and memory requirements.

DATASET_MODE=deployment

The current deployment subset contains 10,000 images selected from the valid dataset using a fixed random seed.

Create the deployment subset with:

py scripts/create_deployment_subset.py

For Render, use:

DATASET_MODE=deployment

Technologies

Machine Learning

Python

PyTorch

Torchvision

NumPy

Scikit-learn

ResNet-50

Cosine Similarity

Backend

FastAPI

Uvicorn

Pydantic

Frontend

HTML5

CSS3

JavaScript

Fetch API

CSS Grid and Flexbox

Deployment

Render

GitHub

Git LFS

Project Structure

fashion-recommender/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── deployment/
│   │   ├── images/
│   │   └── valid_images.csv
│   ├── raw/
│   │   └── outfits.csv
│   └── processed/
│       ├── image_embeddings.npy
│       ├── image_filenames.csv
│       └── valid_images.csv
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── models/
│   └── feature_extractor.py
│
├── recommender/
│   └── recommender.py
│
├── schemas/
│   └── recommendation.py
│
├── scripts/
│   ├── create_embeddings.py
│   ├── create_deployment_subset.py
│   └── evaluate.py
│
└── uploads/

Installation

Clone the repository:

git clone https://github.com/MahdiChowdhuryGif/fashion-recommender.git

Navigate to the project:

cd fashion-recommender

Create a virtual environment:

py -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install the requirements:

pip install -r requirements.txt

Running the Pipeline

Run the complete local pipeline:

py run_pipeline.py

The pipeline can:

Validate the dataset

Detect duplicate images

Generate ResNet-50 embeddings

Save image embeddings

Save corresponding filenames

Evaluate recommendation performance

Produce evaluation reports

Creating the Deployment Subset

Create the current 10,000-image deployment subset with:

py scripts/create_deployment_subset.py

A fixed random seed is used so that the subset can be reproduced.

Running the Backend

Start the FastAPI server:

uvicorn app:app --reload

Open the local API:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

Running the Frontend

The frontend is contained inside frontend.

From the project root:

py -m http.server 5500 --directory frontend

Then open:

http://127.0.0.1:5500

The frontend communicates with the FastAPI backend through the /recommend endpoint.

The backend URL used by the frontend is configured in:

frontend/js/app.js

Frontend Functionality

The frontend workflow is:

Upload an image using Choose Image, clicking the upload area, or drag and drop.

Preview the image before submitting it.

Display image information including filename, file size, and resolution.

Generate recommendations by sending the image to POST /recommend.

Display five ranked recommendations with similarity percentages, progress bars, filenames, and full-image viewing.

Display the AI Processing Summary including dataset size, model, embedding size, similarity metric, processing time, and recommendation count.

API Endpoint

POST /recommend

Uploads a fashion image and returns the five most visually similar items.

The request uses multipart/form-data with the uploaded image in the file field.

Example response:

{
  "success": true,
  "recommendation_count": 5,
  "processing": {
    "feature_extractor": "ResNet-50",
    "similarity_metric": "Cosine Similarity",
    "dataset_size": 10000,
    "embedding_size": 2048,
    "processing_time": 1.23,
    "recommendations_returned": 5
  },
  "recommendations": [
    {
      "rank": 1,
      "filename": "example.jpg",
      "similarity": 83.09,
      "image_url": "/images/example.jpg"
    }
  ]
}

The processing time varies depending on the hardware and deployment environment.

Evaluation

Run the evaluation script with:

py scripts/evaluate.py

The evaluation uses the generated embeddings and recommendation system to assess similarity results.

Deployment

The application is deployed using Render as two services:

FastAPI backend web service

Static frontend site

The deployed backend uses:

DATASET_MODE=deployment

Before deployment:

Test the frontend locally.

Test image upload.

Test recommendations.

Confirm recommendation images load.

Confirm the AI Processing Summary appears.

Commit changes to Git.

Push changes to GitHub.

Allow Render to rebuild.

Test the live frontend and backend again.

Git LFS

The generated embedding file is large and is stored using Git LFS:

data/processed/image_embeddings.npy

Install and initialise Git LFS when setting up the project on a new machine.

Future Improvements

Potential future enhancements include:

Fine-tuning the neural network on fashion-specific data

Using FAISS for faster similarity search

Adding product metadata filtering

Supporting multiple recommendation models

User accounts and recommendation history

Cloud storage for uploaded images

Batch image recommendations

Adding additional fashion-specific visual features

Author

Developed as an Artificial Intelligence and Machine Learning project demonstrating computer vision, deep learning, image feature extraction, image similarity search, recommendation systems, API development, frontend development, cloud deployment, and end-to-end machine learning pipeline development.