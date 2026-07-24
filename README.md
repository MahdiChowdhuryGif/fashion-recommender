# 👗 Fashion Recommendation AI

An AI-powered fashion recommendation system that finds visually similar clothing items using deep learning image embeddings.

The application uses a pretrained ResNet-50 convolutional neural network to extract image features and recommends similar fashion items using cosine similarity. A FastAPI backend provides the recommendation API, while a responsive HTML, CSS, and JavaScript frontend allows users to upload an image and view recommendations.

---

## Features

- Upload any fashion image
- Drag-and-drop image upload
- Automatic image preview
- Deep learning feature extraction using ResNet-50
- Cosine similarity recommendation engine
- Top 5 visually similar fashion items
- AI Processing Summary
- Interactive recommendation cards
- Full-size image viewer
- FastAPI REST API
- Responsive frontend

---

## Machine Learning Pipeline

The recommendation system follows these stages:

1. Process the raw fashion image dataset
2. Clean and validate image files
3. Generate feature embeddings using ResNet-50
4. Store image embeddings
5. Compare embeddings using cosine similarity
6. Rank the most similar images
7. Return the Top 5 recommendations

---

## Dataset

- Fashion image dataset
- Total processed images: **50,293**
- Feature vector size: **2048 dimensions**
- Feature extractor: **ResNet-50 (ImageNet pretrained)**
- This project uses the DeepFashion dataset. Due to GitHub size limitations, the dataset is not included.
    Download the dataset from:

    https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-

    Place the images here:

    data/raw/images/

---

## Technologies

### Machine Learning

- Python
- PyTorch
- Torchvision
- NumPy
- Scikit-learn

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Frontend

- HTML5
- CSS3
- JavaScript

---

## Project Structure

```text
fashion-recommender/

├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
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
│   └── evaluate.py
│
└── uploads/
```

---


## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fashion-recommender.git
```

Navigate to the project

```bash
cd fashion-recommender
```

Create a virtual environment

```bash
py -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the API

Start the FastAPI server

```bash
uvicorn app:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

to access the interactive Swagger documentation.

---

## Running the Frontend

Navigate to the frontend folder

```bash
cd frontend
```

Start a local web server

```bash
py -m http.server 5500
```

Open

```
http://localhost:5500
```

---

## API Endpoint

### POST /recommend

Uploads a fashion image and returns the five most visually similar items.

Example response

```json
{
  "success": true,
  "recommendation_count": 5,
  "processing": {
    "feature_extractor": "ResNet-50",
    "similarity_metric": "Cosine Similarity",
    "dataset_size": 50293,
    "embedding_size": 2048
  },
  "recommendations": [
    {
      "rank": 1,
      "filename": "example.jpg",
      "similarity": 69.77
    }
  ]
}
```

---


## Future Improvements

Potential future enhancements include:

- Fine-tuning the neural network on fashion-specific data
- Using FAISS for faster similarity search
- Adding product metadata filtering
- Supporting multiple recommendation models
- User accounts and recommendation history
- Cloud storage for uploaded images
- Batch image recommendations

---

## Author

Developed as an Artificial Intelligence and Machine Learning project demonstrating computer vision, deep learning, API development, and full-stack web application development.