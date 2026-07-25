from pathlib import Path
import shutil
import uuid
import time
import os


from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from recommender.recommender import FashionRecommender

from schemas.recommendation import (
    HomeResponse,
    HealthResponse,
    Recommendation,
    ProcessingInfo,
    RecommendationResponse,
)


app = FastAPI(
    title="Fashion Recommendation API",
    description="Image-based fashion recommendation system using ResNet50 embeddings and cosine similarity.",
    version="1.0.0",
)

# --------------------------------------------------
# Enable CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# ----------------------------------
# Dataset Mode
# ----------------------------------

DATASET_MODE = os.getenv("DATASET_MODE", "full").lower()

if DATASET_MODE == "deployment":

    print("Running in DEPLOYMENT mode")

    IMAGE_FOLDER = (
        PROJECT_ROOT /
        "data" /
        "deployment" /
        "images"
    )

else:

    print("Running in FULL DATASET mode")

    IMAGE_FOLDER = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "images"
    )

UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# --------------------------------------------------
# Check embeddings
# --------------------------------------------------

# --------------------------------------------------
# Validate required model files
# --------------------------------------------------

EMBEDDINGS_FILE = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "image_embeddings.npy"
)

FILENAMES_FILE = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "image_filenames.csv"
)

if not EMBEDDINGS_FILE.exists():

    raise FileNotFoundError(

        f"\nMissing embedding file:\n"
        f"{EMBEDDINGS_FILE}\n\n"
        "Run 'py run_pipeline.py' before starting the API."

    )

if not FILENAMES_FILE.exists():

    raise FileNotFoundError(

        f"\nMissing filename index:\n"
        f"{FILENAMES_FILE}\n\n"
        "Run 'py run_pipeline.py' before starting the API."

    )

print("✓ Embedding files found.")

# --------------------------------------------------
# Serve dataset images
# --------------------------------------------------

app.mount(
    "/images",
    StaticFiles(directory=IMAGE_FOLDER),
    name="images",
)

# --------------------------------------------------
# Load recommender once
# --------------------------------------------------

print("Loading recommendation model...")

recommender = None
model_loaded = False
model_error = None

try:

    recommender = FashionRecommender()

    model_loaded = True

    print("Recommendation model loaded successfully.")

except Exception as e:

    model_error = str(e)

    print("\nWARNING: Recommendation model could not be loaded.")

    print(model_error)

print("API ready.\n")

# --------------------------------------------------
# Home Endpoint
# --------------------------------------------------

@app.get("/", response_model=HomeResponse)
def home():

    return HomeResponse(
        success=True,
        message="Fashion Recommendation API is running.",
        version=app.version,
        endpoints=[
            "/docs",
            "/health",
            "/recommend",
        ],
    )


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():

    dataset_images = 0

    if model_loaded and recommender is not None:

        dataset_images = len(recommender.filenames)

    return HealthResponse(

        status="healthy" if model_loaded else "degraded",

        model_loaded=model_loaded,

        dataset_images=dataset_images,

        model_error=model_error

    )


# --------------------------------------------------
# Recommendation Endpoint
# --------------------------------------------------

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(file: UploadFile = File(...)):

    start_time = time.perf_counter()

    if not model_loaded or recommender is None:

        raise HTTPException(

            status_code=503,

            detail="Recommendation model is unavailable. Check the /health endpoint."

        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    extension = Path(file.filename).suffix

    if extension == "":
        extension = ".jpg"

    uploaded_filename = f"{uuid.uuid4()}{extension}"

    uploaded_image = UPLOAD_FOLDER / uploaded_filename

    try:

        # Save uploaded image

        with uploaded_image.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate recommendations

        recommendations = recommender.recommend(uploaded_image)

        results = [
            Recommendation(
                rank=item["rank"],
                filename=item["filename"],
                similarity=item["similarity_percent"],
                image_url=f"/images/{item['filename']}"
            )
            for item in recommendations
        ]

        processing_time = round(time.perf_counter() - start_time, 3)

        return RecommendationResponse(

            success=True,

            message="Recommendations generated successfully.",

            query_image=file.filename,

            recommendation_count=len(results),

            processing=ProcessingInfo(

                feature_extractor="ResNet-50",

                    similarity_metric="Cosine Similarity",

                    dataset_size=len(recommender.filenames),

                    embedding_size=int(recommender.embeddings.shape[1]),

                    processing_time=processing_time,

                    recommendations_returned=len(results)

            ),

            recommendations=results

        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )

    finally:

        uploaded_image.unlink(missing_ok=True)