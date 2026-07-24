from pydantic import BaseModel


# --------------------------------------------------
# Individual Recommendation
# --------------------------------------------------

class Recommendation(BaseModel):
    rank: int
    filename: str
    similarity: float
    image_url: str


# --------------------------------------------------
# AI Processing Information
# --------------------------------------------------

class ProcessingInfo(BaseModel):
    feature_extractor: str
    similarity_metric: str
    dataset_size: int
    embedding_size: int
    processing_time: float
    recommendations_returned: int


# --------------------------------------------------
# Recommendation API Response
# --------------------------------------------------

class RecommendationResponse(BaseModel):
    success: bool
    message: str
    query_image: str
    recommendation_count: int
    processing: ProcessingInfo
    recommendations: list[Recommendation]


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dataset_images: int


# --------------------------------------------------
# Home Endpoint
# --------------------------------------------------

class HomeResponse(BaseModel):
    success: bool
    message: str
    version: str
    endpoints: list[str]