
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from src.features.feature_pipeline import build_feature_pipeline
from sklearn.cluster import MiniBatchKMeans
import os
from contextlib import asynccontextmanager

from prometheus_fastapi_instrumentator import Instrumentator
import structlog
import logging

# Configure structlog for structured logging
logging.basicConfig(format="%(message)s", stream=None, level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

MODEL_PATH = os.getenv("MODEL_PATH", "data/minibatch_kmeans_model.joblib")
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "data/feature_pipeline.joblib")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load pre-trained pipeline and model for inference
    pipeline = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    app.state.pipeline = pipeline
    app.state.model = model
    yield



app = FastAPI(title="Uber Customer Segmentation API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

# Define input schema
class CustomerFeatures(BaseModel):
    Booking_Value: float
    Ride_Distance: float
    Driver_Ratings: float
    Customer_Rating: float
    Vehicle_Type: str
    Payment_Method: str



@app.post("/predict")
def predict_cluster(features: CustomerFeatures, request: Request):
    # Convert input to DataFrame
    df = pd.DataFrame([{ 
        'Booking Value': features.Booking_Value,
        'Ride Distance': features.Ride_Distance,
        'Driver Ratings': features.Driver_Ratings,
        'Customer Rating': features.Customer_Rating,
        'Vehicle Type': features.Vehicle_Type,
        'Payment Method': features.Payment_Method
    }])
    pipeline = request.app.state.pipeline
    model = request.app.state.model
    try:
        X = pipeline.transform(df)
        cluster = int(model.predict(X)[0])
        logger.info("Prediction successful", input=features.dict(), cluster=cluster)
        return {"cluster": cluster}
    except Exception as e:
        logger.error("Prediction failed", error=str(e), input=features.dict())
        return {"error": "Prediction failed", "details": str(e)}, 500
