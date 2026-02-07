from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv

# Load env
load_dotenv(override=True)

# Force Offline Mode (Network Instability Fix)
import os
os.environ["OFFLINE_MODE"] = "True"


# Import Dependencies and Routers
from src.api.routes import (
    system, auth, disease, pest, treatment, assistant, feedback, weather
)
from src.api.dependencies import (
    get_db_manager, get_disease_classifier, get_weather_predictor,
    get_ai_recommender, get_spray_scheduler, get_forecasting_model
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    Initializes models and services.
    """
    logger.info("Startup: Initializing Application Services...")
    
    # Initialize DB
    get_db_manager()
    
    # Trigger lazy loading of heavy models if desired
    # or let them load on first request (as they currently do via dependencies)
    # If we want to pre-load, we can call them here:
    # get_disease_classifier()
    # get_weather_predictor()
    # get_ai_recommender()
    
    yield
    
    logger.info("Shutdown: Cleaning up...")

# Initialize FastAPI
app = FastAPI(
    title="Custard Apple Pest Management - AI System (PyTorch)",
    description="AI-powered pest prediction, disease detection, and automated spray scheduling with GPU support",
    version="2.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(system.router, tags=["System"])
app.include_router(auth.router, tags=["Authentication"])
app.include_router(disease.router, tags=["Disease Detection"])
app.include_router(pest.router, tags=["Pest Prediction"])
app.include_router(treatment.router, tags=["Treatment & Scheduling"])
app.include_router(weather.router, tags=["Weather & Environment"])
app.include_router(assistant.router, tags=["AI Assistant"])
app.include_router(feedback.router, tags=["Feedback Loop"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
