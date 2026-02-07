from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
import json
from pathlib import Path
from src.api.dependencies import (
    get_disease_classifier, get_weather_predictor, get_ai_recommender, get_spray_scheduler
)

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def root():
    return {
        "message": "Custard Apple Pest Management - AI System",
        "version": "2.1.0",
        "features": [
            "Disease Detection (Image)",
            "Pest Prediction (Weather)",
            "AI Treatment Recommendations",
            "Automated Spray Scheduling",
            "Real-time Weather Integration",
            "Location-based Predictions",
            "Active Learning Feedback Loop",
            "Phule Purandar Variety Support"
        ]
    }

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "disease_classifier": get_disease_classifier() is not None,
            "weather_predictor": get_weather_predictor() is not None,
            "ai_recommender": get_ai_recommender() is not None,
            "spray_scheduler": get_spray_scheduler() is not None
        }
    }

@router.get("/system-stats")
async def get_system_stats():
    """Get system statistics and performance metrics"""
    try:
        recommender = get_ai_recommender()
        
        return {
            "success": True,
            "statistics": {
                "total_feedbacks": len(recommender.treatment_history),
                "last_model_update": datetime.now().isoformat(),
                "supported_diseases": [
                    "Athracnose", "Mealy Bug", "Scale Insects", "White Flies",
                    "Diplodia Rot", "Leaf spot on fruit", "Leaf spot on Leaves",
                    "Blank Canker"
                ],
                "api_version": "2.0.0",
                "features": {
                    "real_time_weather": True,
                    "ai_recommendations": True,
                    "automated_scheduling": True,
                    "online_learning": True,
                    "research_integration": True
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-base")
async def get_knowledge_base():
    """Get chemical compositions and pest database"""
    try:
        kb_path = Path("knowledge_base")
        chemicals = json.loads((kb_path / "chemical_compositions.json").read_text())
        pests = json.loads((kb_path / "pest_database.json").read_text())
        return {
            "chemicals": chemicals,
            "pests": pests
        }
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Knowledge base not available")
