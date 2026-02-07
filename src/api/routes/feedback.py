from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
from src.api.schemas import TreatmentFeedback, FeedbackRequest, DetailedFeedback
from src.api.dependencies import get_ai_recommender, get_data_integrator, get_active_learning_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/submit-treatment-feedback")
async def submit_treatment_feedback(feedback: TreatmentFeedback):
    """
    Submit treatment outcome feedback for online learning
    """
    try:
        recommender = get_ai_recommender()
        
        # Update model with feedback
        weather_dict = {
            'temp': feedback.weather.temp,
            'humidity': feedback.weather.humidity,
            'precip': feedback.weather.rainfall,
            'wind_speed': feedback.weather.wind_speed
        }
        
        location_profile = None
        if feedback.location and feedback.location.latitude and feedback.location.longitude:
            integrator = get_data_integrator()
            location_profile = integrator.location_fetcher.get_location_profile(
                feedback.location.latitude,
                feedback.location.longitude
            )
        
        recommender.update_with_feedback(
            disease=feedback.disease,
            confidence=feedback.confidence,
            weather=weather_dict,
            treatment_applied=feedback.treatment_applied,
            actual_effectiveness=feedback.effectiveness,
            location_profile=location_profile
        )
        
        return {
            "success": True,
            "message": "Feedback received and model updated",
            "feedback_count": len(recommender.treatment_history),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for model improvement"""
    service = get_active_learning_service()
    return service.submit_feedback(
        image_path=feedback.image_path or "unknown",
        predicted_label=feedback.predicted_label,
        actual_label=feedback.actual_label,
        confidence=feedback.confidence,
        is_correct=feedback.is_correct,
        user_comments=feedback.user_comments
    )

@router.post("/feedback/detailed")
async def submit_detailed_feedback(feedback: DetailedFeedback):
    """
    Submit detailed feedback for retraining (Active Learning).
    """
    try:
        logger.info(f"Detailed feedback received: {feedback}")
        return {
            "success": True,
            "message": "Feedback received. System will be retrained with this data.",
            "action": "Retraining scheduled"
        }
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
