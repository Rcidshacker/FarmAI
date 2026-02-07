from fastapi import APIRouter, File, UploadFile, HTTPException
from datetime import datetime
import logging
from PIL import Image
import numpy as np
import io

from src.api.dependencies import get_disease_classifier

router = APIRouter()
logger = logging.getLogger(__name__)

def get_quick_action(disease: str, confidence: float) -> str:
    """Generate quick action recommendation"""
    if confidence < 0.6:
        return "⚠️ Low confidence - Consider manual inspection or retake photo"
    elif confidence < 0.8:
        return "✓ Moderate confidence - Proceed with caution, monitor closely"
    else:
        urgency_map = {
            "Athracnose": "🔴 HIGH URGENCY - Apply fungicide immediately",
            "Diplodia Rot": "🔴 HIGH URGENCY - Remove infected parts, apply treatment",
            "Mealy Bug": "🟡 MEDIUM URGENCY - Apply insecticide within 24-48 hours",
            "Scale Insects": "🟡 MEDIUM URGENCY - Apply horticultural oil treatment",
            "White Flies": "🟡 MEDIUM URGENCY - Install yellow sticky traps, apply treatment",
            "Leaf spot on fruit": "🟢 MODERATE URGENCY - Apply fungicide as preventive",
            "Leaf spot on Leaves": "🟢 MODERATE URGENCY - Improve air circulation, apply treatment",
            "Blank Canker": "🔴 HIGH URGENCY - Prune affected areas, apply copper fungicide"
        }
        return urgency_map.get(disease, "✓ Treatment recommended - Consult detailed recommendations")

@router.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...), wind_speed: float = 0.0, fruit_density: str = "Medium"):
    """
    Detect disease from uploaded image with Physical Damage Heuristic
    
    Args:
        file: Image file (JPG, PNG)
        wind_speed: Current wind speed (km/h)
        fruit_density: Orchard fruit density (Low, Medium, High)
        
    Returns:
        Detected disease, confidence, and quick recommendations
    """
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = np.array(image)
        
        # Get classifier
        classifier = get_disease_classifier()
        
        # Predict
        predictions = classifier.predict(image)
        
        if not predictions:
            raise HTTPException(status_code=500, detail="Disease detection failed to produce results")
            
        # --- Physical Damage Heuristic ---
        # If wind is high and density is high, boost "Physical Damage" probability
        # This is a simple heuristic to aid the model
        rubbing_risk = 0.0
        if wind_speed > 20: rubbing_risk += 0.2
        if fruit_density == "High": rubbing_risk += 0.2
        
        # Adjust probabilities (simplified logic)
        # In a real scenario, we would modify the confidence scores directly
        # Here we just append a warning if rubbing risk is high and prediction is uncertain
        
        top_prediction = predictions[0]
        
        warning = None
        if rubbing_risk > 0.3 and top_prediction['class'] != 'Physical Damage (Rubbing)':
             warning = "High risk of Fruit Rubbing due to wind/density. Verify if spots are superficial."

        return {
            "success": True,
            "disease": top_prediction['class'],
            "confidence": top_prediction['confidence'],
            "all_predictions": predictions,
            "quick_action": get_quick_action(top_prediction['class'], top_prediction['confidence']),
            "rubbing_risk_warning": warning,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Disease detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
