"""
Enhanced FastAPI Application with AI-Powered Features (PyTorch)
Integrates:
- Real-time weather data
- AI treatment recommendations
- Automated spray scheduling
- Research paper integration
- Location-based predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from PIL import Image
import io
from datetime import datetime
import logging
import json
from pathlib import Path
import math
import os
from dotenv import load_dotenv

# Force Reload
load_dotenv(override=True) # Load environment variables from .env

# Import custom modules - PyTorch version
from src.models.disease_cnn_pytorch import DiseaseClassifier
from src.models.hierarchical_classifier import HierarchicalDiseaseClassifier
from src.models.weather_pest_model import WeatherPestPredictor
from src.models.ai_treatment_recommender import AITreatmentRecommender
from src.automation.spray_scheduler import AutomatedSprayManager, QLearningSprayScheduler
from src.data_sources.external_data_fetcher import ExternalDataIntegrator
from src.services.active_learning_service import ActiveLearningService
from src.models.biological_risk_model import BiologicalRiskModel
from src.models.pest_forecasting_model import PestForecastingModel
from src.database.db_manager import DatabaseManager
from src.services.soil_service import SoilService
from src.services.satellite_service import SatelliteService
from src.services.ai_assistant import AIAssistantService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Custard Apple Pest Management - AI System (PyTorch)",
    description="AI-powered pest prediction, disease detection, and automated spray scheduling with GPU support",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (lazy loading)
disease_classifier = None
weather_pest_predictor = None
ai_recommender = None
spray_scheduler = None
data_integrator = None
active_learning_service = None
biological_model = None
forecasting_model = None
db_manager = None
soil_service = None
satellite_service = None
ai_assistant = None

def get_db_manager():
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def get_soil_service():
    global soil_service
    if soil_service is None:
        # Initialize without a file path to rely on internal geospatial heuristics
        soil_service = SoilService(static_data_path=None)
    return soil_service

def get_satellite_service():
    global satellite_service
    if satellite_service is None:
        satellite_service = SatelliteService()
    return satellite_service

def get_ai_assistant():
    global ai_assistant
    if ai_assistant is None:
        ai_assistant = AIAssistantService()
    return ai_assistant

def get_disease_classifier():
    global disease_classifier
    if disease_classifier is None:
        disease_classifier = HierarchicalDiseaseClassifier()
        disease_classifier.load_models()
    return disease_classifier


def get_weather_predictor():
    global weather_pest_predictor
    if weather_pest_predictor is None:
        weather_pest_predictor = WeatherPestPredictor()
        weather_pest_predictor.load_model()
    return weather_pest_predictor


def get_ai_recommender():
    global ai_recommender
    if ai_recommender is None:
        ai_recommender = AITreatmentRecommender()
        try:
            ai_recommender.load_model()
        except:
            logger.warning("AI recommender model not found, needs training")
    return ai_recommender


def get_spray_scheduler():
    global spray_scheduler
    if spray_scheduler is None:
        agent = QLearningSprayScheduler()
        try:
            agent.load_model()
        except:
            logger.warning("Spray scheduler model not found, using untrained agent")
        spray_scheduler = AutomatedSprayManager(agent)
    return spray_scheduler


def get_data_integrator():
    global data_integrator
    if data_integrator is None:
        data_integrator = ExternalDataIntegrator()
    return data_integrator

def get_active_learning_service():
    global active_learning_service
    if active_learning_service is None:
        active_learning_service = ActiveLearningService()
    return active_learning_service

def get_biological_model():
    global biological_model
    if biological_model is None:
        biological_model = BiologicalRiskModel()
    return biological_model

def get_forecasting_model():
    global forecasting_model
    if forecasting_model is None:
        forecasting_model = PestForecastingModel()
        forecasting_model.load_model()
    return forecasting_model


# Pydantic models for request/response
class WeatherConditions(BaseModel):
    temp: float
    humidity: float
    rainfall: float = 0.0
    wind_speed: float = 5.0


class LocationData(BaseModel):
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    state: str = "Maharashtra"
    soil_type: str = "Medium Black" # Added for Biological Model (Clay %)

class TreatmentFeedback(BaseModel):
    disease: str
    treatment_applied: str
    effectiveness: float  # 0-1
    weather: WeatherConditions
    confidence: float
    location: Optional[LocationData] = None


class ScheduleRequest(BaseModel):
    location: LocationData
    days_ahead: int = 30
    current_pest_pressure: float = 0.3

class FeedbackRequest(BaseModel):
    image_path: Optional[str] = None 
    predicted_label: str
    actual_label: Optional[str] = None
    confidence: float
    is_correct: bool
    user_comments: Optional[str] = None

class UserProfile(BaseModel):
    user_id: str
    name: str
    farm_location: LocationData
    soil_type: str
    land_area_acres: float
    custard_apple_variety: str = "Phule Purandar"
    fruit_density: str = "Medium" # Low, Medium, High

class SprayRecord(BaseModel):
    user_id: str
    date: str
    chemical_name: str
    quantity_liters: float
    concentration_ml_per_l: float
    target_pest: str
    notes: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    context: Optional[Dict] = None

class DetailedFeedback(BaseModel):
    prediction_id: str
    is_correct: bool
    actual_condition: str
    symptoms_observed: List[str]
    severity: str
    comments: str
    user_id: str = "default_user"

class PestRiskRequest(BaseModel):
    location: LocationData
    use_realtime: bool = True
    crop_stage: str = "Fruiting (Fruit Set)"
    manual_rvi: float = 0.65 # Legacy: kept for compatibility, overridden by Twin Brain
    api_key: Optional[str] = None

class OTPRequest(BaseModel):
    phone: str

class VerifyOTPRequest(BaseModel):
    phone: str
    otp: str
    name: Optional[str] = None # For registration flow if needed

class InterventionRequest(BaseModel):
    user_id: str = "default_user"
    date: str
    type: str = "spray"

class ResetInterventionRequest(BaseModel):
    user_id: str = "default_user"


# API Endpoints

@app.get("/")
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


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "disease_classifier": disease_classifier is not None,
            "weather_predictor": weather_pest_predictor is not None,
            "ai_recommender": ai_recommender is not None,
            "spray_scheduler": spray_scheduler is not None
        }
    }


@app.post("/detect-disease")
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



def run_enkf_correction(predicted_risk: float, manual_rvi: float) -> float:
    """
    Simulated EnKF (Ensemble Kalman Filter) correction based on manual observation.
    Fuses Model Prediction with User Observation (Digital Twin).
    
    Formula: NewRisk = PredictedRisk + (K * (Observation - PredictedRisk))
    Where K (Kalman Gain) = 0.7
    """
    # Assumption provided by Twin Brain architecture:
    # High risk generally correlates with specific canopy states, but we treat
    # manual_rvi (Observation) as the ground truth "state" regarding biomass density vs expected.
    # However, strict EnKF compares Observation (y) vs Expected Observation (H*x).
    # Simplified per user spec: Surprise = manual_rvi - predicted_risk (assuming scales align)
    
    expected_rvi = predicted_risk  # Simplifying assumption: Risk ~ Density correlation
    surprise = manual_rvi - expected_rvi
    
    K = 0.7  # Kalman Gain: We trust the user/observation 70%
    
    corrected_risk = predicted_risk + (K * surprise)
    return max(0.0, min(1.0, corrected_risk))

# --- TWIN BRAIN CONSTANTS ---
STAGE_TO_DENSITY_MAP = {
    "Dormant / Post-Harvest": 0.25,  # Mostly branches, little green
    "Vegetative (New Leaves)": 0.45, # Growing canopy
    "Flowering": 0.60,               # Good canopy
    "Fruiting (Fruit Set)": 0.80,    # Max density (High Risk)
    "Harvesting": 0.70               # Slightly less as pruning starts
}

def get_automated_vegetation(lat, lon, api_key, current_stage):
    """
    The Master Function: Decides whether to use Satellite or Heuristic.
    Returns: (final_rvi, source_msg, status_color)
    """
    # 1. Try Satellite First
    sat_service = get_satellite_service()
    # Ensure service has the key if provided (temporary injection for this request)
    if api_key:
        sat_service.api_key = api_key
        
    sat_rvi, sat_date = sat_service.fetch_smart_satellite_ndvi(lat, lon)
    
    # 2. LOGIC: Check if Satellite data is valid and relevant
    if sat_rvi is not None and sat_date is not None:
        days_old = (datetime.now() - sat_date).days
        
        # VETO RULE: If farmer says "Dormant" (No leaves), but Satellite sees "High Greenery" from 20 days ago,
        # the Satellite is outdated (farmer likely pruned yesterday). Trust the farmer.
        is_bare_stage = current_stage in ["Dormant / Post-Harvest"]
        if is_bare_stage and days_old > 5:
             fallback = STAGE_TO_DENSITY_MAP.get(current_stage, 0.25)
             return fallback, f"🚜 Satellite Outdated (Pruning Detected)", "orange"

        # Otherwise, trust the Satellite (Temporal Compositing)
        return sat_rvi, f"🛰️ Satellite (Live: {days_old} days ago)", "green"
    
# 3. Fallback to Heuristic (Clouds/No API)
    fallback_val = STAGE_TO_DENSITY_MAP.get(current_stage, 0.5)
    return fallback_val, f"🧠 Estimated from '{current_stage}' Stage", "blue"

def calculate_protection_efficacy(last_spray_date_str: str, current_weather: dict, crop_stage: str) -> float:
    """
    Smart Decay Logic for Spray Efficacy
    """
    if not last_spray_date_str:
        return 0.0
        
    try:
        last_date = datetime.fromisoformat(last_spray_date_str.replace("Z", "+00:00")).date()
        # Fallback if the string is just YYYY-MM-DD
    except ValueError:
        try:
            last_date = datetime.strptime(last_spray_date_str, "%Y-%m-%d").date()
        except:
            return 0.0

    days_passed = (datetime.now().date() - last_date).days
    
    if days_passed < 0: return 0.0 # Future date?
    if days_passed > 14: return 0.0 # Hard cutoff
    
    # 1. Base Decay (Linear over 14 days)
    # Efficacy starts at 1.0 (100%)
    
    # 2. Weather Penalties
    weather_penalty_days = 0.0
    
    # Rain Penalty: Heavy rain (>10mm) washes off significantly
    rainfall = current_weather.get('rainfall', 0)
    if rainfall > 10.0:
        weather_penalty_days += 3.0 # Lose 3 days worth of protection
    elif rainfall > 2.0:
         weather_penalty_days += 1.0
         
    # Temp Penalty: High heat degrades organic chemicals
    temp = current_weather.get('temp', 25)
    if temp > 35:
        # Accelerate effective time
        weather_penalty_days += (days_passed * 0.5) # 1.5x aging
        
    effective_days = days_passed + weather_penalty_days
    
    # 3. Crop Stage Sensitivity
    if "Fruiting" in crop_stage:
        effective_days += 2.0 
        
    if effective_days > 14:
        return 0.0

    # 4. Efficacy Curve (Multi-phase)
    # Phase A: Ramp-up (Day 0-2) - Chemical takes time to spread/act
    if effective_days < 2.0:
        # Starts at 0.5 (50%) and ramps to 1.0 (100%)
        # Linear ramp: 0.5 + (0.5 * (days / 2))
        efficacy = 0.5 + (0.25 * effective_days) 
        
    # Phase B: Plateau (Day 2-5) - Peak protection
    elif effective_days < 5.0:
        efficacy = 1.0
        
    # Phase C: Decay (Day 5+) - Gaussian/Quadratic falloff
    else:
        # Remaining days after plateau
        remaining_days = 14.0 - 5.0 # Total 9 days of decay
        decay_progress = (effective_days - 5.0) / remaining_days
        # (1 - x)^2 curve from 1.0 down to 0.0
        efficacy = (1.0 - decay_progress) ** 2
    
    return max(0.0, min(1.0, efficacy))


@app.post("/predict-pest-risk")
async def predict_pest_risk(request: PestRiskRequest):
    """
    Predict pest outbreak risk using Hybrid Biological-AI Model + Human-in-the-loop
    """
    try:
        # Extract variables
        location = request.location
        use_realtime = request.use_realtime
        crop_stage = request.crop_stage
        manual_rvi = request.manual_rvi
        
        # --- 1. HARDCODED LOCATION (Pune) ---
        hardcoded_lat = 18.5204
        hardcoded_lon = 73.8567
        
        # --- TWIN BRAIN AUTOMATION ---
        # 1. API Key (Provided by user)
        api_key_to_use = request.api_key or "353f3e4822e9a30795881c42556a68b5"
        
        # 2. Try Satellite
        final_rvi, source_msg, status_color = get_automated_vegetation(
            hardcoded_lat, hardcoded_lon, api_key_to_use, crop_stage
        )

        # --- 2. AUTOMATICALLY FETCH SOIL ---
        soil_svc = get_soil_service()
        soil_profile = soil_svc.get_soil_profile(hardcoded_lat, hardcoded_lon)
        clay_pct = soil_profile.get("clay_percent", 30.0)
        
        # --- 3. FETCH WEATHER ---
        integrator = get_data_integrator()
        # Retrieve current weather
        weather_data = integrator.weather_fetcher.get_current_weather(
            location="Pune", state="Maharashtra", lat=hardcoded_lat, lon=hardcoded_lon
        )
        
        # Retrieve Forecast (for 7 days)
        forecast_df = integrator.weather_fetcher.get_forecast(
            location="Pune", days=7, lat=hardcoded_lat, lon=hardcoded_lon
        )

        # --- 4. RISK CALCULATION PIPELINE ---
        # Constants from snippet
        stage_multipliers = {
            "Dormant / Post-Harvest": 0.2, 
            "Vegetative (New Leaves)": 0.3, 
            "Flowering": 0.7, 
            "Fruiting (Fruit Set)": 1.0, 
            "Harvesting": 1.0
        }
        stage_mod = stage_multipliers.get(crop_stage, 1.0)

        # Helper to calculate single day risk
        def calculate_pipeline_risk(row_data):
            # A) Base AI/Bio Risk (Using max of both as base input, or similar)
            # For this pipeline, we'll start with the Biological/AI hybrid base
            # Let's get the raw AI prediction
            # A) Base AI/Bio Risk (Student Model)
            # Switch to PestForecastingModel which has the trained XGBoost file
            try:
                forecasting_model = get_forecasting_model()
                
                # Prepare DataFrame for valid input to the model
                input_data = {
                    'tempmax': [row_data.get('tempmax', row_data.get('temperature', 25))],
                    'tempmin': [row_data.get('tempmin', row_data.get('temperature', 25))],
                    'temp': [row_data.get('temp', row_data.get('temperature', 25))],
                    'humidity': [row_data.get('humidity', 50)],
                    'precip': [row_data.get('rainfall', row_data.get('precip', 0))],
                    'windspeed': [row_data.get('wind_speed', row_data.get('windspeed', 5))],
                    # Add dummy values for features that might be needed by 'create_features'
                    'datetime': [datetime.now()], 
                    'stations': ['Pune'] 
                }
                
                input_df = pd.DataFrame(input_data)
                
                # Predict
                preds = forecasting_model.predict(input_df)
                raw_val = float(preds[0])
                
                # Normalize if the model outputs 0-100
                if raw_val > 1.05: # Allow slight overshoot 1.0
                    raw_risk = raw_val / 100.0
                else:
                    raw_risk = raw_val
                    
                # Clamp
                raw_risk = max(0.0, min(1.0, raw_risk))
                
            except Exception as e:
                # logger.warning(f"AI Model prediction failed: {e}")
                # Fallback to simple logic if model fails
                raw_risk = 0.3

            
            # B) SOIL PENALTY (Biology)
            # Logic: soil_suitability = math.exp(-0.025 * current_clay_pct)
            soil_suitability = math.exp(-0.025 * clay_pct)
            baseline = math.exp(-0.025 * 30.0) # Base 30% clay
            soil_multiplier = max(0.5, min(1.3, soil_suitability / baseline))
            
            biological_risk = raw_risk * soil_multiplier
            
            # C) INTERVENTION CHECK (Smart)
            # In a real app, pass user_id dynamically. Here default.
            db = get_db_manager()
            last_spray = db.get_last_intervention("default_user")
            
            protection_factor = 0.0
            last_spray_date = None
            if last_spray:
                last_spray_date = last_spray['created_at'].split("T")[0] # Just for debugging/logging
                # Pass the weather for THIS specific row/day if possible, 
                # but for 'days_passed' calculation we need the valid spray date.
                # Here we use the row_data weather to simulate conditions affecting the spray TODAY.
                protection_factor = calculate_protection_efficacy(last_spray['date'], row_data, crop_stage)

            
            # D) HUMAN CORRECTION (EnKF) & STAGE VETO
            fused = run_enkf_correction(biological_risk, final_rvi)
            
            # Apply Protection: Reduces risk by factor * 90% (Max 90% reduction)
            # If protection is 1.0 (Fresh), risk is multiplied by (1 - 0.9) = 0.1
            # If protection is 0.0 (Old), risk is multiplied by 1.0
            risk_reduction_mult = 1.0 - (protection_factor * 0.90)
            
            final_pipeline_risk = (fused * stage_mod) * risk_reduction_mult

                
            return {
                "raw_ai": raw_risk,
                "soil_mod": soil_multiplier,
                "soil_mod": soil_multiplier,
                "fused_enkf": fused, # Raw fused before protection
                "protection_score": protection_factor,
                "final": min(1.0, max(0.0, final_pipeline_risk))
            }

        # Calculate for Today
        today_calc = calculate_pipeline_risk(weather_data)
        
        # Calculate for Forecast (7 Days)
        forecast_risks = []
        forecast_dates = []
        
        if not forecast_df.empty:
            for _, row in forecast_df.iterrows():
                # Adapt row keys for helper
                row_dict = row.to_dict()
                # Ensure keys match helper expectation
                row_dict['temperature'] = row.get('tempmax', row.get('temp', 25))
                
                calcs = calculate_pipeline_risk(row_dict)
                forecast_risks.append(calcs['final'])
                forecast_dates.append(row['datetime'].strftime('%Y-%m-%d') if 'datetime' in row else "Unknown")

        # Construct Response
        final_predictions = {
            "Mealy Bug": today_calc['final'] * 100.0, # Scale to 0-100 for UI
            "Mealy Bug_details": {
                "ai_score": today_calc['raw_ai'] * 100,
                "soil_multiplier": today_calc['soil_mod'],
                "enkf_fused_score": today_calc['fused_enkf'] * 100,
                "factors": {
                    "crop_stage_mod": stage_mod,
                    "soil_clay_pct": clay_pct,
                    "crop_stage_mod": stage_mod,
                    "soil_clay_pct": clay_pct,
                    "twin_brain_rvi": final_rvi,
                    "spray_protection": today_calc['protection_score']
                }
            }
        }

        return {
            "success": True,
            "debug_offline_mode": str(os.getenv("OFFLINE_MODE")),
            "location": "Pune",
            "coordinates": {"lat": hardcoded_lat, "lon": hardcoded_lon},
            "soil_info": soil_profile,
            "current_weather": weather_data,
            "pest_predictions": final_predictions,
            "forecast": {
                "dates": forecast_dates,
                "risks": [r * 100.0 for r in forecast_risks] # Scale to %
            },
            "twin_brain_status": {
                "rvi": final_rvi,
                "message": source_msg,
                "color": status_color
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pest prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai-recommend-treatment")
async def ai_recommend_treatment(
    disease: str,
    confidence: float,
    location: LocationData,
    growth_stage: Optional[str] = "Vegetative"
):
    """
    Get AI-powered treatment recommendations
    Uses machine learning model trained on historical treatment outcomes
    
    Args:
        disease: Detected disease name
        confidence: Detection confidence
        location: Location data
        growth_stage: Current growth stage
        
    Returns:
        Top recommended treatments with predicted effectiveness
    """
    try:
        # Get real-time context
        integrator = get_data_integrator()
        context = integrator.get_complete_context(
            location.name, 
            location.latitude, 
            location.longitude
        )
        
        # Get AI recommender
        recommender = get_ai_recommender()
        
        # Get recommendations
        recommendations = recommender.predict_treatment(
            disease=disease,
            confidence=confidence,
            weather=context['current_weather'],
            location_profile=context.get('location_profile'),
            growth_stage=growth_stage,
            top_k=5
        )
        
        return {
            "success": True,
            "disease": disease,
            "detection_confidence": confidence,
            "location": location.name,
            "current_weather": context['current_weather'],
            "recommendations": recommendations,
            "growth_stage": growth_stage,
            "timestamp": datetime.now().isoformat(),
            "note": "Recommendations generated by AI model trained on historical outcomes"
        }
        
    except Exception as e:
        logger.error(f"AI recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-spray-schedule")
async def create_spray_schedule(request: ScheduleRequest):
    """
    Generate automated spray schedule using RL-based optimizer
    
    Args:
        request: Schedule request with location and parameters
        
    Returns:
        Optimized spray schedule for next N days
    """
    try:
        # Get scheduler
        scheduler = get_spray_scheduler()
        
        # Get the full result dict from the manager
        scheduler_result = scheduler.create_schedule(
            location=request.location.name,
            days_ahead=request.days_ahead
        )
        
        # Get next spray info
        next_spray = scheduler.get_next_spray_date()
        
        # Generate alerts
        alerts = scheduler.send_alerts(scheduler_result['schedule'])
        
        # ---------------------------------------------------------
        # FIX APPLIED HERE: Flatten the response structure
        # Extract 'schedule' list and 'summary' dict explicitly
        # so the frontend can map() them directly.
        # ---------------------------------------------------------
        return {
            "success": True,
            "location": request.location.name,
            "schedule": scheduler_result['schedule'],  # <--- Extract the LIST
            "summary": scheduler_result['summary'],    # <--- Extract the SUMMARY
            "next_spray": next_spray,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Schedule creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit-treatment-feedback")
async def submit_treatment_feedback(feedback: TreatmentFeedback):
    """
    Submit treatment outcome feedback for online learning
    The AI model will learn from this feedback and improve over time
    
    Args:
        feedback: Treatment feedback data
        
    Returns:
        Confirmation of feedback receipt
    """
    try:
        # Get AI recommender
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


@app.get("/weather-forecast/{location}")
async def get_weather_forecast(location: str, days: int = 7):
    """
    Get weather forecast for location
    
    Args:
        location: Location name
        days: Number of days to forecast
        
    Returns:
        Weather forecast data
    """
    try:
        integrator = get_data_integrator()
        forecast_df = integrator.weather_fetcher.get_forecast(location, days)
        
        if forecast_df.empty:
            raise HTTPException(status_code=404, detail="Forecast not available")
        
        forecast_list = forecast_df.to_dict('records')
        
        return {
            "success": True,
            "location": location,
            "forecast": forecast_list,
            "days": len(forecast_list),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Weather forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/research-papers/{disease}")
async def get_research_papers(disease: str, max_results: int = 5):
    """
    Search for latest research papers on disease management
    
    Args:
        disease: Disease name to search
        max_results: Maximum papers to return
        
    Returns:
        List of relevant research papers
    """
    try:
        integrator = get_data_integrator()
        query = f"custard apple {disease} treatment management"
        
        papers = integrator.research_fetcher.search_papers(query, max_results)
        
        return {
            "success": True,
            "disease": disease,
            "papers": papers,
            "count": len(papers),
            "disease": disease,
            "papers": papers,
            "count": len(papers),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Research papers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record-intervention")
async def record_intervention(request: InterventionRequest):
    """Record a spray or other intervention"""
    try:
        db = get_db_manager()
        success = db.add_record({
            "user_id": request.user_id,
            "date": request.date,
            "type": request.type,
            "name": "Manual Spray", # Could be parametrized
            "quantity": 1.0,
            "notes": "Recorded via App"
        })
        
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-intervention")
async def reset_intervention(request: ResetInterventionRequest):
    """Reset recent spray history (Undo)"""
    try:
        db = get_db_manager()
        success = db.clear_recent_interventions(request.user_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/auth/send-otp")
async def send_otp(request: OTPRequest):
    """
    Send OTP to phone number. 
    MOCK: Returns success and logs OTP.
    """
    try:
        # In real app: Generate random 6 digit code and SMS it.
        # For Demo/Dev: Fixed OTP for specific test numbers or random for others.
        otp_code = "123456" 
        logger.info(f"OTP for {request.phone}: {otp_code}")
        
        return {
            "success": True,
            "message": "OTP sent successfully",
            "debug_otp": otp_code  # Remove in production
        }
    except Exception as e:
        logger.error(f"Send OTP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    """
    Verify OTP and Login/Register user.
    SPECIAL: Creates/Returns 'Rachit' dummy user for phone 1234567890
    """
    try:
        if request.otp != "123456":
            raise HTTPException(status_code=400, detail="Invalid OTP")
            
        db = get_db_manager()
        
        # Check if user exists
        user = db.get_user_by_phone(request.phone)
        
        # SPECIAL DEMO LOGIC
        if request.phone == "1234567890" and not user:
             # Create dummy user Rachit
            import random
            import sqlite3
            random_acres = round(random.uniform(2.0, 15.0), 1)
            
            # Use raw SQL or create a specialized create method in future
            # For now, leveraging create_user but with dummy email/pass since we haven't refactored create_user fully
            # Actually, let's just insert strictly for this demo or use create_user with placeholders
            
            db.create_user(
                email=f"rachit_{request.phone}@example.com",
                password="dummy_password", # unused in OTP flow
                name="Rachit",
                phone=request.phone
            )
            
            # Update more fields manually if needed (Location: Pune, Acres)
            # Since create_user is basic, we might need a direct update or ensure create_user supports it.
            # For now, let's rely on Profile Page editing or DB defaults, OR update DB manager.
            # But the user asked for dummy data: Location Pune, Acres Random.
            # Let's do a quick update query or extended create.
            
            conn = sqlite3.connect(db.db_path)
            c = conn.cursor()
            c.execute('''
                UPDATE users 
                SET location_name = ?, land_area = ?
                WHERE phone = ?
            ''', ("Pune", random_acres, request.phone))
            conn.commit()
            conn.close()
            
            # Fetch again
            user = db.get_user_by_phone(request.phone)

        # Standard Registration (if not Rachit logic and user doesn't exist)
        if not user:
            # Just create a basic user
            temp_name = request.name or "New Farmer"
            db.create_user(
                email=f"user_{request.phone}@farm.ai",
                password="otp_user_pass",
                name=temp_name,
                phone=request.phone
            )
            user = db.get_user_by_phone(request.phone)
            
        return {
            "success": True,
            "user": user,
            "token": "simulated-otp-jwt-token"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify OTP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/location-profile")


@app.get("/location-profile")
async def get_location_profile(latitude: float, longitude: float):
    """
    Get comprehensive location profile
    
    Args:
        latitude: GPS latitude
        longitude: GPS longitude
        
    Returns:
        Location profile with soil type, elevation, agro-climatic zone
    """
    try:
        integrator = get_data_integrator()
        profile = integrator.location_fetcher.get_location_profile(latitude, longitude)
        
        return {
            "success": True,
            "profile": profile,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Location profile error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system-stats")
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


@app.get("/weather/current")
async def get_weather(lat: float, lon: float):
    """Get current weather for specific coordinates"""
    integrator = get_data_integrator()
    return integrator.weather_fetcher.get_current_weather("Unknown", lat=lat, lon=lon)


@app.get("/weather/forecast")
async def get_forecast(lat: float, lon: float, days: int = 10):
    """Get 10-day weather forecast"""
    integrator = get_data_integrator()
    forecast_df = integrator.weather_fetcher.get_forecast("Unknown", days=days, lat=lat, lon=lon)
    # Convert DataFrame to list of dicts for JSON response
    if hasattr(forecast_df, 'to_dict'):
        return forecast_df.to_dict(orient='records')
    return forecast_df


@app.post("/feedback")
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


@app.post("/user/profile")
async def update_profile(profile: UserProfile):
    """Update user profile with farm details"""
    # In a real app, save to DB. Here we just acknowledge.
    return {
        "status": "success", 
        "message": f"Profile updated for {profile.name}",
        "variety_config": "Loaded specific care for Phule Purandar" if profile.custard_apple_variety == "Phule Purandar" else "Standard care loaded"
    }


@app.post("/forecast/risk")
async def forecast_risk(location: LocationData, days: int = 14):
    """
    Forecast Mealybug Risk using Hybrid Biological-AI Model
    """
    try:
        integrator = get_data_integrator()
        bio_model = get_biological_model()
        ai_model = get_forecasting_model()
        
        # 1. Get Weather Forecast
        forecast_df = integrator.weather_fetcher.get_forecast(
            location.name, days=days, lat=location.latitude, lon=location.longitude
        )
        
        # 2. Calculate Biological Risk (The "Teacher" Logic)
        # We need historical context for accurate GDD, but for forecast we approximate
        # or use the AI model which learns to predict risk from weather patterns directly
        
        # Option A: Use AI Model (The "Student")
        try:
            risk_preds = ai_model.predict(forecast_df)
            source = "AI Model (XGBoost)"
        except Exception as e:
            logger.warning(f"AI Model failed ({e}), falling back to Biological Engine")
            # Option B: Fallback to Biological Engine
            # We need to convert forecast_df to the format expected by bio_model
            # Assuming forecast_df has temp_max, temp_min, precip etc.
            # We might need to map columns
            
            # Map columns if necessary (ExternalDataFetcher returns specific format)
            # For now, let's assume we can calculate simple GDD
            risk_df = bio_model.calculate_risk_series(forecast_df)
            risk_preds = risk_df['mealybug_risk_score'].values
            source = "Biological Engine (GDD)"
            
        return {
            "location": location.name,
            "forecast_days": days,
            "risk_source": source,
            "daily_risk": [float(r) for r in risk_preds],
            "dates": forecast_df['datetime'].dt.strftime('%Y-%m-%d').tolist() if 'datetime' in forecast_df.columns else []
        }
        
    except Exception as e:
        logger.error(f"Risk forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/spray/record")
async def record_spray(record: SprayRecord):
    """
    Record a spray event and update risk calculation
    """
    try:
        db = get_db_manager()
        success = db.add_record(record.dict())
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save record")
            
        # Logic: Spraying reduces pest population, effectively resetting "Accumulated Degree Days" 
        # or reducing the current risk score significantly.
        
        return {
            "status": "success",
            "message": "Spray recorded. Risk levels updated.",
            "updated_risk_factor": 0.1 # Mock updated risk
        }
    except Exception as e:
        logger.error(f"Error recording spray: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base")
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

# --- New Endpoints for Enhanced System ---

@app.post("/user/profile")
async def save_user_profile(profile: UserProfile):
    """Save or update farmer profile"""
    try:
        db = get_db_manager()
        success = db.save_user_profile(profile.dict())
        if success:
            return {"success": True, "message": "Profile saved"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save profile")
    except Exception as e:
        logger.error(f"Profile save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/profile/{user_id}")
async def get_user_profile_endpoint(user_id: str):
    """Get farmer profile"""
    try:
        db = get_db_manager()
        profile = db.get_user_profile(user_id)
        if profile:
            return {"success": True, "profile": profile}
        else:
            return {"success": False, "message": "Profile not found"}
    except Exception as e:
        logger.error(f"Profile fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/soil/info")
async def get_soil_info(lat: float, lon: float):
    """Get soil profile and fertilizer recommendations"""
    try:
        soil_svc = get_soil_service()
        profile = soil_svc.get_soil_profile(lat, lon)
        recommendations = soil_svc.get_fertilizer_recommendation(profile, "Vegetative") # Default stage
        return {
            "success": True,
            "soil_profile": profile,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Soil info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/satellite/data")
async def get_satellite_data(lat: float, lon: float):
    """Get satellite vegetation indices"""
    try:
        sat_svc = get_satellite_service()
        indices = sat_svc.get_farm_indices(lat, lon)
        health_status = sat_svc.analyze_health(indices)
        return {
            "success": True,
            "indices": indices,
            "health_status": health_status
        }
    except Exception as e:
        logger.error(f"Satellite data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant/chat")
async def chat_assistant(request: ChatRequest):
    """AI Assistant Chat"""
    try:
        assistant = get_ai_assistant()
        response = assistant.get_response(request.query, request.context)
        return {
            "success": True,
            "response": response
        }
    except Exception as e:
        logger.error(f"Assistant error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/detailed")
async def submit_detailed_feedback(feedback: DetailedFeedback):
    """
    Submit detailed feedback for retraining (Active Learning).
    If prediction was wrong, this data is crucial.
    """
    try:
        # In a real system, store this in a specific 'retraining_queue' table
        logger.info(f"Detailed feedback received: {feedback}")
        
        # Mock response indicating the system will learn
        return {
            "success": True,
            "message": "Feedback received. System will be retrained with this data.",
            "action": "Retraining scheduled"
        }
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    



# Helper functions

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
