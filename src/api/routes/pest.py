from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
import math
import os
import pandas as pd
from src.api.schemas import PestRiskRequest, LocationData
from src.api.dependencies import (
    get_data_integrator, get_soil_service, get_satellite_service,
    get_forecasting_model, get_db_manager, get_biological_model
)

router = APIRouter()
logger = logging.getLogger(__name__)

# --- TWIN BRAIN CONSTANTS ---
STAGE_TO_DENSITY_MAP = {
    "Dormant / Post-Harvest": 0.25,  # Mostly branches, little green
    "Vegetative (New Leaves)": 0.45, # Growing canopy
    "Flowering": 0.60,               # Good canopy
    "Fruiting (Fruit Set)": 0.80,    # Max density (High Risk)
    "Harvesting": 0.70               # Slightly less as pruning starts
}

def run_enkf_correction(predicted_risk: float, manual_rvi: float) -> float:
    """
    Simulated EnKF (Ensemble Kalman Filter) correction based on manual observation.
    Fuses Model Prediction with User Observation (Digital Twin).
    """
    expected_rvi = predicted_risk  # Simplifying assumption: Risk ~ Density correlation
    surprise = manual_rvi - expected_rvi
    
    K = 0.7  # Kalman Gain: We trust the user/observation 70%
    
    corrected_risk = predicted_risk + (K * surprise)
    return max(0.0, min(1.0, corrected_risk))

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
        efficacy = 0.5 + (0.25 * effective_days) 
        
    # Phase B: Plateau (Day 2-5) - Peak protection
    elif effective_days < 5.0:
        efficacy = 1.0
        
    # Phase C: Decay (Day 5+) - Gaussian/Quadratic falloff
    else:
        remaining_days = 14.0 - 5.0 # Total 9 days of decay
        decay_progress = (effective_days - 5.0) / remaining_days
        efficacy = (1.0 - decay_progress) ** 2
    
    return max(0.0, min(1.0, efficacy))


@router.post("/predict-pest-risk")
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
            # A) Base AI/Bio Risk
            try:
                forecasting_model = get_forecasting_model()
                
                input_data = {
                    'tempmax': [row_data.get('tempmax', row_data.get('temperature', 25))],
                    'tempmin': [row_data.get('tempmin', row_data.get('temperature', 25))],
                    'temp': [row_data.get('temp', row_data.get('temperature', 25))],
                    'humidity': [row_data.get('humidity', 50)],
                    'precip': [row_data.get('rainfall', row_data.get('precip', 0))],
                    'windspeed': [row_data.get('wind_speed', row_data.get('windspeed', 5))],
                    'datetime': [datetime.now()], 
                    'stations': ['Pune'] 
                }
                
                input_df = pd.DataFrame(input_data)
                
                # Predict
                preds = forecasting_model.predict(input_df)
                raw_val = float(preds[0])
                
                # Normalize
                if raw_val > 1.05: 
                    raw_risk = raw_val / 100.0
                else:
                    raw_risk = raw_val
                    
                raw_risk = max(0.0, min(1.0, raw_risk))
                
            except Exception as e:
                # Fallback
                raw_risk = 0.3

            # B) SOIL PENALTY (Biology)
            soil_suitability = math.exp(-0.025 * clay_pct)
            baseline = math.exp(-0.025 * 30.0)
            soil_multiplier = max(0.5, min(1.3, soil_suitability / baseline))
            
            biological_risk = raw_risk * soil_multiplier
            
            # C) INTERVENTION CHECK
            db = get_db_manager()
            last_spray = db.get_last_intervention("default_user")
            
            protection_factor = 0.0
            if last_spray:
                protection_factor = calculate_protection_efficacy(last_spray['date'], row_data, crop_stage)
            
            # D) HUMAN CORRECTION (EnKF) & STAGE VETO
            fused = run_enkf_correction(biological_risk, final_rvi)
            
            risk_reduction_mult = 1.0 - (protection_factor * 0.90)
            
            final_pipeline_risk = (fused * stage_mod) * risk_reduction_mult
                
            return {
                "raw_ai": raw_risk,
                "soil_mod": soil_multiplier,
                "fused_enkf": fused, 
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
                row_dict = row.to_dict()
                row_dict['temperature'] = row.get('tempmax', row.get('temp', 25))
                
                calcs = calculate_pipeline_risk(row_dict)
                forecast_risks.append(calcs['final'])
                forecast_dates.append(row['datetime'].strftime('%Y-%m-%d') if 'datetime' in row else "Unknown")

        # Construct Response
        final_predictions = {
            "Mealy Bug": today_calc['final'] * 100.0,
            "Mealy Bug_details": {
                "ai_score": today_calc['raw_ai'] * 100,
                "soil_multiplier": today_calc['soil_mod'],
                "enkf_fused_score": today_calc['fused_enkf'] * 100,
                "factors": {
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
                "risks": [r * 100.0 for r in forecast_risks]
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


@router.post("/forecast/risk")
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
        
        # Option A: Use AI Model (The "Student")
        try:
            risk_preds = ai_model.predict(forecast_df)
            source = "AI Model (XGBoost)"
        except Exception as e:
            logger.warning(f"AI Model failed ({e}), falling back to Biological Engine")
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
