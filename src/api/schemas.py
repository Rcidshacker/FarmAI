from pydantic import BaseModel
from typing import Optional, List, Dict

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
