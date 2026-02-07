from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
from src.api.dependencies import (
    get_data_integrator, get_soil_service, get_satellite_service
)

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/weather-forecast/{location}")
async def get_weather_forecast(location: str, days: int = 7):
    """
    Get weather forecast for location
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

@router.get("/weather/current")
async def get_weather(lat: float, lon: float):
    """Get current weather for specific coordinates"""
    integrator = get_data_integrator()
    return integrator.weather_fetcher.get_current_weather("Unknown", lat=lat, lon=lon)

@router.get("/weather/forecast")
async def get_forecast(lat: float, lon: float, days: int = 10):
    """Get 10-day weather forecast"""
    integrator = get_data_integrator()
    forecast_df = integrator.weather_fetcher.get_forecast("Unknown", days=days, lat=lat, lon=lon)
    if hasattr(forecast_df, 'to_dict'):
        return forecast_df.to_dict(orient='records')
    return forecast_df

@router.get("/location-profile")
async def get_location_profile(latitude: float, longitude: float):
    """Get comprehensive location profile"""
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

@router.get("/soil/info")
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

@router.get("/satellite/data")
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
