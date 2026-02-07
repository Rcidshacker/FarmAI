from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
from src.api.schemas import ChatRequest
from src.api.dependencies import get_ai_assistant, get_data_integrator

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/assistant/chat")
async def chat_assistant(request: ChatRequest):
    """AI Assistant Chat with Auto-Context Injection"""
    try:
        assistant = get_ai_assistant()
        
        # 1. Enrich Context if missing
        context = request.context or {}
        
        # Hardcoded location for now (matching system default)
        lat, lon = 18.5204, 73.8567 
        location_name = "Pune"
        
        # Auto-fetch Weather if missing
        if not context.get('weather'):
            try:
                integrator = get_data_integrator()
                weather = integrator.weather_fetcher.get_current_weather(
                    location=location_name, lat=lat, lon=lon
                )
                context['weather'] = {
                    'temp': weather.get('temp'),
                    'humidity': weather.get('humidity'),
                    'rainfall': weather.get('rainfall', 0)
                }
            except Exception as w_err:
                logger.warning(f"Failed to auto-inject weather: {w_err}")

        # Auto-fetch Soil if missing
        if not context.get('soil'):
            try:
                from src.api.dependencies import get_soil_service
                soil_svc = get_soil_service()
                soil_data = soil_svc.get_soil_profile(lat, lon)
                context['soil'] = soil_data
            except Exception as s_err:
                logger.warning(f"Failed to auto-inject soil: {s_err}")
                
        # Ensure location is set
        if not context.get('location'):
            context['location'] = location_name

        # 2. Get Response
        response = assistant.get_response(request.query, context)
        
        return {
            "success": True,
            "response": response
        }
    except Exception as e:
        logger.error(f"Assistant error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/research-papers/{disease}")
async def get_research_papers(disease: str, max_results: int = 5):
    """
    Search for latest research papers on disease management
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
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Research papers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
