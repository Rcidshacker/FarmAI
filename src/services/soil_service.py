import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SoilService:
    """
    Service to retrieve soil data and maps.
    Integrates logic from legacy 'Ruchit's Folder' (Sitaphal_Soil_Static).
    """
    
    def __init__(self, static_data_path: Optional[str] = None):
        self.static_data_path = static_data_path
        self.soil_data = None
        self._load_data()
        
    def _load_data(self):
        """Load static soil CSV if available"""
        if self.static_data_path is None:
            logger.info("No static soil file provided. Using geospatial heuristics.")
            return

        if Path(self.static_data_path).exists():
            try:
                self.soil_data = pd.read_csv(self.static_data_path)
                logger.info("Loaded static soil data.")
            except Exception as e:
                logger.error(f"Failed to load soil data: {e}")
        else:
            logger.warning(f"Soil data file not found at {self.static_data_path}. Using geospatial heuristics.")

    def get_soil_profile(self, lat: float, lon: float) -> Dict:
        """
        Get soil profile for a specific location using live ISRIC SoilGrids API.
        Returns Clay %, pH, etc.
        """
        import requests
        import os
        
        # Default Fallback
        profile = {
            "type": "Loam", # Default
            "clay_percent": 30.0,
            "ph": 7.0,
            "nitrogen": "Medium",
            "phosphorus": "Medium",
            "potassium": "Medium",
            "organic_carbon": "Medium",
            "suitability": "Medium"
        }

        if os.getenv("OFFLINE_MODE", "False").lower() == "true":
            logger.info("Offline Mode: Using soil heuristics fallback.")
            if 18.0 <= lat <= 20.0 and 73.5 <= lon <= 75.0: # Pune/Nashik
                profile["type"] = "Heavy Clay" 
                profile["clay_percent"] = 55.0
            elif 16.0 <= lat <= 18.0 and 73.0 <= lon <= 74.5: # Kolhapur
                profile["type"] = "Sandy Loam"
                profile["clay_percent"] = 25.0
            return profile

        try:
            # Live Fetch from ISRIC SoilGrids
            # Query for Clay content at depth 0-5cm
            url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}&property=clay&depth=0-5cm&value=mean"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Extract mean value
                clay_raw = data['properties']['layers'][0]['depths'][0]['values']['mean']
                
                # Unit Conversion: raw / 10.0 = percentage
                clay_pct = clay_raw / 10.0
                profile["clay_percent"] = clay_pct
                
                # Determine Soil Type based on Clay %
                if clay_pct > 35:
                    profile["type"] = "Heavy Clay"
                elif clay_pct < 20:
                    profile["type"] = "Sandy Loam"
                else:
                    profile["type"] = "Loam"
                    
                logger.info(f"Live Soil Data: {profile['type']} ({clay_pct}%)")
            else:
                logger.info(f"SoilAPI Unavailable ({response.status_code}). Using fallback.")
                
        except Exception as e:
            logger.info(f"Soil Fetch Skipped: {e}. Using fallback.")
            # Fallback to Heuristics if API fails
            if 18.0 <= lat <= 20.0 and 73.5 <= lon <= 75.0: # Pune/Nashik
                profile["type"] = "Heavy Clay" # Black Cotton Soil is heavy clay
                profile["clay_percent"] = 55.0
            elif 16.0 <= lat <= 18.0 and 73.0 <= lon <= 74.5: # Kolhapur
                profile["type"] = "Sandy Loam" # Red Soil is lighter
                profile["clay_percent"] = 25.0
                
        return profile

    def get_fertilizer_recommendation(self, soil_profile: Dict, growth_stage: str) -> List[str]:
        """Get fertilizer schedule based on soil and stage"""
        recommendations = []
        
        # Nitrogen Management
        if soil_profile['nitrogen'] == 'Low':
            recommendations.append("Apply Urea (46% N) - 100g/plant")
            
        # Stage based
        if growth_stage == "Flowering":
            recommendations.append("0:52:34 (Mono Potassium Phosphate) - Foliar Spray")
            if soil_profile['type'] == "Black Cotton Soil":
                recommendations.append("Ensure drainage to prevent root rot")
                
        return recommendations
