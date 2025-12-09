"""
External Data Fetcher - Fetches real-time data from multiple sources
- Weather: IMD (India Meteorological Department) API
- Fertilizers: Agricultural research databases
- Research Papers: arXiv, PubMed, ResearchGate
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import time
from bs4 import BeautifulSoup
import logging
import numpy as np

logger = logging.getLogger(__name__)


class IMDWeatherFetcher:
    """Fetch real-time and forecast weather data from IMD"""
    
    def __init__(self):
        self.base_url = "https://api.data.gov.in/resource"
        # ---------------------------------------------------------
        # FIX: Use Visual Crossing Key
        # ---------------------------------------------------------
        self.api_key = "YOUR_DATA_GOV_IN_API_KEY"  # Keep for IMD fallback
        self.vc_key = "ENPYVRQT5SRYBZFGW68BCB4CP"  # Your Visual Crossing Key
        
    def get_current_weather(self, location: str, state: str = "Maharashtra", lat: Optional[float] = None, lon: Optional[float] = None) -> Dict:
        """
        Get current weather for a location or coordinates
        
        Args:
            location: City/District name
            state: State name
            lat: Latitude (optional)
            lon: Longitude (optional)
            
        Returns:
            Dict with temperature, humidity, rainfall, wind speed
        """
        try:
            # If coordinates are provided, use OpenWeatherMap (more accurate for specific farm location)
            if lat is not None and lon is not None:
                return self._get_weather_by_coords(lat, lon)

            # IMD API endpoint
            endpoint = f"{self.base_url}/9ef84268-d588-465a-a308-a864a43d0070"
            
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'filters[state]': state,
                'filters[district]': location,
                'limit': 1
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'records' in data and len(data['records']) > 0:
                record = data['records'][0]
                return {
                    'location': location,
                    'state': state,
                    'temperature': float(record.get('temperature', 0)),
                    'humidity': float(record.get('humidity', 0)),
                    'rainfall': float(record.get('rainfall', 0)),
                    'wind_speed': float(record.get('wind_speed', 0)),
                    'datetime': record.get('last_updated', datetime.now().isoformat())
                }
            else:
                logger.warning(f"No weather data found for {location}, {state}")
                return self._get_fallback_weather(location)
                
        except Exception as e:
            logger.error(f"Error fetching IMD weather: {e}")
            return self._get_fallback_weather(location)

    def _get_weather_by_coords(self, lat: float, lon: float) -> Dict:
        """Helper to get weather by coordinates using Visual Crossing"""
        try:
            # Visual Crossing Timeline API (Current Weather)
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}?unitGroup=metric&key={self.vc_key}&include=current"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Parse Current Conditions
            current = data.get('currentConditions', {})
            
            return {
                'location': f"Lat: {lat}, Lon: {lon}",
                'state': 'Unknown',
                'temperature': current.get('temp', 0),
                'humidity': current.get('humidity', 0),
                'rainfall': current.get('precip', 0),  # VC returns 0 if no rain
                'wind_speed': current.get('windspeed', 0),
                'datetime': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching Visual Crossing weather: {e}")
            return self._get_fallback_weather("Unknown")
    
    def get_forecast(self, location: str, days: int = 14, lat: Optional[float] = None, lon: Optional[float] = None) -> pd.DataFrame:
        """
        Get weather forecast using Visual Crossing
        """
        try:
            # Determine location string (Lat,Lon is more precise)
            loc_query = f"{lat},{lon}" if (lat and lon) else location
            
            # Visual Crossing Timeline API (Forecast)
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{loc_query}?unitGroup=metric&key={self.vc_key}&include=days"
            
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            # VC returns daily data directly - No aggregation needed!
            for day_data in data.get('days', [])[:days]:
                forecasts.append({
                    'datetime': pd.to_datetime(day_data['datetime']),
                    'temp': day_data.get('temp'),          # Daily Mean
                    'tempmax': day_data.get('tempmax'),    # Daily Max
                    'tempmin': day_data.get('tempmin'),    # Daily Min
                    'humidity': day_data.get('humidity'),
                    'precip': day_data.get('precip', 0.0),
                    'windspeed': day_data.get('windspeed', 0.0),
                    'cloudcover': day_data.get('cloudcover', 0.0),
                    'solarradiation': day_data.get('solarradiation', 0.0),
                    'uvindex': day_data.get('uvindex', 0.0)
                })
            
            return pd.DataFrame(forecasts).sort_values('datetime')
            
        except Exception as e:
            logger.error(f"Error fetching forecast from Visual Crossing: {e}")
            # Fallback to mock if API fails
            return self._generate_mock_forecast(days)
    
    def _generate_mock_forecast(self, days: int) -> pd.DataFrame:
        """Generate mock forecast data for testing"""
        dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
        forecasts = []
        
        for date in dates:
            forecasts.append({
                'datetime': date,
                'tempmax': 32.0 + np.random.uniform(-2, 2),
                'tempmin': 22.0 + np.random.uniform(-2, 2),
                'humidity': 65.0 + np.random.uniform(-10, 10),
                'precip': 0.0 if np.random.random() > 0.3 else np.random.uniform(0, 15),
                'wind_speed': 5.0 + np.random.uniform(-2, 2)
            })
            
        return pd.DataFrame(forecasts)
    
    def _get_fallback_weather(self, location: str) -> Dict:
        """Fallback to local CSV data if API fails"""
        try:
            # Try to load from local CSV files
            import glob
            csv_files = glob.glob("*.csv")
            
            for file in csv_files:
                if location.lower() in file.lower():
                    df = pd.read_csv(file)
                    latest = df.iloc[-1]
                    
                    return {
                        'location': location,
                        'temperature': float(latest.get('temperature', latest.get('temp', 25))),
                        'humidity': float(latest.get('humidity', 70)),
                        'rainfall': float(latest.get('precipitation', latest.get('precip', 0))),
                        'wind_speed': float(latest.get('wind_speed', 5)),
                        'datetime': datetime.now().isoformat(),
                        'source': 'local_csv'
                    }
            
            # Ultimate fallback - typical Maharashtra values
            return {
                'location': location,
                'temperature': 28.0,
                'humidity': 65.0,
                'rainfall': 0.0,
                'wind_speed': 5.0,
                'datetime': datetime.now().isoformat(),
                'source': 'default'
            }
            
        except Exception as e:
            logger.error(f"Fallback weather also failed: {e}")
            return {}


class FertilizerDataFetcher:
    """Fetch fertilizer and chemical composition data from agricultural databases"""
    
    def __init__(self):
        self.cache_file = "data/fertilizer_cache.json"
        self.cache_duration = 30  # days
        
    def fetch_fertilizer_data(self, product_name: str) -> Dict:
        """
        Fetch detailed fertilizer composition and effectiveness data
        
        Args:
            product_name: Name of fertilizer/chemical
            
        Returns:
            Dict with composition, effectiveness, application guidelines
        """
        # Check cache first
        cached = self._get_from_cache(product_name)
        if cached:
            return cached
        
        try:
            # Method 1: Scrape from Mendeley dataset
            mendeley_data = self._fetch_from_mendeley(product_name)
            if mendeley_data:
                self._save_to_cache(product_name, mendeley_data)
                return mendeley_data
            
            # Method 2: Scrape from agricultural databases
            agri_data = self._fetch_from_agri_databases(product_name)
            if agri_data:
                self._save_to_cache(product_name, agri_data)
                return agri_data
            
            logger.warning(f"No data found for {product_name}, using defaults")
            return self._get_default_fertilizer_data(product_name)
            
        except Exception as e:
            logger.error(f"Error fetching fertilizer data: {e}")
            return {}
    
    def _fetch_from_mendeley(self, product_name: str) -> Optional[Dict]:
        """Fetch from Mendeley datasets"""
        try:
            # Mendeley dataset API (requires authentication)
            # For demonstration purposes
            url = "https://data.mendeley.com/api/datasets/jrgh288syf/2"
            
            # This would require proper API authentication
            # Placeholder for actual implementation
            logger.info(f"Fetching {product_name} from Mendeley...")
            
            return None  # Implement actual API call
            
        except Exception as e:
            logger.error(f"Mendeley fetch error: {e}")
            return None
    
    def _fetch_from_agri_databases(self, product_name: str) -> Optional[Dict]:
        """Fetch from agricultural research databases"""
        try:
            # Example: Scrape from ICAR, agricultural university databases
            # This is a placeholder for actual implementation
            
            return None
            
        except Exception as e:
            logger.error(f"Agri database fetch error: {e}")
            return None
    
    def _get_from_cache(self, product_name: str) -> Optional[Dict]:
        """Get data from local cache if recent"""
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            
            if product_name in cache:
                entry = cache[product_name]
                cached_date = datetime.fromisoformat(entry['cached_at'])
                
                if (datetime.now() - cached_date).days < self.cache_duration:
                    return entry['data']
            
            return None
            
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return None
    
    def _save_to_cache(self, product_name: str, data: Dict):
        """Save data to local cache"""
        try:
            cache = {}
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
            except FileNotFoundError:
                pass
            
            cache[product_name] = {
                'data': data,
                'cached_at': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
                
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    def _get_default_fertilizer_data(self, product_name: str) -> Dict:
        """Return default data structure"""
        return {
            'name': product_name,
            'composition': 'Unknown',
            'effectiveness': 0.75,
            'application_rate': 'As per label',
            'source': 'default'
        }


class ResearchPaperFetcher:
    """Fetch and extract knowledge from agricultural research papers"""
    
    def __init__(self):
        self.arxiv_base = "http://export.arxiv.org/api/query"
        self.cache_file = "data/research_cache.json"
        
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for research papers on pest management
        
        Args:
            query: Search query (e.g., "custard apple anthracnose treatment")
            max_results: Maximum papers to return
            
        Returns:
            List of paper metadata with abstracts
        """
        try:
            # Search arXiv for agricultural papers
            papers = self._search_arxiv(query, max_results)
            
            # Could also search PubMed, Google Scholar, ResearchGate
            # papers.extend(self._search_pubmed(query, max_results))
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
    
    def _search_arxiv(self, query: str, max_results: int) -> List[Dict]:
        """Search arXiv for papers"""
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.arxiv_base, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse XML response
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = {
                    'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip(),
                    'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text.strip(),
                    'authors': [author.find('{http://www.w3.org/2005/Atom}name').text 
                               for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
                    'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
                    'link': entry.find('{http://www.w3.org/2005/Atom}id').text
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []
    
    def extract_treatment_info(self, abstract: str) -> Dict:
        """
        Extract treatment information from paper abstract using NLP
        
        Args:
            abstract: Paper abstract text
            
        Returns:
            Dict with extracted treatment info
        """
        # This would use NLP (spaCy, transformers) to extract:
        # - Chemical names
        # - Dosages
        # - Effectiveness percentages
        # - Application methods
        
        # Placeholder for actual NLP implementation
        return {
            'chemicals': [],
            'dosages': [],
            'effectiveness': [],
            'methods': []
        }


class LocationDataFetcher:
    """Fetch location-specific data including soil type, microclimate, etc."""
    
    def __init__(self):
        pass
    
    def get_location_profile(self, latitude: float, longitude: float) -> Dict:
        """
        Get comprehensive location profile
        
        Args:
            latitude: GPS latitude
            longitude: GPS longitude
            
        Returns:
            Dict with soil type, elevation, microclimate zone
        """
        try:
            # This would fetch from:
            # - Soil data: NBSS&LUP (National Bureau of Soil Survey)
            # - Elevation: SRTM data
            # - Microclimate: AgriStack API
            
            location_data = {
                'latitude': latitude,
                'longitude': longitude,
                'district': self._get_district(latitude, longitude),
                'soil_type': self._get_soil_type(latitude, longitude),
                'elevation': self._get_elevation(latitude, longitude),
                'agro_climatic_zone': self._get_agro_zone(latitude, longitude)
            }
            
            return location_data
            
        except Exception as e:
            logger.error(f"Error fetching location profile: {e}")
            return {}
    
    def _get_district(self, lat: float, lon: float) -> str:
        """Reverse geocode to get district"""
        try:
            # Use OpenStreetMap Nominatim API
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            address = data.get('address', {})
            return address.get('state_district', address.get('county', 'Unknown'))
            
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return "Unknown"
    
    def _get_soil_type(self, lat: float, lon: float) -> str:
        """Get soil type from coordinates"""
        # Placeholder - would use actual soil database API
        return "Medium Black Soil (typical for Maharashtra)"
    
    def _get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation in meters"""
        try:
            # Use Open-Elevation API
            url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            return data['results'][0]['elevation']
            
        except Exception as e:
            logger.error(f"Elevation fetch error: {e}")
            return 0.0
    
    def _get_agro_zone(self, lat: float, lon: float) -> str:
        """Get agro-climatic zone"""
        # Maharashtra has 9 agro-climatic zones
        # This would map coordinates to zones
        return "Coastal Konkan Zone"  # Placeholder


# Main integration class
class ExternalDataIntegrator:
    """Main class to coordinate all external data sources"""
    
    def __init__(self):
        self.weather_fetcher = IMDWeatherFetcher()
        self.fertilizer_fetcher = FertilizerDataFetcher()
        self.research_fetcher = ResearchPaperFetcher()
        self.location_fetcher = LocationDataFetcher()
        
    def get_complete_context(self, location: str, latitude: float = None, 
                            longitude: float = None) -> Dict:
        """
        Get complete context for prediction
        
        Args:
            location: Location name
            latitude: GPS latitude (optional)
            longitude: GPS longitude (optional)
            
        Returns:
            Dict with all contextual data
        """
        context = {
            'location': location,
            'timestamp': datetime.now().isoformat()
        }
        
        # Fetch weather
        context['current_weather'] = self.weather_fetcher.get_current_weather(location, lat=latitude, lon=longitude)
        context['forecast'] = self.weather_fetcher.get_forecast(location, days=10, lat=latitude, lon=longitude)
        
        # Fetch location profile if coordinates provided
        if latitude and longitude:
            context['location_profile'] = self.location_fetcher.get_location_profile(
                latitude, longitude
            )
        
        return context
    
    def update_knowledge_base(self, disease: str) -> Dict:
        """
        Update knowledge base with latest research
        
        Args:
            disease: Disease name to search for
            
        Returns:
            Dict with updated treatment information
        """
        # Search for latest research papers
        query = f"custard apple {disease} treatment management"
        papers = self.research_fetcher.search_papers(query, max_results=5)
        
        # Extract treatment information
        treatments = []
        for paper in papers:
            treatment_info = self.research_fetcher.extract_treatment_info(
                paper['abstract']
            )
            if treatment_info:
                treatments.append({
                    'source': paper['title'],
                    'year': paper['published'][:4],
                    'info': treatment_info
                })
        
        return {
            'disease': disease,
            'research_papers': papers,
            'extracted_treatments': treatments,
            'updated_at': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the data fetchers
    logging.basicConfig(level=logging.INFO)
    
    integrator = ExternalDataIntegrator()
    
    # Test weather fetch
    print("=" * 70)
    print("TESTING WEATHER FETCH")
    print("=" * 70)
    
    weather = integrator.weather_fetcher.get_current_weather("Thane", "Maharashtra")
    print(json.dumps(weather, indent=2))
    
    # Test location fetch
    print("\n" + "=" * 70)
    print("TESTING LOCATION FETCH")
    print("=" * 70)
    
    # Coordinates for Pune, Maharashtra
    location_profile = integrator.location_fetcher.get_location_profile(18.5204, 73.8567)
    print(json.dumps(location_profile, indent=2))
    
    # Test complete context
    print("\n" + "=" * 70)
    print("TESTING COMPLETE CONTEXT")
    print("=" * 70)
    
    context = integrator.get_complete_context("Pune", 18.5204, 73.8567)
    print(f"Context keys: {list(context.keys())}")
