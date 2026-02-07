import logging
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

logger = logging.getLogger(__name__)

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
