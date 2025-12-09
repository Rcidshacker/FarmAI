"""
Configuration file for the Custard Apple Pest Management System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'
KNOWLEDGE_BASE_DIR = BASE_DIR / 'knowledge_base'

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, LOGS_DIR, KNOWLEDGE_BASE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset paths
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

CUSTARD_APPLE_DATASET = BASE_DIR / 'Custard Apple dataset'
METADATA_CSV = BASE_DIR / 'custard_apple_metadata.csv'
MEALYBUG_METADATA_CSV = BASE_DIR / 'mealybug_only_metadata.csv'

# Disease classes
DISEASE_CLASSES = [
    'Athracnose',
    'Blank Canker',
    'Diplodia Rot',
    'Leaf spot on fruit',
    'Leaf spot on Leaves',
    'Mealy Bug'
]

# Model parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Weather parameters
WEATHER_FEATURES = [
    'tempmax', 'tempmin', 'temp', 'humidity', 
    'precip', 'precipprob', 'windspeed', 'cloudcover',
    'solarradiation', 'uvindex'
]

# Pest outbreak thresholds (based on literature)
PEST_THRESHOLDS = {
    'Mealy Bug': {
        'temp_range': (25, 35),  # Celsius
        'humidity_min': 60,
        'rainfall_pattern': 'moderate'
    },
    'Athracnose': {
        'temp_range': (20, 30),
        'humidity_min': 80,
        'rainfall_pattern': 'high'
    },
    'Leaf spot on fruit': {
        'temp_range': (22, 32),
        'humidity_min': 70,
        'rainfall_pattern': 'high'
    }
}

# Maharashtra regions
REGIONS = {
    'nimgaon_bhogi': {
        'name': 'Nimgaon-Bhogi',
        'state': 'Maharashtra',
        'lat': 18.85,
        'lon': 74.65
    },
    'thane': {
        'name': 'Thane',
        'state': 'Maharashtra',
        'lat': 19.2183,
        'lon': 72.9781
    }
}

# Chemical recommendations (based on agricultural guidelines)
CHEMICAL_DATABASE = {
    'Mealy Bug': [
        {
            'name': 'Imidacloprid',
            'concentration': '0.3 ml/L',
            'application': 'Spray on affected parts',
            'frequency': 'Every 15 days',
            'precautions': 'Avoid during flowering'
        },
        {
            'name': 'Thiamethoxam',
            'concentration': '0.4 g/L',
            'application': 'Foliar spray',
            'frequency': 'Every 20 days',
            'precautions': 'Use protective equipment'
        },
        {
            'name': 'Neem Oil (Organic)',
            'concentration': '5 ml/L',
            'application': 'Spray thoroughly',
            'frequency': 'Weekly',
            'precautions': 'Safe for organic farming'
        }
    ],
    'Athracnose': [
        {
            'name': 'Mancozeb',
            'concentration': '2.5 g/L',
            'application': 'Foliar spray',
            'frequency': 'Every 10-12 days',
            'precautions': 'Start before symptom appearance'
        },
        {
            'name': 'Copper Oxychloride',
            'concentration': '3 g/L',
            'application': 'Protective spray',
            'frequency': 'Every 15 days',
            'precautions': 'Continue during rainy season'
        }
    ],
    'Diplodia Rot': [
        {
            'name': 'Thiophanate Methyl',
            'concentration': '1 g/L',
            'application': 'Spray on fruits',
            'frequency': 'Every 15 days',
            'precautions': 'Post-harvest treatment recommended'
        }
    ],
    'Leaf spot on fruit': [
        {
            'name': 'Carbendazim',
            'concentration': '1 g/L',
            'application': 'Foliar spray',
            'frequency': 'Every 14 days',
            'precautions': 'Rotate with other fungicides'
        }
    ]
}

# API Configuration
API_HOST = '0.0.0.0'
API_PORT = 8000
API_WORKERS = 4

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Model file names - PyTorch
DISEASE_MODEL_NAME = 'disease_classifier_resnet18.pth'
WEATHER_MODEL_NAME = 'weather_pest_predictor.pkl'
AI_TREATMENT_MODEL_NAME = 'ai_treatment_recommender.pth'
SPRAY_SCHEDULER_MODEL_NAME = 'spray_scheduler.pkl'

# Critical weather conditions for alerts
ALERT_CONDITIONS = {
    'high_rain': {
        'threshold': 50,  # mm
        'diseases': ['Athracnose', 'Leaf spot on fruit']
    },
    'high_humidity': {
        'threshold': 80,  # percentage
        'diseases': ['Athracnose', 'Mealy Bug']
    },
    'temperature_spike': {
        'threshold': 35,  # celsius
        'diseases': ['Mealy Bug']
    }
}

# Training parameters
RANDOM_SEED = 42
USE_GPU = True
EARLY_STOPPING_PATIENCE = 10
CHECKPOINT_MONITOR = 'val_accuracy'

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'brightness_range': [0.8, 1.2]
}
