"""
Helper utilities for the Custard Apple Pest Management System
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit"""
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius"""
    return (fahrenheit - 32) * 5/9

def parse_date(date_str: str) -> datetime:
    """
    Parse various date formats
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        datetime object
    """
    formats = [
        '%Y-%m-%d',
        '%d-%m-%Y',
        '%Y:%m:%d %H:%M:%S',
        '%m-%d-%Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date: {date_str}")

def calculate_season(date: datetime) -> str:
    """
    Calculate agricultural season for Maharashtra
    
    Args:
        date: datetime object
        
    Returns:
        Season name
    """
    month = date.month
    
    if month in [6, 7, 8, 9]:
        return 'Kharif'  # Monsoon season
    elif month in [10, 11, 12, 1]:
        return 'Rabi'  # Winter season
    else:
        return 'Summer'  # Summer season

def get_growth_stage(date: datetime) -> str:
    """
    Determine custard apple growth stage based on month (Maharashtra)
    
    Args:
        date: datetime object
        
    Returns:
        Growth stage
    """
    month = date.month
    
    if month in [6, 7, 8]:
        return 'Flowering'
    elif month in [9, 10, 11]:
        return 'Fruit Development'
    elif month in [12, 1, 2]:
        return 'Fruit Maturity'
    elif month in [3, 4, 5]:
        return 'Harvesting'
    
    return 'Unknown'

def calculate_risk_score(weather_data: Dict, pest_type: str, thresholds: Dict) -> float:
    """
    Calculate pest outbreak risk score based on weather conditions
    
    Args:
        weather_data: Dictionary with weather parameters
        pest_type: Type of pest/disease
        thresholds: Threshold values for the pest
        
    Returns:
        Risk score (0-100)
    """
    risk_score = 0
    
    temp = weather_data.get('temp', 0)
    humidity = weather_data.get('humidity', 0)
    rainfall = weather_data.get('precip', 0)
    
    # Temperature check
    temp_range = thresholds.get('temp_range', (0, 100))
    if temp_range[0] <= temp <= temp_range[1]:
        risk_score += 35
    
    # Humidity check
    humidity_min = thresholds.get('humidity_min', 0)
    if humidity >= humidity_min:
        risk_score += 35
    
    # Rainfall check
    rainfall_pattern = thresholds.get('rainfall_pattern', 'low')
    if rainfall_pattern == 'high' and rainfall > 10:
        risk_score += 30
    elif rainfall_pattern == 'moderate' and 2 < rainfall <= 10:
        risk_score += 30
    elif rainfall_pattern == 'low' and rainfall <= 2:
        risk_score += 30
    
    return min(risk_score, 100)

def get_weather_forecast_features(df: pd.DataFrame, days_ahead: int = 7) -> Dict:
    """
    Extract weather features for forecasting
    
    Args:
        df: DataFrame with weather data
        days_ahead: Number of days to forecast
        
    Returns:
        Dictionary with aggregated weather features
    """
    recent_data = df.tail(days_ahead)
    
    features = {
        'avg_temp': recent_data['temp'].mean(),
        'max_temp': recent_data['tempmax'].max(),
        'min_temp': recent_data['tempmin'].min(),
        'avg_humidity': recent_data['humidity'].mean(),
        'total_rainfall': recent_data['precip'].sum(),
        'avg_windspeed': recent_data['windspeed'].mean(),
        'avg_cloudcover': recent_data['cloudcover'].mean()
    }
    
    return features

def format_treatment_recommendation(disease: str, severity: str, chemicals: List[Dict]) -> Dict:
    """
    Format treatment recommendation for user display
    
    Args:
        disease: Disease name
        severity: Severity level (Low/Medium/High)
        chemicals: List of chemical recommendations
        
    Returns:
        Formatted recommendation dictionary
    """
    recommendation = {
        'disease': disease,
        'severity': severity,
        'urgency': 'High' if severity in ['High', 'Severe'] else 'Medium' if severity == 'Medium' else 'Low',
        'treatments': chemicals,
        'preventive_measures': get_preventive_measures(disease),
        'cultural_practices': get_cultural_practices(disease)
    }
    
    return recommendation

def get_preventive_measures(disease: str) -> List[str]:
    """Get preventive measures for a disease"""
    measures = {
        'Athracnose': [
            'Remove and destroy infected fruits',
            'Avoid overhead irrigation',
            'Ensure proper drainage',
            'Maintain plant spacing for air circulation',
            'Apply fungicides before rainy season'
        ],
        'Mealy Bug': [
            'Regular monitoring of plants',
            'Remove weeds around the orchard',
            'Prune affected branches',
            'Release natural predators (ladybugs)',
            'Apply neem oil preventively'
        ],
        'Diplodia Rot': [
            'Proper post-harvest handling',
            'Avoid fruit injuries during harvesting',
            'Store fruits in cool, dry conditions',
            'Remove infected fruits immediately'
        ],
        'Leaf spot on fruit': [
            'Remove fallen leaves regularly',
            'Improve orchard sanitation',
            'Balanced fertilization',
            'Avoid water stress'
        ]
    }
    
    return measures.get(disease, ['Maintain good agricultural practices'])

def get_cultural_practices(disease: str) -> List[str]:
    """Get cultural practices to prevent disease"""
    practices = [
        'Maintain optimal plant spacing (6m x 6m)',
        'Regular pruning to improve air circulation',
        'Balanced fertilization (NPK based on soil test)',
        'Proper irrigation management',
        'Mulching to conserve moisture',
        'Regular field monitoring'
    ]
    
    return practices

def save_json(data: Any, file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(file_path: str) -> Any:
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_time_features(df: pd.DataFrame, date_column: str = 'datetime') -> pd.DataFrame:
    """
    Create time-based features from datetime column
    
    Args:
        df: Input DataFrame
        date_column: Name of datetime column
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['season'] = df[date_column].apply(calculate_season)
    df['growth_stage'] = df[date_column].apply(get_growth_stage)
    
    return df

def normalize_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize weather data (handle different units)
    
    Args:
        df: DataFrame with weather data
        
    Returns:
        Normalized DataFrame
    """
    df = df.copy()
    
    # Check if temperature is in Fahrenheit (values > 50 likely Fahrenheit)
    if df['temp'].mean() > 50:
        for col in ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike']:
            if col in df.columns:
                df[col] = df[col].apply(fahrenheit_to_celsius)
    
    return df
