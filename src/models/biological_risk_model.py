import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BiologicalRiskModel:
    """
    Biological Engine for Mealybug Risk Assessment.
    Based on GDD (Growing Degree Days), Rain Washout, and Ant-Soil Symbiosis.
    Matches logic from Ruchit's 'Teacher' model.
    """
    
    def __init__(self, 
                 biofix_start_month: int = 6, 
                 biofix_end_month: int = 8, 
                 biofix_min_rain: float = 10.0,
                 dd_per_generation: float = 350.0):
        self.biofix_start_month = biofix_start_month
        self.biofix_end_month = biofix_end_month
        self.biofix_min_rain = biofix_min_rain
        self.dd_per_generation = dd_per_generation
        
    def calculate_risk_series(self, weather_df: pd.DataFrame, clay_pct: float = 30.0) -> pd.DataFrame:
        """
        Calculate risk score series for a dataframe of weather data.
        
        Args:
            weather_df: DataFrame with columns ['datetime', 'tempmax', 'tempmin', 'precip', 'humidity']
            clay_pct: Soil clay percentage (default 30.0)
            
        Returns:
            DataFrame with added 'mealybug_risk_score' column
        """
        df = weather_df.copy()
        
        # Ensure datetime
        if not np.issubdtype(df['datetime'].dtype, np.datetime64):
            df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
            
        df['month'] = df['datetime'].dt.month
        # If tempmax/min not present, try to infer from 'temp'
        if 'tempmax' not in df.columns and 'temp' in df.columns:
             df['tempmax'] = df['temp'] + 5
             df['tempmin'] = df['temp'] - 5
             
        df['avg_temp'] = (df['tempmax'] + df['tempmin']) / 2
        
        risk_scores = []
        accumulated_dd = 0.0
        season_active = False
        
        # Iterate through rows to simulate time progression
        for i, row in df.iterrows():
            # A. Biofix (Season Start)
            # Reset if before season (Jan/Feb)
            if row['month'] <= 2: 
                season_active = False
                accumulated_dd = 0.0
                
            # Check for season start trigger
            if not season_active and (self.biofix_start_month <= row['month'] <= self.biofix_end_month):
                if row['precip'] >= self.biofix_min_rain:
                    season_active = True
                    accumulated_dd = 35.0 # Initial population kickstart

            current_risk = 0.0
            if season_active:
                # B. Degree Days Calculation
                # Mealybugs develop when temp > 15C. Cap at 35C.
                daily_dd = max(0, min(row['avg_temp'], 35) - 15)
                accumulated_dd += daily_dd
                
                # Rain Washout: Heavy rain washes away nymphs
                if row['precip'] > 80: 
                    accumulated_dd *= 0.5 
                
                # Base Risk Score (0.0 to 1.0)
                # Normalized by generations (3.5 generations max per season usually)
                base_score = min(accumulated_dd / (self.dd_per_generation * 3.5), 1.0)
                
                # Peak Season Boost (Sept-Nov)
                if row['month'] in [9, 10, 11]: 
                    base_score = min(base_score * 1.5, 1.0)
                
                current_risk = base_score
                
                # --- C. ANT & SOIL LOGIC ---
                # Ants protect mealybugs. They thrive in dry, warm weather.
                ant_weather_good = (20 <= row['tempmax'] <= 35) and (row['humidity'] < 80) and (row['precip'] < 1)
                
                if ant_weather_good:
                    # Soil Penalty: Ants struggle in Heavy Clay (>35%)
                    # If Clay is high, the "Ant Boost" is dampened (0.9x).
                    # If Soil is sandy/loam, the boost is full (1.2x).
                    soil_factor = 0.9 if clay_pct > 35 else 1.2
                    
                    # Apply the boost/penalty
                    current_risk = min(current_risk * soil_factor, 1.0)
                    
            risk_scores.append(current_risk)
            
        df['mealybug_risk_score'] = risk_scores
        return df
        return df

    def calculate_single_day_risk(self, 
                                temp_max: float, 
                                temp_min: float, 
                                precip: float, 
                                humidity: float, 
                                current_accumulated_dd: float,
                                season_active: bool,
                                month: int,
                                clay_pct: float = 30.0) -> Dict:
        """
        Calculate risk for a single day (for real-time forecasting).
        """
        avg_temp = (temp_max + temp_min) / 2
        new_accumulated_dd = current_accumulated_dd
        new_season_active = season_active
        
        # Biofix Check
        if month <= 2:
            new_season_active = False
            new_accumulated_dd = 0.0
        
        if not new_season_active and (self.biofix_start_month <= month <= self.biofix_end_month):
            if precip >= self.biofix_min_rain:
                new_season_active = True
                new_accumulated_dd = 35.0
        
        risk_score = 0.0
        
        if new_season_active:
            daily_dd = max(0, min(avg_temp, 35) - 15)
            new_accumulated_dd += daily_dd
            
            if precip > 80:
                new_accumulated_dd *= 0.5
                
            base_score = min(new_accumulated_dd / (self.dd_per_generation * 3.5), 1.0)
            
            if month in [9, 10, 11]:
                base_score = min(base_score * 1.5, 1.0)
                
            risk_score = base_score
            
            ant_weather_good = (20 <= temp_max <= 35) and (humidity < 80) and (precip < 1)
            if ant_weather_good:
                soil_factor = 0.9 if clay_pct > 35 else 1.2
                risk_score = min(risk_score * soil_factor, 1.0)
                
        return {
            'risk_score': risk_score,
            'accumulated_dd': new_accumulated_dd,
            'season_active': new_season_active
        }
