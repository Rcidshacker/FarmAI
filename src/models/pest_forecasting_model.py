import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from scipy.signal import savgol_filter
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PestForecastingModel:
    """
    AI Forecasting Model (The "Student").
    Learns from the Biological Engine ("Teacher") to predict risk scores
    based on weather and satellite data.
    """
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'CustardApple_Blind_Model.joblib')
        self.model = None
        self.feature_names = []
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering Pipeline.
        Matches the logic in 03_train_forecasting_model.py
        """
        df = df.copy()
        
        # Ensure datetime
        if 'datetime' in df.columns and not np.issubdtype(df['datetime'].dtype, np.datetime64):
            df['datetime'] = pd.to_datetime(df['datetime'])
            
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
            
        # 1. Winsorization (Clipping) - Only if columns exist
        for col in ['VH', 'VV', 'RVI', 'Radar_Ratio']:
            if col in df.columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)
                
        # 2. Savitzky-Golay Smoothing
        # Requires at least window_length samples
        for col in ['VH', 'VV', 'RVI', 'Radar_Ratio']:
            if col in df.columns:
                if len(df) >= 15:
                    try:
                        df[f'{col}_smooth'] = savgol_filter(df[col], window_length=15, polyorder=2)
                    except Exception as e:
                        logger.warning(f"Smoothing failed for {col}: {e}")
                        df[f'{col}_smooth'] = df[col] # Fallback
                else:
                    df[f'{col}_smooth'] = df[col] # Not enough data

        # 3. Interaction Features
        if 'tempmax' in df.columns and 'humidity' in df.columns:
            df['Bio_Heat_Index'] = df['tempmax'] * df['humidity']
            
        if 'precip' in df.columns:
            df['Rain_Intensity_Index'] = df['precip'] * df['precip']
            
        # 4. Rolling Windows
        # We need to handle cases where we might be predicting on a small window
        # For training, we assume full history. For prediction, we might need to fetch history.
        for window in [14, 30, 60]:
            if 'tempmax' in df.columns:
                df[f'temp_roll_mean_{window}'] = df['tempmax'].rolling(window, min_periods=1).mean()
            if 'precip' in df.columns:
                df[f'rain_roll_sum_{window}'] = df['precip'].rolling(window, min_periods=1).sum()
            if 'humidity' in df.columns:
                df[f'humid_roll_mean_{window}'] = df['humidity'].rolling(window, min_periods=1).mean()
            if 'Bio_Heat_Index' in df.columns:
                df[f'heat_index_roll_{window}'] = df['Bio_Heat_Index'].rolling(window, min_periods=1).mean()
                
        # 5. Lags
        if 'RVI_smooth' in df.columns:
            df['rvi_lag_30'] = df['RVI_smooth'].shift(30)
            
        if 'Radar_Ratio_smooth' in df.columns:
            df['radar_ratio_roll_30'] = df['Radar_Ratio_smooth'].rolling(30, min_periods=1).mean()
            df['VH_roll_30'] = df['VH_smooth'].rolling(30, min_periods=1).mean()
            df['VV_roll_30'] = df['VV_smooth'].rolling(30, min_periods=1).mean()
            df['radar_ratio_lag_14'] = df['Radar_Ratio_smooth'].shift(14)
            
        return df

    def train(self, df: pd.DataFrame, target_col: str = 'mealybug_risk_score'):
        """Train the XGBoost model"""
        logger.info("Starting XGBoost training...")
        
        # Feature Engineering
        df_processed = self._create_features(df)
        
        # Drop rows with NaN created by lags/rolling
        df_clean = df_processed.dropna()
        
        if len(df_clean) < 100:
            logger.warning("Dataset too small for robust training!")
            
        # Define features (exclude target and metadata)
        # Fix: Exclude non-numeric columns that XGBoost cannot handle
        exclude_cols = [
            'datetime', target_col, 'month', 'avg_temp',
            'moonrise', 'moonset', 'sunrise', 'sunset', 'location',
            'conditions', 'description', 'icon', 'stations', 'source'
        ]
        features = [c for c in df_clean.columns if c not in exclude_cols and df_clean[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]
        self.feature_names = features
        
        X = df_clean[features]
        y = df_clean[target_col]
        
        # Train XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )
        
        self.model.fit(X, y)
        
        # Save
        joblib.dump({'model': self.model, 'features': self.feature_names}, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
    def load_model(self) -> bool:
        """Load trained model"""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                
                # Check if it's a dict (our format) or raw model (legacy format)
                if isinstance(data, dict) and 'model' in data:
                    self.model = data['model']
                    self.feature_names = data.get('features', [])
                else:
                    # Assume it's the raw model object (e.g. XGBRegressor)
                    self.model = data
                    logger.info("Loaded raw model object.")
                    
                    # Try to recover feature names from the model
                    if hasattr(self.model, 'feature_names_in_'):
                        self.feature_names = list(self.model.feature_names_in_)
                    elif hasattr(self.model, 'get_booster'):
                        try:
                            self.feature_names = self.model.get_booster().feature_names
                        except:
                            self.feature_names = []
                    
                logger.info(f"Forecasting model loaded successfully. Features: {len(self.feature_names)}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False
        return False
        
    def predict(self, weather_data: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores for future weather data.
        
        Args:
            weather_data: DataFrame with weather forecast. 
                          Must contain history for rolling features if possible,
                          or be appended to historical data.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")
                
        # Feature Engineering
        df_processed = self._create_features(weather_data)
        
        # Ensure all features exist (fill missing with 0 or mean)
        for feat in self.feature_names:
            if feat not in df_processed.columns:
                df_processed[feat] = 0 # Default fallback
                
        X = df_processed[self.feature_names]
        
        return self.model.predict(X)
