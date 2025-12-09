"""
Weather Data Processor for Pest Management System
Handles weather data loading, cleaning, and feature extraction
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob

from ..utils.logger import setup_logger
from ..utils.helpers import (
    normalize_weather_data, 
    create_time_features,
    get_weather_forecast_features,
    calculate_risk_score
)
from ..utils.config import WEATHER_FEATURES, PEST_THRESHOLDS, BASE_DIR

logger = setup_logger('weather_processor')


class WeatherProcessor:
    """Process and analyze weather data for pest prediction"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize Weather Processor
        
        Args:
            data_dir: Directory containing weather CSV files
        """
        self.data_dir = data_dir or BASE_DIR
        self.weather_data = None
        self.processed_data = None
        
    def load_weather_data(self, region: str = None) -> pd.DataFrame:
        """
        Load weather data from CSV files
        
        Args:
            region: Specific region to load (e.g., 'nimgaon', 'thane')
            
        Returns:
            Combined DataFrame with weather data
        """
        logger.info(f"Loading weather data from {self.data_dir}")
        
        # Find all weather CSV files
        pattern = str(self.data_dir / "*.csv")
        weather_files = glob.glob(pattern)
        
        # Filter by region if specified
        if region:
            weather_files = [f for f in weather_files if region.lower() in f.lower()]
        
        # Exclude metadata files
        weather_files = [f for f in weather_files if 'metadata' not in f.lower()]
        
        logger.info(f"Found {len(weather_files)} weather files")
        
        dataframes = []
        for file in weather_files:
            try:
                df = pd.read_csv(file)
                
                # Extract region name from filename
                region_name = Path(file).stem.split(',')[0] if ',' in Path(file).stem else Path(file).stem
                df['region'] = region_name
                
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} records from {Path(file).name}")
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
        
        if dataframes:
            self.weather_data = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Total weather records: {len(self.weather_data)}")
            return self.weather_data
        else:
            logger.warning("No weather data loaded")
            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Clean and preprocess weather data
        
        Args:
            df: DataFrame to clean (uses self.weather_data if None)
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.weather_data.copy()
        else:
            df = df.copy()
        
        logger.info("Cleaning weather data")
        
        # Normalize temperature units
        df = normalize_weather_data(df)
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Parse datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Create time-based features
        if 'datetime' in df.columns:
            df = create_time_features(df)
        
        logger.info(f"Cleaned data: {len(df)} records")
        self.processed_data = df
        
        return df
    
    def extract_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract relevant weather features for pest prediction
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with extracted features
        """
        if df is None:
            df = self.processed_data.copy()
        else:
            df = df.copy()
        
        logger.info("Extracting weather features")
        
        # Calculate rolling averages (7-day window)
        for feature in WEATHER_FEATURES:
            if feature in df.columns:
                df[f'{feature}_7day_avg'] = df[feature].rolling(window=7, min_periods=1).mean()
                df[f'{feature}_7day_std'] = df[feature].rolling(window=7, min_periods=1).std()
        
        # Calculate temperature difference
        if 'tempmax' in df.columns and 'tempmin' in df.columns:
            df['temp_range'] = df['tempmax'] - df['tempmin']
        
        # Rain intensity
        if 'precip' in df.columns:
            df['rain_intensity'] = pd.cut(df['precip'], 
                                         bins=[-np.inf, 0, 2, 10, np.inf],
                                         labels=['None', 'Light', 'Moderate', 'Heavy'])
        
        # Humidity categories
        if 'humidity' in df.columns:
            df['humidity_category'] = pd.cut(df['humidity'],
                                            bins=[0, 40, 60, 80, 100],
                                            labels=['Low', 'Medium', 'High', 'Very High'])
        
        return df
    
    def calculate_pest_risk(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate pest outbreak risk scores for each day
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with risk scores
        """
        if df is None:
            df = self.processed_data.copy()
        else:
            df = df.copy()
        
        logger.info("Calculating pest risk scores")
        
        # Calculate risk for each pest type
        for pest_type, thresholds in PEST_THRESHOLDS.items():
            risk_scores = []
            
            for idx, row in df.iterrows():
                weather_data = {
                    'temp': row.get('temp', 0),
                    'humidity': row.get('humidity', 0),
                    'precip': row.get('precip', 0)
                }
                
                risk = calculate_risk_score(weather_data, pest_type, thresholds)
                risk_scores.append(risk)
            
            df[f'{pest_type}_risk'] = risk_scores
        
        return df
    
    def get_seasonal_analysis(self, df: pd.DataFrame = None) -> Dict:
        """
        Analyze pest risk by season
        
        Args:
            df: DataFrame with weather and risk data
            
        Returns:
            Dictionary with seasonal analysis
        """
        if df is None:
            df = self.processed_data.copy()
        else:
            df = df.copy()
        
        if 'season' not in df.columns:
            df = create_time_features(df)
        
        logger.info("Performing seasonal analysis")
        
        seasonal_stats = {}
        
        for season in df['season'].unique():
            season_data = df[df['season'] == season]
            
            stats = {
                'avg_temp': season_data['temp'].mean(),
                'avg_humidity': season_data['humidity'].mean(),
                'total_rainfall': season_data['precip'].sum(),
                'record_count': len(season_data)
            }
            
            # Add pest risk averages
            for pest_type in PEST_THRESHOLDS.keys():
                risk_col = f'{pest_type}_risk'
                if risk_col in season_data.columns:
                    stats[f'avg_{pest_type}_risk'] = season_data[risk_col].mean()
            
            seasonal_stats[season] = stats
        
        return seasonal_stats
    
    def get_monthly_trends(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get monthly weather and pest risk trends
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with monthly aggregations
        """
        if df is None:
            df = self.processed_data.copy()
        else:
            df = df.copy()
        
        logger.info("Calculating monthly trends")
        
        # Group by year and month
        monthly = df.groupby(['year', 'month']).agg({
            'temp': 'mean',
            'tempmax': 'max',
            'tempmin': 'min',
            'humidity': 'mean',
            'precip': 'sum',
            'windspeed': 'mean',
            'cloudcover': 'mean'
        }).reset_index()
        
        # Add risk scores if available
        for pest_type in PEST_THRESHOLDS.keys():
            risk_col = f'{pest_type}_risk'
            if risk_col in df.columns:
                pest_risk = df.groupby(['year', 'month'])[risk_col].mean().reset_index()
                monthly = monthly.merge(pest_risk, on=['year', 'month'], how='left')
        
        return monthly
    
    def predict_upcoming_risk(self, days_ahead: int = 7) -> Dict:
        """
        Predict pest risk for upcoming days based on recent trends
        
        Args:
            days_ahead: Number of days to forecast
            
        Returns:
            Dictionary with risk predictions
        """
        if self.processed_data is None or len(self.processed_data) == 0:
            logger.warning("No processed data available for prediction")
            return {}
        
        logger.info(f"Predicting risk for next {days_ahead} days")
        
        # Get recent weather trends
        recent_data = self.processed_data.tail(days_ahead * 2)
        forecast_features = get_weather_forecast_features(recent_data, days_ahead)
        
        # Calculate risk for each pest based on forecasted conditions
        risk_predictions = {}
        
        for pest_type, thresholds in PEST_THRESHOLDS.items():
            weather_data = {
                'temp': forecast_features['avg_temp'],
                'humidity': forecast_features['avg_humidity'],
                'precip': forecast_features['total_rainfall'] / days_ahead
            }
            
            risk = calculate_risk_score(weather_data, pest_type, thresholds)
            
            risk_predictions[pest_type] = {
                'risk_score': risk,
                'risk_level': 'High' if risk > 70 else 'Medium' if risk > 40 else 'Low',
                'forecast_conditions': forecast_features
            }
        
        return risk_predictions
    
    def _aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates hourly weather data to daily summaries to match model training format.
        """
        # Return if empty or likely already daily (unique dates approx equal to row count)
        if df.empty or 'datetime' not in df.columns:
            return df
            
        # Check if aggregation is needed (if we have many rows per day)
        df['datetime'] = pd.to_datetime(df['datetime'])
        if df['datetime'].dt.date.nunique() > (len(df) * 0.9):
            return df

        logger.info("Aggregating hourly weather data to daily summaries...")
        
        # Create a date key for grouping
        df['date_key'] = df['datetime'].dt.date
        
        # Define how to aggregate each column
        # Only aggregate columns that actually exist in the dataframe
        agg_rules = {}
        if 'temp' in df.columns: agg_rules['temp'] = 'mean'
        if 'temperature' in df.columns: agg_rules['temperature'] = 'mean' # Handle alias
        if 'humidity' in df.columns: agg_rules['humidity'] = 'mean'
        if 'precip' in df.columns: agg_rules['precip'] = 'sum'
        if 'rainfall' in df.columns: agg_rules['rainfall'] = 'sum' # Handle alias
        if 'windspeed' in df.columns: agg_rules['windspeed'] = 'mean'
        if 'wind_speed' in df.columns: agg_rules['wind_speed'] = 'mean' # Handle alias
        
        # Calculate Max/Min temp from the same 'temp' column if distinct columns don't exist
        # (This mimics creating tempmax/tempmin from hourly data)
        if 'temp' in df.columns and 'tempmax' not in df.columns:
            agg_rules['temp'] = ['mean', 'max', 'min']
            
        # Perform aggregation
        daily_df = df.groupby('date_key').agg(agg_rules).reset_index()
        
        # Flatten MultiIndex columns if created (e.g., temp -> mean, max, min)
        if isinstance(daily_df.columns, pd.MultiIndex):
            new_cols = []
            for col in daily_df.columns:
                if col[1] == '': new_cols.append(col[0])
                elif col[1] == 'mean': new_cols.append(col[0]) # Keep original name for mean
                elif col[1] == 'max': new_cols.append(f"{col[0]}max") # e.g. tempmax
                elif col[1] == 'min': new_cols.append(f"{col[0]}min") # e.g. tempmin
                else: new_cols.append(f"{col[0]}_{col[1]}")
            daily_df.columns = new_cols

        # Rename date_key back to datetime
        daily_df.rename(columns={'date_key': 'datetime'}, inplace=True)
        
        logger.info(f"Aggregation complete. Reduced to {len(daily_df)} daily records.")
        return daily_df

    def process_pipeline(self, region: str = None) -> pd.DataFrame:
        """
        Complete processing pipeline
        """
        logger.info("Running complete weather processing pipeline")
        
        # 1. Load data
        self.load_weather_data(region)
        
        # --- NEW STEP: Aggregate Hourly to Daily ---
        if self.weather_data is not None and not self.weather_data.empty:
            self.weather_data = self._aggregate_to_daily(self.weather_data)
        # -------------------------------------------
        
        # 2. Clean data
        self.clean_data()
        
        # 3. Extract features
        self.processed_data = self.extract_features()
        
        # 4. Calculate risk scores
        self.processed_data = self.calculate_pest_risk()
        
        logger.info("Weather processing pipeline complete")
        
        return self.processed_data
    
    def save_processed_data(self, output_path: str):
        """Save processed data to CSV"""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        else:
            logger.warning("No processed data to save")


if __name__ == '__main__':
    # Example usage
    processor = WeatherProcessor()
    data = processor.process_pipeline()
    
    print("\n=== Weather Data Summary ===")
    print(data.head())
    print(f"\nTotal records: {len(data)}")
    print(f"\nColumns: {data.columns.tolist()}")
    
    # Seasonal analysis
    seasonal = processor.get_seasonal_analysis()
    print("\n=== Seasonal Analysis ===")
    for season, stats in seasonal.items():
        print(f"\n{season}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Upcoming risk prediction
    predictions = processor.predict_upcoming_risk(7)
    print("\n=== 7-Day Pest Risk Forecast ===")
    for pest, info in predictions.items():
        print(f"\n{pest}:")
        print(f"  Risk Score: {info['risk_score']:.1f}")
        print(f"  Risk Level: {info['risk_level']}")
