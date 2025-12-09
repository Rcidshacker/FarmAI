"""
Data Integration Module
Combines weather data with pest/disease information for correlation analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from ..utils.logger import setup_logger
from ..utils.helpers import parse_date, calculate_season, get_growth_stage
from .weather_processor import WeatherProcessor
from .image_processor import ImageProcessor

logger = setup_logger('data_integration')


class DataIntegrator:
    """Integrate weather and pest/disease data for comprehensive analysis"""
    
    def __init__(self):
        """Initialize Data Integrator"""
        self.weather_processor = WeatherProcessor()
        self.image_processor = ImageProcessor()
        self.integrated_data = None
    
    def align_temporal_data(self,
                           weather_df: pd.DataFrame,
                           image_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Align weather data with image metadata by date
        
        Args:
            weather_df: Weather data DataFrame
            image_metadata: Image metadata DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("Aligning temporal data")
        
        # Ensure datetime columns
        if 'datetime' in weather_df.columns:
            weather_df['date'] = pd.to_datetime(weather_df['datetime']).dt.date
        
        if 'date_taken' in image_metadata.columns:
            image_metadata['date'] = pd.to_datetime(image_metadata['date_taken']).dt.date
        
        # Merge on date
        merged = image_metadata.merge(
            weather_df,
            on='date',
            how='left'
        )
        
        logger.info(f"Aligned {len(merged)} records")
        
        return merged
    
    def create_disease_weather_correlation(self,
                                          weather_df: pd.DataFrame,
                                          image_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Create dataset correlating disease occurrences with weather patterns
        
        Args:
            weather_df: Processed weather data
            image_metadata: Image metadata with disease labels
            
        Returns:
            Correlation dataset
        """
        logger.info("Creating disease-weather correlation dataset")
        
        # Align data
        aligned_df = self.align_temporal_data(weather_df, image_metadata)
        
        # Group by date and disease
        daily_disease_counts = aligned_df.groupby(['date', 'disease_label']).size().reset_index(name='count')
        
        # Merge with weather
        correlation_df = daily_disease_counts.merge(
            weather_df,
            on='date',
            how='left'
        )
        
        # Add temporal features
        if 'datetime' in correlation_df.columns:
            correlation_df['season'] = correlation_df['datetime'].apply(
                lambda x: calculate_season(x) if pd.notna(x) else 'Unknown'
            )
            correlation_df['growth_stage'] = correlation_df['datetime'].apply(
                lambda x: get_growth_stage(x) if pd.notna(x) else 'Unknown'
            )
        
        logger.info(f"Created correlation dataset with {len(correlation_df)} records")
        
        return correlation_df
    
    def calculate_disease_prevalence(self,
                                    image_metadata: pd.DataFrame,
                                    time_period: str = 'month') -> pd.DataFrame:
        """
        Calculate disease prevalence over time
        
        Args:
            image_metadata: Image metadata DataFrame
            time_period: Time period for aggregation ('day', 'week', 'month')
            
        Returns:
            Disease prevalence DataFrame
        """
        logger.info(f"Calculating disease prevalence by {time_period}")
        
        df = image_metadata.copy()
        
        if 'date_taken' in df.columns:
            df['date'] = pd.to_datetime(df['date_taken'])
            
            # Create period column
            if time_period == 'day':
                df['period'] = df['date'].dt.date
            elif time_period == 'week':
                df['period'] = df['date'].dt.to_period('W').astype(str)
            elif time_period == 'month':
                df['period'] = df['date'].dt.to_period('M').astype(str)
            
            # Calculate prevalence
            prevalence = df.groupby(['period', 'disease_label']).size().reset_index(name='count')
            total_per_period = df.groupby('period').size().reset_index(name='total')
            
            prevalence = prevalence.merge(total_per_period, on='period')
            prevalence['prevalence_pct'] = (prevalence['count'] / prevalence['total'] * 100).round(2)
            
            return prevalence
        
        return pd.DataFrame()
    
    def identify_weather_triggers(self,
                                 correlation_df: pd.DataFrame,
                                 disease: str,
                                 lookback_days: int = 7) -> Dict:
        """
        Identify weather conditions that trigger specific diseases
        
        Args:
            correlation_df: Disease-weather correlation data
            disease: Disease name to analyze
            lookback_days: Number of days to look back for weather patterns
            
        Returns:
            Dictionary with trigger conditions
        """
        logger.info(f"Identifying weather triggers for {disease}")
        
        disease_data = correlation_df[correlation_df['disease_label'] == disease]
        
        if len(disease_data) == 0:
            logger.warning(f"No data found for disease: {disease}")
            return {}
        
        # Calculate statistics for conditions when disease is present
        triggers = {
            'disease': disease,
            'sample_size': len(disease_data),
            'conditions': {
                'temperature': {
                    'mean': disease_data['temp'].mean(),
                    'min': disease_data['temp'].min(),
                    'max': disease_data['temp'].max(),
                    'std': disease_data['temp'].std()
                },
                'humidity': {
                    'mean': disease_data['humidity'].mean(),
                    'min': disease_data['humidity'].min(),
                    'max': disease_data['humidity'].max(),
                    'std': disease_data['humidity'].std()
                },
                'rainfall': {
                    'mean': disease_data['precip'].mean(),
                    'total': disease_data['precip'].sum(),
                    'max': disease_data['precip'].max()
                }
            }
        }
        
        # Identify most common season
        if 'season' in disease_data.columns:
            season_counts = disease_data['season'].value_counts()
            triggers['most_common_season'] = season_counts.index[0] if len(season_counts) > 0 else 'Unknown'
        
        # Identify most common growth stage
        if 'growth_stage' in disease_data.columns:
            stage_counts = disease_data['growth_stage'].value_counts()
            triggers['most_common_stage'] = stage_counts.index[0] if len(stage_counts) > 0 else 'Unknown'
        
        return triggers
    
    def create_prediction_dataset(self,
                                 weather_df: pd.DataFrame,
                                 image_metadata: pd.DataFrame,
                                 forecast_days: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create dataset for training prediction models
        
        Args:
            weather_df: Weather data
            image_metadata: Disease occurrence data
            forecast_days: Number of days to forecast ahead
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info("Creating prediction dataset")
        
        # Align data
        aligned_df = self.align_temporal_data(weather_df, image_metadata)
        
        features = []
        labels = []
        
        # Extract features for each disease occurrence
        for idx, row in aligned_df.iterrows():
            if pd.notna(row.get('temp')):
                feature_vector = [
                    row.get('temp', 0),
                    row.get('tempmax', 0),
                    row.get('tempmin', 0),
                    row.get('humidity', 0),
                    row.get('precip', 0),
                    row.get('windspeed', 0),
                    row.get('cloudcover', 0),
                    row.get('solarradiation', 0),
                    row.get('uvindex', 0)
                ]
                
                features.append(feature_vector)
                
                # One-hot encode disease label
                disease_label = row.get('disease_label', '')
                label_vector = [1 if disease_label == cls else 0 for cls in self.image_processor.class_mapping.keys()]
                labels.append(label_vector)
        
        features = np.array(features)
        labels = np.array(labels)
        
        logger.info(f"Created prediction dataset: {features.shape[0]} samples, {features.shape[1]} features")
        
        return features, labels
    
    def generate_report(self,
                       weather_df: pd.DataFrame,
                       image_metadata: pd.DataFrame) -> Dict:
        """
        Generate comprehensive analysis report
        
        Args:
            weather_df: Weather data
            image_metadata: Image metadata
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Generating comprehensive report")
        
        report = {
            'summary': {},
            'disease_triggers': {},
            'seasonal_analysis': {},
            'recommendations': []
        }
        
        # Summary statistics
        report['summary'] = {
            'total_weather_records': len(weather_df),
            'total_disease_images': len(image_metadata),
            'date_range': {
                'start': str(weather_df['datetime'].min()) if 'datetime' in weather_df.columns else 'Unknown',
                'end': str(weather_df['datetime'].max()) if 'datetime' in weather_df.columns else 'Unknown'
            },
            'disease_distribution': image_metadata['disease_label'].value_counts().to_dict() if 'disease_label' in image_metadata.columns else {}
        }
        
        # Create correlation dataset
        correlation_df = self.create_disease_weather_correlation(weather_df, image_metadata)
        
        # Analyze each disease
        for disease in self.image_processor.class_mapping.keys():
            triggers = self.identify_weather_triggers(correlation_df, disease)
            if triggers:
                report['disease_triggers'][disease] = triggers
        
        # Seasonal analysis
        seasonal_data = self.weather_processor.get_seasonal_analysis(weather_df)
        report['seasonal_analysis'] = seasonal_data
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        logger.info("Report generation complete")
        
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Weather-based recommendations
        for disease, triggers in report['disease_triggers'].items():
            conditions = triggers.get('conditions', {})
            
            if conditions:
                temp_mean = conditions.get('temperature', {}).get('mean', 0)
                humidity_mean = conditions.get('humidity', {}).get('mean', 0)
                
                recommendations.append(
                    f"{disease}: High risk when temperature is around {temp_mean:.1f}°C "
                    f"and humidity is around {humidity_mean:.1f}%. Apply preventive measures during these conditions."
                )
        
        # Seasonal recommendations
        for season, stats in report['seasonal_analysis'].items():
            recommendations.append(
                f"{season}: Average temperature {stats.get('avg_temp', 0):.1f}°C, "
                f"humidity {stats.get('avg_humidity', 0):.1f}%. Monitor closely for pest outbreaks."
            )
        
        return recommendations
    
    def save_integrated_data(self, output_path: str):
        """Save integrated data to file"""
        if self.integrated_data is not None:
            self.integrated_data.to_csv(output_path, index=False)
            logger.info(f"Saved integrated data to {output_path}")


if __name__ == '__main__':
    # Example usage
    integrator = DataIntegrator()
    
    # Load data
    weather_data = integrator.weather_processor.process_pipeline()
    image_metadata = integrator.image_processor.load_metadata()
    
    if not weather_data.empty and not image_metadata.empty:
        # Generate report
        report = integrator.generate_report(weather_data, image_metadata)
        
        print("\n=== Integrated Analysis Report ===")
        print(f"\nTotal Weather Records: {report['summary']['total_weather_records']}")
        print(f"Total Disease Images: {report['summary']['total_disease_images']}")
        
        print("\n=== Disease Distribution ===")
        for disease, count in report['summary']['disease_distribution'].items():
            print(f"{disease}: {count}")
        
        print("\n=== Key Recommendations ===")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"{i}. {rec}")
