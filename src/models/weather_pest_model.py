"""
Weather-Pest Correlation Model
ML model to predict pest outbreak probability based on weather conditions
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json

from ..utils.logger import setup_logger
from ..utils.config import MODEL_DIR, WEATHER_MODEL_NAME, PEST_THRESHOLDS
from ..data_processing.weather_processor import WeatherProcessor

logger = setup_logger('weather_pest_model')


class WeatherPestPredictor:
    """
    Predict pest outbreak probability based on weather patterns
    Uses ensemble ML models (Random Forest + Gradient Boosting)
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize Weather-Pest Predictor
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.pest_types = list(PEST_THRESHOLDS.keys())
        self.model_path = model_path or MODEL_DIR / WEATHER_MODEL_NAME
        self.trained = False
        
    def prepare_features(self, weather_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features from weather data
        
        Args:
            weather_df: DataFrame with weather data
            
        Returns:
            Tuple of (feature array, feature names)
        """
        logger.info("Preparing features from weather data")
        
        feature_cols = [
            'temp', 'tempmax', 'tempmin', 'humidity', 'precip',
            'windspeed', 'cloudcover', 'solarradiation', 'uvindex'
        ]
        
        # Add derived features
        if 'tempmax' in weather_df.columns and 'tempmin' in weather_df.columns:
            weather_df['temp_range'] = weather_df['tempmax'] - weather_df['tempmin']
            feature_cols.append('temp_range')
        
        # Rolling averages (if enough data)
        if len(weather_df) > 7:
            for col in ['temp', 'humidity', 'precip']:
                if col in weather_df.columns:
                    weather_df[f'{col}_7day_avg'] = weather_df[col].rolling(window=7, min_periods=1).mean()
                    feature_cols.append(f'{col}_7day_avg')
        
        # Seasonal encoding
        if 'month' in weather_df.columns:
            weather_df['season_sin'] = np.sin(2 * np.pi * weather_df['month'] / 12)
            weather_df['season_cos'] = np.cos(2 * np.pi * weather_df['month'] / 12)
            feature_cols.extend(['season_sin', 'season_cos'])
        
        # Select available features
        available_features = [col for col in feature_cols if col in weather_df.columns]
        
        # Fill missing values
        features = weather_df[available_features].fillna(method='ffill').fillna(0)
        
        logger.info(f"Prepared {len(available_features)} features")
        self.feature_names = available_features
        
        return features.values, available_features
    
    def create_labels(self, weather_df: pd.DataFrame, pest_type: str) -> np.ndarray:
        """
        Create labels for pest outbreak (binary or multi-class)
        
        Args:
            weather_df: DataFrame with weather data and risk scores
            pest_type: Type of pest to create labels for
            
        Returns:
            Label array
        """
        risk_col = f'{pest_type}_risk'
        
        if risk_col not in weather_df.columns:
            logger.warning(f"Risk column {risk_col} not found, creating based on thresholds")
            # Calculate risk if not present
            processor = WeatherProcessor()
            weather_df = processor.calculate_pest_risk(weather_df)
        
        # Create binary labels (High risk vs Low risk)
        # Threshold: > 70 = High risk (1), <= 70 = Low risk (0)
        labels = (weather_df[risk_col] > 70).astype(int)
        
        logger.info(f"Created labels for {pest_type}: High risk={sum(labels)}, Low risk={len(labels)-sum(labels)}")
        
        return labels.values
    
    def train_model(self, X: np.ndarray, y: np.ndarray, pest_type: str) -> Dict:
        """
        Train Random Forest model for pest prediction
        
        Args:
            X: Feature array
            y: Label array
            pest_type: Type of pest being modeled
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training model for {pest_type}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluation metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'pest_type': pest_type,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, rf_model.feature_importances_))
        metrics['feature_importance'] = dict(sorted(feature_importance.items(), 
                                                    key=lambda x: x[1], 
                                                    reverse=True)[:10])
        
        logger.info(f"Model trained - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
        
        self.model = rf_model
        self.trained = True
        
        return metrics
    
    def train_all_pests(self, weather_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Train models for all pest types
        
        Args:
            weather_df: DataFrame with weather data and risk scores
            
        Returns:
            Dictionary with metrics for each pest type
        """
        logger.info("Training models for all pest types")
        
        all_metrics = {}
        
        # Prepare features once
        X, feature_names = self.prepare_features(weather_df)
        
        # Train model for each pest
        for pest_type in self.pest_types:
            try:
                y = self.create_labels(weather_df, pest_type)
                
                # Check if we have enough positive samples
                if sum(y) < 5:
                    logger.warning(f"Not enough positive samples for {pest_type}, skipping")
                    continue
                
                metrics = self.train_model(X, y, pest_type)
                all_metrics[pest_type] = metrics
                
            except Exception as e:
                logger.error(f"Error training model for {pest_type}: {str(e)}")
                continue
        
        # Save comprehensive results
        self._save_training_results(all_metrics)
        
        return all_metrics
    
    def _save_training_results(self, all_metrics: Dict[str, Dict]):
        """
        Save comprehensive training results
        
        Args:
            all_metrics: Dictionary with metrics for all pests
        """
        from pathlib import Path
        from datetime import datetime
        import matplotlib.pyplot as plt
        
        # Create results directory
        results_dir = Path("training_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to JSON
        metrics_file = results_dir / f"weather_pest_predictor_metrics_{timestamp}.json"
        
        save_metrics = {
            'timestamp': timestamp,
            'model_type': 'Random Forest',
            'device': 'CPU',
            'n_estimators': 100,
            'pest_metrics': {}
        }
        
        for pest_type, metrics in all_metrics.items():
            save_metrics['pest_metrics'][pest_type] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'roc_auc': float(metrics['roc_auc']),
                'n_samples': int(metrics['n_samples']),
                'n_features': int(metrics['n_features']),
                'top_5_features': dict(list(metrics['feature_importance'].items())[:5])
            }
        
        # Calculate overall statistics
        accuracies = [m['accuracy'] for m in all_metrics.values()]
        f1_scores = [m['f1_score'] for m in all_metrics.values()]
        
        save_metrics['overall'] = {
            'avg_accuracy': float(np.mean(accuracies)),
            'avg_f1_score': float(np.mean(f1_scores)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            'n_models_trained': len(all_metrics)
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(save_metrics, f, indent=4)
        
        logger.info(f"Training metrics saved to {metrics_file}")
        
        # Create visualization
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            pests = list(all_metrics.keys())
            
            # 1. Accuracy comparison
            accuracies = [all_metrics[p]['accuracy'] for p in pests]
            colors = ['green' if a > 0.8 else 'orange' if a > 0.6 else 'red' for a in accuracies]
            ax1.barh(pests, accuracies, color=colors, alpha=0.7)
            ax1.set_xlabel('Accuracy')
            ax1.set_title('Model Accuracy by Pest Type', fontweight='bold')
            ax1.set_xlim([0, 1])
            ax1.grid(True, alpha=0.3, axis='x')
            
            # 2. Multiple metrics comparison
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            x = np.arange(len(pests))
            width = 0.2
            
            for i, metric_name in enumerate(metrics_names):
                metric_key = metric_name.lower().replace('-', '_')
                values = [all_metrics[p][metric_key] for p in pests]
                ax2.bar(x + i*width, values, width, label=metric_name, alpha=0.8)
            
            ax2.set_ylabel('Score')
            ax2.set_title('Performance Metrics Comparison', fontweight='bold')
            ax2.set_xticks(x + width * 1.5)
            ax2.set_xticklabels(pests, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 3. Feature importance (top pest)
            best_pest = max(all_metrics.keys(), key=lambda p: all_metrics[p]['f1_score'])
            top_features = all_metrics[best_pest]['feature_importance']
            features = list(top_features.keys())[:8]
            importances = [top_features[f] for f in features]
            
            ax3.barh(features, importances, color='steelblue', alpha=0.7)
            ax3.set_xlabel('Importance')
            ax3.set_title(f'Top Features - {best_pest}', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # 4. ROC-AUC scores
            roc_aucs = [all_metrics[p]['roc_auc'] for p in pests]
            colors = ['darkgreen' if a > 0.8 else 'gold' if a > 0.7 else 'coral' for a in roc_aucs]
            ax4.barh(pests, roc_aucs, color=colors, alpha=0.7)
            ax4.set_xlabel('ROC-AUC Score')
            ax4.set_title('ROC-AUC by Pest Type', fontweight='bold')
            ax4.set_xlim([0, 1])
            ax4.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            plot_file = results_dir / f"weather_pest_predictor_training_{timestamp}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to {plot_file}")
            
        except Exception as e:
            logger.warning(f"Could not create training plots: {e}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY - Weather Pest Predictor (Random Forest)")
        logger.info("=" * 60)
        logger.info(f"Models Trained: {len(all_metrics)}")
        logger.info(f"Average Accuracy: {save_metrics['overall']['avg_accuracy']:.4f}")
        logger.info(f"Average F1-Score: {save_metrics['overall']['avg_f1_score']:.4f}")
        logger.info("")
        logger.info("Individual Model Performance:")
        for pest, metrics in all_metrics.items():
            logger.info(f"  {pest}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}, AUC={metrics['roc_auc']:.3f}")
        logger.info("=" * 60)
    
    def predict_risk(self, weather_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Predict pest outbreak risk for given weather conditions
        
        Args:
            weather_data: DataFrame with weather features
            
        Returns:
            Dictionary with predictions for each pest type
        """
        if not self.trained:
            logger.error("Model not trained. Train model before prediction.")
            return {}
        
        logger.info("Predicting pest outbreak risk")
        
        # Prepare features
        X, _ = self.prepare_features(weather_data)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = {}
        
        try:
            risk_proba = self.model.predict_proba(X_scaled)[:, 1]
            risk_class = self.model.predict(X_scaled)
            
            for i, pest_type in enumerate(self.pest_types):
                predictions[pest_type] = {
                    'risk_probability': float(risk_proba[i]) if i < len(risk_proba) else 0.0,
                    'risk_level': 'High' if risk_class[i] == 1 else 'Low',
                    'risk_score': float(risk_proba[i] * 100) if i < len(risk_proba) else 0.0
                }
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
        
        return predictions
    
    def predict_from_conditions(self, 
                                temp: float,
                                humidity: float,
                                rainfall: float,
                                **kwargs) -> Dict[str, Dict]:
        """
        Predict pest risk from individual weather parameters
        
        Args:
            temp: Temperature (Celsius)
            humidity: Humidity (%)
            rainfall: Rainfall (mm)
            **kwargs: Additional weather parameters
            
        Returns:
            Prediction dictionary
        """
        # Create DataFrame from parameters
        weather_data = pd.DataFrame([{
            'temp': temp,
            'tempmax': kwargs.get('tempmax', temp + 5),
            'tempmin': kwargs.get('tempmin', temp - 5),
            'humidity': humidity,
            'precip': rainfall,
            'windspeed': kwargs.get('windspeed', 10),
            'cloudcover': kwargs.get('cloudcover', 50),
            'solarradiation': kwargs.get('solarradiation', 200),
            'uvindex': kwargs.get('uvindex', 5),
            'month': kwargs.get('month', 7)
        }])
        
        return self.predict_risk(weather_data)
    
    def save_model(self, path: Optional[Path] = None):
        """Save trained model to file"""
        if not self.trained:
            logger.warning("No trained model to save")
            return
        
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'pest_types': self.pest_types
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: Optional[Path] = None):
        """Load trained model from file"""
        load_path = path or self.model_path
        
        if not load_path.exists():
            logger.error(f"Model file not found: {load_path}")
            return False
        
        try:
            model_data = joblib.load(load_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.pest_types = model_data['pest_types']
            self.trained = True
            
            logger.info(f"Model loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.trained or self.model is None:
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    # Example usage
    from ..data_processing import WeatherProcessor
    
    # Load and process weather data
    processor = WeatherProcessor()
    weather_data = processor.process_pipeline()
    
    if not weather_data.empty:
        # Train predictor
        predictor = WeatherPestPredictor()
        metrics = predictor.train_all_pests(weather_data)
        
        print("\n=== Training Results ===")
        for pest, metric in metrics.items():
            print(f"\n{pest}:")
            print(f"  Accuracy: {metric['accuracy']:.3f}")
            print(f"  Precision: {metric['precision']:.3f}")
            print(f"  Recall: {metric['recall']:.3f}")
            print(f"  F1 Score: {metric['f1_score']:.3f}")
        
        # Save model
        predictor.save_model()
        
        # Test prediction
        test_prediction = predictor.predict_from_conditions(
            temp=28, humidity=75, rainfall=5
        )
        
        print("\n=== Sample Prediction ===")
        print("Conditions: 28°C, 75% humidity, 5mm rainfall")
        for pest, pred in test_prediction.items():
            print(f"\n{pest}:")
            print(f"  Risk Level: {pred['risk_level']}")
            print(f"  Risk Score: {pred['risk_score']:.1f}%")
