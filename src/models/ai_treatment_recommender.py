"""
AI-Powered Treatment Recommendation System
Replaces rule-based knowledge base with ML models that learn from:
- Historical treatment outcomes
- Weather patterns
- Disease severity
- Location-specific factors
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TreatmentDataset(Dataset):
    """PyTorch Dataset for treatment recommendation"""
    
    def __init__(self, features, treatments, outcomes):
        self.features = torch.FloatTensor(features)
        self.treatments = torch.LongTensor(treatments)
        self.outcomes = torch.FloatTensor(outcomes)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.treatments[idx], self.outcomes[idx]


class TreatmentRecommenderNN(nn.Module):
    """
    Neural Network for Treatment Recommendation
    Architecture: Multi-task learning
    - Task 1: Predict best treatment (classification)
    - Task 2: Predict treatment effectiveness (regression)
    """
    
    def __init__(self, input_dim: int, num_treatments: int, hidden_dims: List[int] = [256, 128, 64]):
        super(TreatmentRecommenderNN, self).__init__()
        
        # Shared layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Treatment classification head
        self.treatment_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_treatments)
        )
        
        # Effectiveness prediction head
        self.effectiveness_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0-1
        )
    
    def forward(self, x):
        shared = self.shared_layers(x)
        treatment_logits = self.treatment_head(shared)
        effectiveness = self.effectiveness_head(shared)
        return treatment_logits, effectiveness


class AITreatmentRecommender:
    """
    AI-powered treatment recommendation system
    Learns from historical treatment outcomes to make predictions
    """
    
    def __init__(self, model_path: str = "models/ai_recommender.pth"):
        self.model_path = model_path
        self.model = None
        self.treatment_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.disease_encoder = LabelEncoder()
        self.feature_names = []
        
        # Disease-specific treatment history
        self.treatment_history = []
        
    def prepare_features(self, disease: str, confidence: float, weather: Dict, 
                        location_profile: Dict = None, growth_stage: str = None) -> np.ndarray:
        """
        Prepare feature vector for prediction
        
        Args:
            disease: Detected disease name
            confidence: Detection confidence (0-1)
            weather: Dict with temp, humidity, rainfall, etc.
            location_profile: Dict with soil type, elevation, etc.
            growth_stage: Current growth stage of plant
            
        Returns:
            Feature array
        """
        features = []
        
        # Disease features (one-hot encoded)
        disease_code = self.disease_encoder.transform([disease])[0] if hasattr(self.disease_encoder, 'classes_') else 0
        features.append(disease_code)
        features.append(confidence)
        
        # Weather features
        features.extend([
            weather.get('temp', 25),
            weather.get('humidity', 70),
            weather.get('precip', 0),
            weather.get('wind_speed', 5),
            weather.get('risk_score', 50)
        ])
        
        # Calculate rolling averages if available
        features.extend([
            weather.get('temp_7day_avg', weather.get('temp', 25)),
            weather.get('humidity_7day_avg', weather.get('humidity', 70)),
            weather.get('precip_7day_sum', weather.get('precip', 0))
        ])
        
        # Seasonal features
        from datetime import datetime
        now = datetime.now()
        features.extend([
            now.month,
            now.day,
            np.sin(2 * np.pi * now.month / 12),  # Circular encoding
            np.cos(2 * np.pi * now.month / 12)
        ])
        
        # Location features (if available)
        if location_profile:
            # ---------------------------------------------------------
            # FIX APPLIED HERE: Deterministic Encoding
            # Replaced hash() with fixed dictionaries to ensure model stability.
            # ---------------------------------------------------------
            soil_map = {
                'Black Soil': 0.1, 'Red Soil': 0.2, 'Laterite Soil': 0.3, 
                'Alluvial Soil': 0.4, 'Medium Black Soil': 0.5, 'Clay': 0.6, 
                'Loam': 0.7, 'Sandy': 0.8
            }
            # Add Maharashtra agro-climatic zones
            zone_map = {
                'Coastal Konkan': 0.1, 'Vidarbha': 0.2, 'Marathwada': 0.3, 
                'Western Maharashtra': 0.4, 'North Maharashtra': 0.5
            }

            soil_val = soil_map.get(location_profile.get('soil_type', ''), 0.0)
            zone_val = zone_map.get(location_profile.get('agro_climatic_zone', ''), 0.0)

            features.extend([
                location_profile.get('elevation', 0) / 1000,  # Elevation
                soil_val,                                     # Soil Type (Stable)
                zone_val                                      # Agro Zone (Stable)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Growth stage encoding
        growth_stages = ['Vegetative', 'Flowering', 'Fruit Development', 'Maturity']
        stage_code = growth_stages.index(growth_stage) if growth_stage in growth_stages else 0
        features.append(stage_code / len(growth_stages))
        
        return np.array(features, dtype=np.float32)
    
    def train_from_history(self, history_df: pd.DataFrame, epochs: int = 100, 
                          batch_size: int = 32, lr: float = 0.001):
        """
        Train the model from historical treatment data
        
        Args:
            history_df: DataFrame with columns:
                - disease, confidence, temp, humidity, rainfall, wind_speed,
                  treatment_applied, effectiveness, location, soil_type, etc.
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        # Check for GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        logger.info("Preparing training data...")
        
        # Encode diseases and treatments
        self.disease_encoder.fit(history_df['disease'].unique())
        self.treatment_encoder.fit(history_df['treatment_applied'].unique())
        
        # Prepare features
        features_list = []
        for idx, row in history_df.iterrows():
            weather = {
                'temp': row.get('temp', 25),
                'humidity': row.get('humidity', 70),
                'precip': row.get('rainfall', 0),
                'wind_speed': row.get('wind_speed', 5)
            }
            
            location_profile = {
                'elevation': row.get('elevation', 0),
                'soil_type': row.get('soil_type', ''),
                'agro_climatic_zone': row.get('agro_zone', '')
            }
            
            feat = self.prepare_features(
                row['disease'],
                row.get('confidence', 0.8),
                weather,
                location_profile,
                row.get('growth_stage', 'Vegetative')
            )
            features_list.append(feat)
        
        X = np.array(features_list)
        y_treatment = self.treatment_encoder.transform(history_df['treatment_applied'])
        y_effectiveness = history_df['effectiveness'].values
        
        # Scale features
        X = self.feature_scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_treat_train, y_treat_val, y_eff_train, y_eff_val = train_test_split(
            X, y_treatment, y_effectiveness, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = TreatmentDataset(X_train, y_treat_train, y_eff_train)
        val_dataset = TreatmentDataset(X_val, y_treat_val, y_eff_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_dim = X.shape[1]
        num_treatments = len(self.treatment_encoder.classes_)
        
        self.model = TreatmentRecommenderNN(input_dim, num_treatments)
        self.model = self.model.to(device)  # Move model to GPU
        
        # Loss functions and optimizer
        criterion_treatment = nn.CrossEntropyLoss()
        criterion_effectiveness = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'device': str(device)
        }
        
        # Training loop
        logger.info(f"Training for {epochs} epochs on {device}...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for features, treatments, effectiveness in train_loader:
                # Move data to GPU
                features = features.to(device)
                treatments = treatments.to(device)
                effectiveness = effectiveness.to(device)
                
                optimizer.zero_grad()
                
                treat_logits, eff_pred = self.model(features)
                
                loss_treat = criterion_treatment(treat_logits, treatments)
                loss_eff = criterion_effectiveness(eff_pred.squeeze(), effectiveness)
                
                # Combined loss
                loss = loss_treat + 0.5 * loss_eff
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, treatments, effectiveness in val_loader:
                    # Move validation data to GPU
                    features = features.to(device)
                    treatments = treatments.to(device)
                    effectiveness = effectiveness.to(device)
                    
                    treat_logits, eff_pred = self.model(features)
                    
                    loss_treat = criterion_treatment(treat_logits, treatments)
                    loss_eff = criterion_effectiveness(eff_pred.squeeze(), effectiveness)
                    loss = loss_treat + 0.5 * loss_eff
                    
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(treat_logits, 1)
                    total += treatments.size(0)
                    correct += (predicted == treatments).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(accuracy)
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
                logger.info(f"Best model saved at epoch {epoch+1} with val loss: {best_val_loss:.4f}")
        
        # Save training results
        self._save_training_results(history)
        logger.info("Training completed!")
    
    def predict_treatment(self, disease: str, confidence: float, weather: Dict,
                         location_profile: Dict = None, growth_stage: str = None,
                         top_k: int = 3) -> List[Dict]:
        """
        Predict best treatments using the trained model
        
        Args:
            disease: Detected disease
            confidence: Detection confidence
            weather: Weather conditions
            location_profile: Location data
            growth_stage: Growth stage
            top_k: Number of top treatments to return
            
        Returns:
            List of dicts with treatment, probability, predicted_effectiveness
        """
        if self.model is None:
            logger.warning("Model not trained, loading from disk...")
            self.load_model()
        
        # Prepare features
        features = self.prepare_features(disease, confidence, weather, 
                                        location_profile, growth_stage)
        features = self.feature_scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            treat_logits, eff_pred = self.model(features_tensor)
            
            # Get top-k treatments
            probabilities = torch.softmax(treat_logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
            
            recommendations = []
            for prob, idx in zip(top_probs, top_indices):
                treatment_name = self.treatment_encoder.inverse_transform([idx.item()])[0]
                
                recommendations.append({
                    'treatment': treatment_name,
                    'probability': prob.item(),
                    'predicted_effectiveness': eff_pred[0].item(),
                    'confidence_level': 'High' if prob.item() > 0.6 else 'Medium' if prob.item() > 0.3 else 'Low'
                })
        
        return recommendations
    
    def update_with_feedback(self, disease: str, confidence: float, weather: Dict,
                            treatment_applied: str, actual_effectiveness: float,
                            location_profile: Dict = None, growth_stage: str = None):
        """
        Update model with real-world feedback (online learning)
        
        Args:
            disease: Disease treated
            confidence: Detection confidence
            weather: Weather conditions during treatment
            treatment_applied: Treatment that was applied
            actual_effectiveness: Actual effectiveness observed (0-1)
            location_profile: Location data
            growth_stage: Growth stage
        """
        # Add to history
        feedback = {
            'disease': disease,
            'confidence': confidence,
            'temp': weather.get('temp', 25),
            'humidity': weather.get('humidity', 70),
            'rainfall': weather.get('precip', 0),
            'wind_speed': weather.get('wind_speed', 5),
            'treatment_applied': treatment_applied,
            'effectiveness': actual_effectiveness,
            'timestamp': pd.Timestamp.now()
        }
        
        if location_profile:
            feedback.update(location_profile)
        if growth_stage:
            feedback['growth_stage'] = growth_stage
        
        self.treatment_history.append(feedback)
        
        # Retrain periodically (e.g., every 100 feedbacks)
        if len(self.treatment_history) % 100 == 0:
            logger.info("Retraining model with updated feedback...")
            history_df = pd.DataFrame(self.treatment_history)
            self.train_from_history(history_df, epochs=20)
    
    def _save_training_results(self, history: Dict):
        """
        Save training results including metrics and plots
        
        Args:
            history: Training history with losses and accuracies
        """
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        # Create results directory
        results_dir = Path("training_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to JSON
        metrics_file = results_dir / f"ai_recommender_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            # Convert lists to JSON serializable format
            save_history = {
                'train_loss': [float(x) for x in history['train_loss']],
                'val_loss': [float(x) for x in history['val_loss']],
                'val_accuracy': [float(x) for x in history['val_accuracy']],
                'epochs': history['epochs'],
                'batch_size': history['batch_size'],
                'learning_rate': history['learning_rate'],
                'device': history['device'],
                'best_val_loss': float(min(history['val_loss'])),
                'best_val_accuracy': float(max(history['val_accuracy'])),
                'final_train_loss': float(history['train_loss'][-1]),
                'final_val_loss': float(history['val_loss'][-1]),
                'final_val_accuracy': float(history['val_accuracy'][-1])
            }
            json.dump(save_history, f, indent=4)
        
        logger.info(f"Training metrics saved to {metrics_file}")
        
        # Plot training curves
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss curves
            epochs_range = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Accuracy curve
            ax2.plot(epochs_range, history['val_accuracy'], 'g-', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_file = results_dir / f"ai_recommender_training_{timestamp}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to {plot_file}")
            
        except Exception as e:
            logger.warning(f"Could not create training plots: {e}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Device: {history['device']}")
        logger.info(f"Epochs: {history['epochs']}")
        logger.info(f"Batch Size: {history['batch_size']}")
        logger.info(f"Learning Rate: {history['learning_rate']}")
        logger.info(f"Best Val Loss: {min(history['val_loss']):.4f}")
        logger.info(f"Best Val Accuracy: {max(history['val_accuracy']):.2f}%")
        logger.info(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"Final Val Accuracy: {history['val_accuracy'][-1]:.2f}%")
        logger.info("=" * 60)
    
    def save_model(self):
        """Save model and preprocessing objects"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'treatment_encoder': self.treatment_encoder,
            'disease_encoder': self.disease_encoder,
            'feature_scaler': self.feature_scaler,
            'treatment_history': self.treatment_history
        }, self.model_path)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model and preprocessing objects"""
        try:
            checkpoint = torch.load(self.model_path)
            
            self.treatment_encoder = checkpoint['treatment_encoder']
            self.disease_encoder = checkpoint['disease_encoder']
            self.feature_scaler = checkpoint['feature_scaler']
            self.treatment_history = checkpoint.get('treatment_history', [])
            
            # Recreate model architecture
            input_dim = self.feature_scaler.n_features_in_
            num_treatments = len(self.treatment_encoder.classes_)
            self.model = TreatmentRecommenderNN(input_dim, num_treatments)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"Model loaded from {self.model_path}")
            
        except FileNotFoundError:
            logger.warning(f"Model file not found: {self.model_path}")
            logger.warning("Please train the model first using train_from_history()")


def generate_synthetic_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic training data for initial model training
    In production, this would be replaced with real historical data
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic treatment history
    """
    np.random.seed(42)
    
    diseases = ['Athracnose', 'Mealy Bug', 'Diplodia Rot', 'Leaf spot on fruit', 
                'Leaf spot on Leaves', 'Blank Canker', 'Scale Insects', 'White Flies']
    
    treatments = {
        'Athracnose': ['Mancozeb 75% WP', 'Copper Oxychloride 50% WP', 'Carbendazim 50% WP'],
        'Mealy Bug': ['Imidacloprid 17.8% SL', 'Thiamethoxam 25% WG', 'Neem Oil 1500 ppm'],
        'Diplodia Rot': ['Bordeaux Mixture', 'Copper Oxychloride 50% WP'],
        'Leaf spot on fruit': ['Mancozeb 75% WP', 'Carbendazim 50% WP'],
        'Leaf spot on Leaves': ['Copper Oxychloride 50% WP', 'Mancozeb 75% WP'],
        'Blank Canker': ['Bordeaux Mixture', 'Copper Oxychloride 50% WP'],
        'Scale Insects': ['Buprofezin 25% SC', 'Spirotetramat 150 OD'],
        'White Flies': ['Diafenthiuron 50% WP', 'Pyriproxyfen 10% EC', 'Neem Oil + Soap']
    }
    
    growth_stages = ['Vegetative', 'Flowering', 'Fruit Development', 'Maturity']
    soil_types = ['Black Soil', 'Red Soil', 'Laterite Soil', 'Alluvial Soil']
    agro_zones = ['Coastal Konkan', 'Vidarbha', 'Marathwada', 'Western Maharashtra']
    
    data = []
    
    for _ in range(n_samples):
        disease = np.random.choice(diseases)
        treatment = np.random.choice(treatments[disease])
        
        # Weather conditions influence effectiveness
        temp = np.random.normal(28, 5)
        humidity = np.random.normal(70, 15)
        rainfall = np.random.exponential(5)
        wind_speed = np.random.normal(5, 2)
        
        # Calculate effectiveness based on conditions
        # (simplified logic - in reality would be learned from data)
        base_effectiveness = 0.75
        
        # Temperature effect
        if 25 <= temp <= 30:
            temp_effect = 0.1
        else:
            temp_effect = -0.1
        
        # Humidity effect
        if humidity > 80:
            humidity_effect = -0.05
        elif humidity < 50:
            humidity_effect = -0.05
        else:
            humidity_effect = 0.05
        
        # Add randomness
        noise = np.random.normal(0, 0.1)
        
        effectiveness = np.clip(base_effectiveness + temp_effect + humidity_effect + noise, 0.3, 1.0)
        
        data.append({
            'disease': disease,
            'confidence': np.random.uniform(0.7, 0.95),
            'temp': temp,
            'humidity': humidity,
            'rainfall': rainfall,
            'wind_speed': wind_speed,
            'treatment_applied': treatment,
            'effectiveness': effectiveness,
            'growth_stage': np.random.choice(growth_stages),
            'soil_type': np.random.choice(soil_types),
            'agro_zone': np.random.choice(agro_zones),
            'elevation': np.random.uniform(50, 800),
            'location': f"Location_{np.random.randint(1, 50)}"
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("AI TREATMENT RECOMMENDER - TRAINING")
    print("=" * 70)
    
    # Generate synthetic training data
    print("\n1. Generating synthetic training data...")
    history_df = generate_synthetic_training_data(n_samples=2000)
    print(f"Generated {len(history_df)} training samples")
    print(f"\nSample data:")
    print(history_df.head())
    
    # Save training data
    history_df.to_csv("data/treatment_history.csv", index=False)
    print("\n✓ Training data saved to data/treatment_history.csv")
    
    # Train model
    print("\n2. Training AI model...")
    recommender = AITreatmentRecommender()
    recommender.train_from_history(history_df, epochs=50, batch_size=32)
    
    # Test predictions
    print("\n" + "=" * 70)
    print("TESTING PREDICTIONS")
    print("=" * 70)
    
    test_cases = [
        {
            'disease': 'Athracnose',
            'confidence': 0.85,
            'weather': {'temp': 28, 'humidity': 85, 'precip': 15, 'wind_speed': 5},
            'growth_stage': 'Fruit Development'
        },
        {
            'disease': 'Mealy Bug',
            'confidence': 0.92,
            'weather': {'temp': 32, 'humidity': 65, 'precip': 0, 'wind_speed': 8},
            'growth_stage': 'Vegetative'
        },
        {
            'disease': 'White Flies',
            'confidence': 0.78,
            'weather': {'temp': 30, 'humidity': 55, 'precip': 0, 'wind_speed': 6},
            'growth_stage': 'Flowering'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['disease']}")
        print(f"  Weather: {test['weather']['temp']}°C, {test['weather']['humidity']}% humidity")
        print(f"  Growth Stage: {test['growth_stage']}")
        
        recommendations = recommender.predict_treatment(**test, top_k=3)
        
        print(f"\n  Recommended Treatments:")
        for j, rec in enumerate(recommendations, 1):
            print(f"    {j}. {rec['treatment']}")
            print(f"       Probability: {rec['probability']:.2%}")
            print(f"       Predicted Effectiveness: {rec['predicted_effectiveness']:.2%}")
            print(f"       Confidence: {rec['confidence_level']}")
    
    print("\n" + "=" * 70)
    print("AI MODEL READY FOR DEPLOYMENT!")
    print("=" * 70)
