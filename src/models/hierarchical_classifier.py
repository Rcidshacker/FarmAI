import sys
import os
from pathlib import Path

# Add project root to path to allow imports from src
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import os
from PIL import Image
from src.models.disease_cnn_pytorch import DiseaseClassifier

logger = logging.getLogger(__name__)

class HierarchicalDiseaseClassifier:
    """
    Two-stage classifier:
    1. Binary Classification (Affected vs Healthy)
    2. Multiclass Classification (Specific Disease)
    """
    
    def __init__(self):
        # Binary Model: 2 classes (Healthy, Affected)
        self.binary_model = DiseaseClassifier(
            num_classes=2,
            class_names=['Healthy', 'Affected'],
            model_name='binary_classifier'
        )
        
        # Multiclass Model: Specific diseases
        # These should match the folders in the dataset
        self.disease_classes = [
            'Anthracnose', 'Blank Canker', 'Diplodia Rot', 
            'Leaf spot on fruit', 'Leaf spot on Leaves', 'Mealy Bug',
            'Physical Damage (Rubbing)' # Added new class
        ]
        
        self.multiclass_model = DiseaseClassifier(
            num_classes=len(self.disease_classes),
            class_names=self.disease_classes,
            model_name='multiclass_classifier'
        )
        
    def load_models(self):
        """Load both models"""
        binary_loaded = self.binary_model.load_model()
        multiclass_loaded = self.multiclass_model.load_model()
        
        if not binary_loaded:
            logger.warning("Binary model not found. Please train it first.")
        if not multiclass_loaded:
            logger.warning("Multiclass model not found. Please train it first.")
            
        return binary_loaded and multiclass_loaded
        
    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Predict disease using hierarchical approach
        """
        # --- NEW CHECK: Mock response if models are missing ---
        if not self.binary_model.model or not self.multiclass_model.model:
            logger.warning("Models not loaded. Returning mock prediction for testing.")
            return [{
                'class': 'Mock Disease (Model Missing)',
                'confidence': 0.95,
                'model': 'mock'
            }]
        # ------------------------------------------------------
        
        # Ensure image is (1, H, W, C)
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image

        # Step 1: Binary Check
        try:
            # binary_model.predict returns (indices, probabilities)
            binary_indices, binary_probs = self.binary_model.predict(image_batch)
            
            # Get top prediction for the single image
            top_idx = binary_indices[0]
            top_prob = np.max(binary_probs[0])
            top_class = self.binary_model.class_names[top_idx]
            
            binary_result = {
                'class': top_class,
                'confidence': float(top_prob),
                'model': 'binary'
            }
            
            # If Healthy, return immediately
            if top_class == 'Healthy':
                return [binary_result]
            
            # Step 2: Specific Disease
            # If Affected, run the multiclass model
            multi_indices, multi_probs = self.multiclass_model.predict(image_batch)
            
            # Get all predictions sorted by confidence
            probs = multi_probs[0]
            sorted_indices = np.argsort(probs)[::-1]
            
            results = []
            for idx in sorted_indices:
                results.append({
                    'class': self.multiclass_model.class_names[idx],
                    'confidence': float(probs[idx]),
                    'model': 'multiclass'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hierarchical prediction: {e}")
            # Fallback to multiclass if binary fails or vice versa?
            # For now, re-raise or return empty
            return []

    def _load_images_from_folder(self, folder_path: str, image_size: int = 224) -> List[np.ndarray]:
        """Helper to load all images from a folder"""
        images = []
        if not os.path.exists(folder_path):
            return images
            
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(folder_path, filename)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((image_size, image_size))
                    images.append(np.array(img))
                except Exception as e:
                    logger.warning(f"Failed to load image {filename}: {e}")
        return images

    def prepare_binary_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for Healthy vs Affected"""
        X = []
        y = []
        
        logger.info("Preparing Binary Dataset...")
        
        # Check for Healthy folder
        healthy_path = os.path.join(data_dir, 'Healthy')
        if os.path.exists(healthy_path):
            healthy_imgs = self._load_images_from_folder(healthy_path)
            X.extend(healthy_imgs)
            y.extend([0] * len(healthy_imgs)) # 0 = Healthy
            logger.info(f"Loaded {len(healthy_imgs)} Healthy images")
        else:
            logger.warning("No 'Healthy' folder found! Binary model cannot be trained properly.")
            
        # Load Affected (all other folders)
        affected_count = 0
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path) and folder != 'Healthy':
                imgs = self._load_images_from_folder(folder_path)
                X.extend(imgs)
                y.extend([1] * len(imgs)) # 1 = Affected
                affected_count += len(imgs)
        
        logger.info(f"Loaded {affected_count} Affected images")
        
        return np.array(X), np.array(y)

    def prepare_multiclass_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for Specific Diseases"""
        X = []
        y = []
        
        logger.info("Preparing Multiclass Dataset...")
        
        for idx, disease in enumerate(self.disease_classes):
            folder_path = os.path.join(data_dir, disease)
            if os.path.exists(folder_path):
                imgs = self._load_images_from_folder(folder_path)
                X.extend(imgs)
                y.extend([idx] * len(imgs))
                logger.info(f"Loaded {len(imgs)} images for {disease}")
            else:
                logger.warning(f"Folder not found for disease: {disease}")
                
        return np.array(X), np.array(y)

    def train(self, data_dir: str, epochs: int = 20):
        """
        Train both models from the same dataset structure.
        """
        logger.info(f"Starting Hierarchical Training from {data_dir}")
        
        # 1. Train Binary Model
        logger.info("--- Stage 1: Binary Model Training ---")
        X_bin, y_bin = self.prepare_binary_data(data_dir)
        
        if len(X_bin) > 0 and len(np.unique(y_bin)) > 1:
            self.binary_model.train(X_bin, y_bin, epochs=epochs)
        else:
            logger.error("Insufficient data for binary training (need both Healthy and Affected classes)")
            
        # 2. Train Multiclass Model
        logger.info("--- Stage 2: Multiclass Model Training ---")
        X_multi, y_multi = self.prepare_multiclass_data(data_dir)
        
        if len(X_multi) > 0:
            self.multiclass_model.train(X_multi, y_multi, epochs=epochs)
        else:
            logger.error("No data found for multiclass training")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Initializing Hierarchical Classifier...")
    try:
        classifier = HierarchicalDiseaseClassifier()
        print("Success! Hierarchical Classifier initialized.")
        
        # Check for dataset
        dataset_path = r"d:\Pest_prevention_prediction and medication\Custard Apple dataset"
        if os.path.exists(dataset_path):
            print(f"Dataset found at {dataset_path}")
            print("To train, uncomment the following line in the script:")
            # classifier.train(dataset_path, epochs=5)
        else:
            print(f"Dataset not found at {dataset_path}")
            
    except Exception as e:
        print(f"Initialization failed: {e}")
