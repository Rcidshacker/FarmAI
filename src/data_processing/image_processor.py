"""
Image Processor for Disease Detection
Handles image loading, preprocessing, and augmentation
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import albumentations as A
from PIL import Image
import glob

from ..utils.logger import setup_logger
from ..utils.config import (
    IMAGE_SIZE, CUSTARD_APPLE_DATASET, 
    DISEASE_CLASSES, AUGMENTATION_CONFIG, 
    METADATA_CSV, BASE_DIR
)

logger = setup_logger('image_processor')


class ImageProcessor:
    """Process images for disease detection model"""
    
    def __init__(self, image_size: Tuple[int, int] = IMAGE_SIZE):
        """
        Initialize Image Processor
        
        Args:
            image_size: Target image size (height, width)
        """
        self.image_size = image_size
        self.metadata = None
        self.class_mapping = {cls: idx for idx, cls in enumerate(DISEASE_CLASSES)}
        self.reverse_mapping = {idx: cls for cls, idx in self.class_mapping.items()}
        
        # Setup augmentation pipeline
        self.augmentation = A.Compose([
            A.Rotate(limit=AUGMENTATION_CONFIG['rotation_range'], p=0.5),
            A.ShiftScaleRotate(
                shift_limit=AUGMENTATION_CONFIG['width_shift_range'],
                scale_limit=AUGMENTATION_CONFIG['zoom_range'],
                rotate_limit=0,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5 if AUGMENTATION_CONFIG['horizontal_flip'] else 0),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Basic preprocessing (no augmentation)
        self.preprocessing = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_metadata(self, metadata_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load image metadata from CSV
        
        Args:
            metadata_path: Path to metadata CSV file
            
        Returns:
            DataFrame with metadata
        """
        if metadata_path is None:
            metadata_path = METADATA_CSV
        
        logger.info(f"Loading metadata from {metadata_path}")
        
        try:
            self.metadata = pd.read_csv(metadata_path)
            logger.info(f"Loaded metadata for {len(self.metadata)} images")
            
            # Parse dates
            if 'date_taken' in self.metadata.columns:
                self.metadata['date_taken'] = pd.to_datetime(
                    self.metadata['date_taken'], 
                    format='%Y:%m:%d %H:%M:%S',
                    errors='coerce'
                )
            
            return self.metadata
        
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return pd.DataFrame()
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array or None if error
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = cv2.resize(image, self.image_size)
            
            return image
        
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def preprocess_image(self, image: np.ndarray, augment: bool = False) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input image array
            augment: Whether to apply augmentation
            
        Returns:
            Preprocessed image
        """
        if augment:
            transformed = self.augmentation(image=image)
        else:
            transformed = self.preprocessing(image=image)
        
        return transformed['image']
    
    def load_dataset(self, 
                    dataset_path: Optional[Path] = None,
                    class_filter: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load complete dataset from directory structure
        
        Args:
            dataset_path: Path to dataset directory
            class_filter: List of classes to load (None for all)
            
        Returns:
            Tuple of (images, labels, image_paths)
        """
        if dataset_path is None:
            dataset_path = CUSTARD_APPLE_DATASET
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        images = []
        labels = []
        image_paths = []
        
        # Get disease classes to process
        classes = class_filter if class_filter else DISEASE_CLASSES
        
        for disease_class in classes:
            class_dir = dataset_path / disease_class
            
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            # Get all images in class directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(glob.glob(str(class_dir / ext)))
            
            logger.info(f"Loading {len(image_files)} images for class: {disease_class}")
            
            for img_path in image_files:
                image = self.load_image(img_path)
                
                if image is not None:
                    images.append(image)
                    labels.append(self.class_mapping.get(disease_class, -1))
                    image_paths.append(img_path)
        
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} images from {len(set(labels))} classes")
        logger.info(f"Image shape: {images.shape}")
        
        return images, labels, image_paths
    
    def get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """
        Get distribution of classes in dataset
        
        Args:
            labels: Array of label indices
            
        Returns:
            Dictionary with class counts
        """
        distribution = {}
        unique, counts = np.unique(labels, return_counts=True)
        
        for label_idx, count in zip(unique, counts):
            class_name = self.reverse_mapping.get(label_idx, 'Unknown')
            distribution[class_name] = int(count)
        
        return distribution
    
    def create_data_generator(self, 
                             images: np.ndarray,
                             labels: np.ndarray,
                             batch_size: int = 32,
                             augment: bool = False,
                             shuffle: bool = True):
        """
        Create data generator for training
        
        Args:
            images: Array of images
            labels: Array of labels
            batch_size: Batch size
            augment: Whether to apply augmentation
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of (images, labels)
        """
        num_samples = len(images)
        indices = np.arange(num_samples)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_images = []
                batch_labels = labels[batch_indices]
                
                for idx in batch_indices:
                    img = images[idx]
                    processed = self.preprocess_image(img, augment=augment)
                    batch_images.append(processed)
                
                yield np.array(batch_images), batch_labels
    
    def extract_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Color statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            features[f'{channel}_mean'] = np.mean(image[:, :, i])
            features[f'{channel}_std'] = np.std(image[:, :, i])
        
        # Brightness
        features['brightness'] = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        
        # Saturation
        features['saturation'] = np.mean(hsv[:, :, 1])
        
        # Edge density (indicator of disease texture)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        return features
    
    def visualize_augmentation(self, image: np.ndarray, num_samples: int = 5) -> List[np.ndarray]:
        """
        Generate augmented versions of an image for visualization
        
        Args:
            image: Input image
            num_samples: Number of augmented samples to generate
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for _ in range(num_samples):
            transformed = self.augmentation(image=image)
            augmented_images.append(transformed['image'])
        
        return augmented_images
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, any]:
        """
        Assess image quality for disease detection
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with quality metrics
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast
        contrast = gray.std()
        
        # Overall quality score
        quality_score = min(100, (sharpness / 100 + brightness / 2.55 + contrast / 2.55) / 3 * 100)
        
        quality = {
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'quality_score': quality_score,
            'is_acceptable': quality_score > 30  # Threshold for acceptable images
        }
        
        return quality
    
    def get_disease_severity_indicators(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract indicators that might correlate with disease severity
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with severity indicators
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Brown/dark spots (potential disease areas)
        brown_mask = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 200]))
        brown_percentage = np.sum(brown_mask > 0) / brown_mask.size * 100
        
        # Green health (healthy tissue)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_percentage = np.sum(green_mask > 0) / green_mask.size * 100
        
        # Texture variation (rough texture indicates disease)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'brown_percentage': brown_percentage,
            'green_percentage': green_percentage,
            'texture_variance': texture_variance,
            'severity_estimate': min(100, brown_percentage * 1.5)  # Rough estimate
        }


if __name__ == '__main__':
    # Example usage
    processor = ImageProcessor()
    
    # Load metadata
    metadata = processor.load_metadata()
    print(f"\n=== Metadata Summary ===")
    print(f"Total images: {len(metadata)}")
    print(f"Disease classes: {metadata['disease_label'].unique()}")
    
    # Load a sample image
    sample_dir = CUSTARD_APPLE_DATASET / 'Mealy Bug'
    sample_images = list(sample_dir.glob('*.jpg'))[:1]
    
    if sample_images:
        image = processor.load_image(str(sample_images[0]))
        if image is not None:
            print(f"\n=== Sample Image Analysis ===")
            print(f"Image shape: {image.shape}")
            
            # Quality assessment
            quality = processor.assess_image_quality(image)
            print(f"\nQuality metrics:")
            for key, value in quality.items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
            
            # Severity indicators
            severity = processor.get_disease_severity_indicators(image)
            print(f"\nSeverity indicators:")
            for key, value in severity.items():
                print(f"  {key}: {value:.2f}")
