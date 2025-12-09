"""
Disease Detection CNN Model - PyTorch Implementation
Deep learning model for custard apple disease classification with GPU acceleration
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import json
from datetime import datetime
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DiseaseDataset(Dataset):
    """PyTorch Dataset for disease images"""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        """
        Args:
            images: Array of images (N, H, W, C)
            labels: Array of labels (N,)
            transform: Optional transform to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DiseaseCNN(nn.Module):
    """
    PyTorch CNN for disease classification with transfer learning
    Supports: ResNet, EfficientNet, MobileNet
    """
    
    def __init__(self, num_classes: int, model_type: str = 'resnet18', pretrained: bool = True):
        """
        Args:
            num_classes: Number of disease classes
            model_type: Model architecture ('resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v2')
            pretrained: Use pretrained weights
        """
        super(DiseaseCNN, self).__init__()
        
        self.model_type = model_type
        self.num_classes = num_classes
        
        # Load pretrained model
        if model_type == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original FC layer
            
        elif model_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_type == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_type == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class DiseaseClassifier:
    """
    Complete Disease Classification System with PyTorch
    """
    
    def __init__(self, 
                 num_classes: int = 8,
                 class_names: List[str] = None,
                 model_type: str = 'resnet18',
                 image_size: int = 224,
                 model_dir: str = 'models',
                 model_name: str = None):
        """
        Initialize Disease Classifier
        
        Args:
            num_classes: Number of disease classes
            class_names: List of class names
            model_type: Architecture type
            image_size: Input image size
            model_dir: Directory to save models
            model_name: Custom model filename (optional)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.model_type = model_type
        self.image_size = image_size
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name:
            self.model_path = self.model_dir / f"{model_name}.pth"
        else:
            self.model_path = self.model_dir / f"disease_classifier_{model_type}.pth"
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Initialize model
        self.model = None
        self.trained = False
        self.history = None
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def build_model(self):
        """Build the CNN model"""
        self.model = DiseaseCNN(self.num_classes, self.model_type, pretrained=True)
        self.model = self.model.to(self.device)
        logger.info(f"Model built: {self.model_type} with {self.num_classes} classes")
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray = None,
             y_val: np.ndarray = None,
             epochs: int = 50,
             batch_size: int = 32,
             learning_rate: float = 0.001,
             val_split: float = 0.2) -> Dict:
        """
        Train the disease classification model
        
        Args:
            X_train: Training images (N, H, W, C)
            y_train: Training labels (N,)
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio if X_val not provided
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        logger.info(f"Starting training with {len(X_train)} samples")
        
        # Create datasets
        if X_val is None:
            # Split training data
            dataset = DiseaseDataset(X_train, y_train, transform=self.train_transform)
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Apply validation transform to val_dataset
            val_dataset.dataset.transform = self.val_transform
        else:
            train_dataset = DiseaseDataset(X_train, y_train, transform=self.train_transform)
            val_dataset = DiseaseDataset(X_val, y_val, transform=self.val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 10
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate training metrics
            epoch_train_loss = train_loss / len(train_dataset)
            epoch_train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device, dtype=torch.long)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate validation metrics
            epoch_val_loss = val_loss / len(val_dataset)
            epoch_val_acc = val_correct / val_total
            
            # Update learning rate
            scheduler.step(epoch_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            history['learning_rates'].append(current_lr)
            
            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
                          f"LR: {current_lr:.6f}")
            
            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_val_loss = epoch_val_loss
                self.save_model()
                logger.info(f"Best model saved at epoch {epoch+1} with val acc: {best_val_acc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        self.trained = True
        self.history = history
        
        # Save training results
        results = {
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'epochs_trained': len(history['train_loss']),
            'history': history
        }
        
        self._save_training_results(results)
        
        logger.info("Training completed!")
        return results
    
    def predict(self, images: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict disease classes for images
        
        Args:
            images: Input images (N, H, W, C)
            batch_size: Batch size for prediction
            
        Returns:
            predictions: Predicted class indices
            probabilities: Class probabilities
        """
        if not self.trained:
            raise ValueError("Model not trained. Train or load model first.")
        
        self.model.eval()
        
        # Create dataset and dataloader
        dataset = DiseaseDataset(images, np.zeros(len(images)), transform=self.val_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_images, _ in dataloader:
                batch_images = batch_images.to(self.device)
                outputs = self.model(batch_images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 32) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test images
            y_test: Test labels
            batch_size: Batch size
            
        Returns:
            Evaluation metrics dictionary
        """
        predictions, probabilities = self.predict(X_test, batch_size)
        
        accuracy = accuracy_score(y_test, predictions)
        
        # Classification report
        report = classification_report(y_test, predictions, target_names=self.class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, predictions, target_names=self.class_names))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def _save_training_results(self, results: Dict):
        """Save comprehensive training results"""
        from datetime import datetime
        
        results_dir = Path("training_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to JSON
        metrics_file = results_dir / f"disease_cnn_pytorch_metrics_{timestamp}.json"
        save_results = {
            'best_val_acc': float(results['best_val_acc']),
            'best_val_loss': float(results['best_val_loss']),
            'final_train_acc': float(results['final_train_acc']),
            'final_val_acc': float(results['final_val_acc']),
            'epochs_trained': results['epochs_trained'],
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'device': str(self.device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'timestamp': timestamp
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(save_results, f, indent=4)
        
        logger.info(f"Training metrics saved to {metrics_file}")
        
        # Plot training curves
        try:
            self._plot_training_curves(results['history'], results_dir / f"disease_cnn_pytorch_training_{timestamp}.png")
        except Exception as e:
            logger.warning(f"Could not create training plots: {e}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY - Disease CNN (PyTorch)")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model_type}")
        logger.info(f"Epochs: {results['epochs_trained']}")
        logger.info(f"Best Val Acc: {results['best_val_acc']:.4f}")
        logger.info(f"Best Val Loss: {results['best_val_loss']:.4f}")
        logger.info(f"Final Val Acc: {results['final_val_acc']:.4f}")
        logger.info("=" * 60)
    
    def _plot_training_curves(self, history: Dict, save_path: Path):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Best metrics text
        ax4.axis('off')
        summary_text = f"""
        Model: {self.model_type}
        Device: {self.device}
        
        Best Validation Accuracy: {max(history['val_acc']):.4f}
        Best Validation Loss: {min(history['val_loss']):.4f}
        
        Final Train Accuracy: {history['train_acc'][-1]:.4f}
        Final Val Accuracy: {history['val_acc'][-1]:.4f}
        
        Epochs Trained: {len(history['train_loss'])}
        Final Learning Rate: {history['learning_rates'][-1]:.6f}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {save_path}")
    
    def save_model(self):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'image_size': self.image_size,
            'trained': self.trained
        }
        torch.save(checkpoint, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """Load model checkpoint"""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.model_type = checkpoint['model_type']
            self.num_classes = checkpoint['num_classes']
            self.class_names = checkpoint['class_names']
            self.image_size = checkpoint['image_size']
            
            # Build model
            self.build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Fix: Ensure trained flag is True after loading valid weights
            self.trained = True
            
            logger.info(f"Model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
