import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import shutil

logger = logging.getLogger(__name__)

class ActiveLearningService:
    """
    Service to handle user feedback and active learning loop.
    Stores feedback and manages the dataset for future retraining.
    """
    
    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = feedback_dir
        self.images_dir = os.path.join(feedback_dir, "images")
        self.metadata_file = os.path.join(feedback_dir, "feedback_metadata.json")
        
        # Create directories if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize metadata file if it doesn't exist
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump([], f)
                
    def submit_feedback(self, image_path: str, predicted_label: str, actual_label: str, 
                       confidence: float, user_comments: Optional[str] = None,
                       is_correct: bool = False) -> Dict:
        """
        Submit feedback for a prediction.
        
        Args:
            image_path: Path to the temporary image file
            predicted_label: The label predicted by the model
            actual_label: The label provided by the user (correction)
            confidence: Model's confidence score
            user_comments: Optional comments from the user
            is_correct: Whether the prediction was correct (if actual_label is not provided)
            
        Returns:
            Dict with status and feedback ID
        """
        try:
            feedback_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Copy image to feedback directory
            filename = f"{feedback_id}_{os.path.basename(image_path)}"
            dest_path = os.path.join(self.images_dir, filename)
            
            # If image_path is a temp file or uploaded file, we might need to handle it differently
            # For now assuming it's a path we can copy
            if os.path.exists(image_path):
                shutil.copy2(image_path, dest_path)
            else:
                logger.warning(f"Image file not found at {image_path}, saving metadata only")
                dest_path = "IMAGE_NOT_FOUND"

            feedback_entry = {
                "id": feedback_id,
                "timestamp": datetime.now().isoformat(),
                "image_path": dest_path,
                "predicted_label": predicted_label,
                "actual_label": actual_label if actual_label else (predicted_label if is_correct else "Unknown"),
                "confidence": confidence,
                "is_correct": is_correct,
                "user_comments": user_comments,
                "status": "pending_review" # pending_review, approved_for_training, rejected
            }
            
            # Append to metadata
            self._append_feedback(feedback_entry)
            
            # Check if we need to trigger retraining (mock logic)
            self._check_retraining_trigger()
            
            return {"status": "success", "feedback_id": feedback_id, "message": "Feedback received"}
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return {"status": "error", "message": str(e)}

    def _append_feedback(self, entry: Dict):
        """Append feedback entry to JSON file"""
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            data.append(entry)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback metadata: {e}")

    def _check_retraining_trigger(self):
        """Check if we have enough new data to retrain"""
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            # Count pending reviews
            pending_count = sum(1 for item in data if item["status"] == "pending_review")
            
            if pending_count >= 50: # Threshold for retraining
                logger.info(f"Feedback threshold reached ({pending_count} items). Triggering retraining pipeline...")
                # In a real system, this would call a background task or airflow DAG
                # For now, we just log it.
        except Exception as e:
            logger.error(f"Error checking retraining trigger: {e}")

    def get_feedback_stats(self) -> Dict:
        """Get statistics about collected feedback"""
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            total = len(data)
            correct = sum(1 for item in data if item["is_correct"])
            incorrect = total - correct
            
            return {
                "total_feedback": total,
                "correct_predictions": correct,
                "incorrect_predictions": incorrect,
                "accuracy_on_feedback": (correct / total) if total > 0 else 0
            }
        except Exception as e:
            return {"error": str(e)}
