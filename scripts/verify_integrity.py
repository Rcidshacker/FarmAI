import sys
import os
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrityVerifier")

def verify_database():
    print("\n--- Testing Database ---")
    try:
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager()
        print(f"DatabaseManager instantiated: {db}")
        print(f"DB Path: {db.db_path}")
        
        if hasattr(db, 'client'):
             print(f"Database Client Type: {type(db.client)}")
        else:
             import sqlite3
             print("Database Type: SQLite (inferred)")
             conn = sqlite3.connect(db.db_path)
             cursor = conn.cursor()
             cursor.execute("SELECT sqlite_version()")
             version = cursor.fetchone()[0]
             print(f"SQLite Connection Successful. Version: {version}")
             conn.close()

    except Exception as e:
        print(f"Database Verification Failed: {e}")
        import traceback
        traceback.print_exc()

def verify_disease_brain():
    print("\n--- Testing Disease Brain ---")
    try:
        from src.models.hierarchical_classifier import HierarchicalDiseaseClassifier
        classifier = HierarchicalDiseaseClassifier()
        print(f"Classifier instantiated.")
        
        # Check model paths
        if hasattr(classifier, 'binary_model') and hasattr(classifier.binary_model, 'model_path'):
            print(f"Binary Model Path Conf: {classifier.binary_model.model_path}")
            print(f"Binary Model Path Exists: {classifier.binary_model.model_path.exists()}")
            # Print absolute path
            print(f"Binary Model Abs Path: {classifier.binary_model.model_path.resolve()}")

        # Check for mock flag
        if hasattr(classifier, 'use_mocks'):
            print(f"Internal 'use_mocks' flag: {classifier.use_mocks}")
        
        # Try to load models explicitly
        print("Calling load_models()...")
        success = classifier.load_models()
        print(f"load_models() result: {success}")
        
        # Determine if it's actually using the files
        if hasattr(classifier, 'binary_model'):
            print(f"Binary Model type: {type(classifier.binary_model)}")
            if hasattr(classifier.binary_model, 'model') and classifier.binary_model.model is not None:
                 print("Binary Model .model attribute is populated (Real Model Loaded).")
            else:
                 print("Binary Model .model attribute is None (Model NOT Loaded).")
        
    except Exception as e:
        print(f"Disease Brain Verification Failed: {e}")
        import traceback
        traceback.print_exc()

def verify_spray_brain():
    print("\n--- Testing Spray Brain ---")
    try:
        from src.automation.spray_scheduler import AutomatedSprayManager
        manager = AutomatedSprayManager()
        print(f"SprayManager instantiated.")
        
        print("Testing create_schedule with dummy data...")
        schedule = manager.create_schedule("Test Farm", days_ahead=2)
        print("Schedule generated.")
        print(f"Schedule keys: {list(schedule.keys())}")
        
    except Exception as e:
        print(f"Spray Brain Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting Truth Serum Protocol (Strict Mode)...")
    verify_database()
    verify_disease_brain()
    verify_spray_brain()
    print("\n--- Verification Complete ---")
