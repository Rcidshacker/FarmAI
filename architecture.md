# FarmAI System Architecture

**Status:** Verified & Active
**Date:** 2026-02-08

## 1. System Overview

FarmAI follows a **"Twin Brain" Architecture**, combining hybrid AI models with biological rules to provide real-time disease detection and automated treatment scheduling.

### Technology Stack

* **Backend:** Python (FastAPI)
* **Frontend:** React (Vite)
* **Database:** SQLite (`data/farm_data.db`)
* **AI/ML Engine:** PyTorch (Inference), XGBoost
* **Job Scheduling:** `APScheduler` (Background tasks)

### Core Pattern

The system separates concerns into three layers:

1. **API Layer:** RESTful endpoints for frontend interaction.
2. **Intelligence Layer (Brains):**
    * **Disease Brain:** Hierarchical Deep Learning (Binary -> Multiclass).
    * **Spray Brain:** Reinforcement Learning (Q-Learning) for scheduling.
3. **Data Layer:** SQLite for persistence and file system for model artifacts.

---

## 2. Data Flow Diagrams

### Disease Detection Flow

Real-time analysis of uploaded leaf images.

```mermaid
graph LR
    User[User Upload] --> API[api/routes/disease.py]
    API --> Classifier[HierarchicalDiseaseClassifier]
    
    subgraph "Disease Brain (src/models)"
        Classifier --> Binary[Binary Model\n(Healthy vs Affected)]
        Binary -- Affected --> Multi[Multiclass Model\n(Specific Disease)]
        Binary -- Healthy --> Result
        Multi --> Result
    end
    
    Result[JSON Result] --> API
    API --> DB[(SQLite Logs)]
```

### Spray Scheduling Flow

Automated treatment planning based on weather and pest pressure.

```mermaid
graph LR
    UI[SpraySchedule.jsx] --> API[api/routes/treatment.py]
    API --> Brain[AutomatedSprayManager]
    
    subgraph "Spray Brain (src/automation)"
        Brain --> Scheduler[SprayScheduler]
        Scheduler --> Weather[Weather API / Mock]
        Scheduler --> Rules[RL Model (.pkl)]
    end
    
    Brain --> DB[(SQLite History)]
    Brain --> UI
```

---

## 3. Verified Model Inventory

The following models have been verified as present and active in the system.

| Model File | Python Wrapper Class | Type | Function |
| :--- | :--- | :--- | :--- |
| `binary_classifier.pth` | `DiseaseClassifier` (in `HierarchicalDiseaseClassifier`) | PyTorch (CNN) | Determines if a leaf is Healthy or Affected. |
| `multiclass_classifier.pth` | `DiseaseClassifier` (in `HierarchicalDiseaseClassifier`) | PyTorch (CNN) | Identifies specific disease if Affected. |
| `spray_scheduler.pkl` | `QLearningSprayScheduler` | Pickle (RL Table) | Determines optimal spray actions based on state. |
| `pest_forecast_xgb.joblib` | `PestForecaster` (in `src.models`) | XGBoost | Predicts future pest pressure based on weather. |
| `ai_recommender.pth` | `RecommenderSystem` | PyTorch | Suggests products/interventions (Experimental). |

---

## 4. Directory Map

### `src/api`

**The Gateway.** Contains FastAPI routes and schemas.

* `routes/disease.py`: Endpoints for image upload and prediction.
* `routes/treatment.py`: Endpoints for retrieving and managing spray schedules.

### `src/models`

**The Intelligence (Pattern Recognition).**

* `hierarchical_classifier.py`: The main entry point for disease detection. Orchestrates the binary and multiclass models.
* `disease_cnn_pytorch.py`: The low-level PyTorch wrapper for loading `.pth` files and running inference.

### `src/automation`

**The Strategy (Decision Making).**

* `spray_scheduler.py`: Contains the Reinforcement Learning logic (`QLearningSprayScheduler`) and the environment definition.
* `weather_service.py`: (Implicit) Handles weather data fetching for the scheduler.

### `src/database`

**The Memory.**

* `db_manager.py`: Manages SQLite connections and schema (`farm_data.db`).
