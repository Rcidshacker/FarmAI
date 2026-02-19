# Data Flow Documentation

## 1. Disease Detection Pipeline

**Flow:** `User Camera` -> `FastAPI (/detect-disease)` -> `ImageProcessor` -> `DiseaseClassifier (CNN)` -> `Heuristic Layer` -> `Response`

### Steps

1. **Input:** Image file uploaded via `multipart/form-data` + Context (Wind Speed, Fruit Density).
2. **Preprocessing (`src/data_processing/image_processor.py`):**
    - Image loaded via OpenCV.
    - Resized to 224x224 and normalized.
    - Quality assessment (Blur/Brightness check).
3. **Inference (`src/models/disease_cnn_pytorch.py`):**
    - PyTorch model predicts class probabilities.
    - Returns top prediction + confidence.
4. **Heuristic Refinement (`src/api/routes/disease.py`):**
    - **Physical Damage Logic:** If `wind_speed > 20` and `fruit_density == "High"`, boosts probability of "Physical Damage" and adds a specific warning.
5. **Output:** JSON with detected disease, confidence, and actionable advice.

## 2. "Twin Brain" Pest Prediction

**Flow:** `User Location` -> `FastAPI (/predict-pest-risk)` -> `External APIs` + `ML Model` -> `EnKF Fusion` -> `Response`

### Steps

1. **Input:** User Lat/Lon, Crop Stage.
2. **Data Gathering (Sequential & Blocking):**
    - **Weather:** Fetches forecast (likely mock or external) or uses provided context.
    - **Satellite (`src/services/satellite_service.py`):** Calls AgroMonitoring API (using `requests`).
        - *Note: This is a synchronous blocking call inside an async route.*
        - Retries 30-day history to find cloud-free NDVI/RVI data.
3. **ML Forecast (`src/models/pest_forecasting_model.py`):**
    - Features: Weather rolling avgs + Satellite indices.
    - Model: XGBoost Regressor predicts baseline risk.
4. **Data Fusion (EnKF):**
    - Combines ML prediction with "Biological Risk" (theoretical model) using an Ensemble Kalman Filter.
5. **Output:** Final Risk Score (0-100) and color code.

## 3. Active Learning Loop

**Flow:** `User Feedback` -> `FastAPI (/submit-treatment-feedback)` -> `ActiveLearningService` -> `JSON Storage`

### Steps

1. **Input:** Image, Predicted Label, Actual Label (User Correction).
2. **Storage (`src/services/active_learning_service.py`):**
    - Image copied to `data/feedback/images/`.
    - Metadata appended to `feedback_metadata.json`.
3. **Trigger:**
    - Checks if `pending_review` count >= 50.
    - *Current Behavior:* Logs "Triggering retraining pipeline" (Mock).

## 4. AI Assistant w/ Context Injection

**Flow:** `User Query` -> `FastAPI (/assistant/chat)` -> `Context Builder` -> `AIAssistantService` -> `OpenRouter / Offline KB`

### Steps

1. **Context Injection:**
    - route automatically injects: Weather (current), Soil Type, Crop Stage.
2. **Routing (`src/services/ai_assistant.py`):**
    - **Online:** Uses OpenRouter (Llama 3.3). System prompt includes the injected context.
    - **Offline/Fallback:** Keywords search in `knowledge_base/*.json`.
3. **Output:** Text response + metadata.
