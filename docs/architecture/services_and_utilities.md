# Services and Utilities Documentation

## 1. Services (`src/services/`)

### `active_learning_service.py`

- **Purpose:** Manages the feedback loop for model improvement.
- **Storage:** Local JSON (`data/feedback/feedback_metadata.json`) + Image Storage.
- **Key Methods:**
  - `submit_feedback()`: Saves image and metadata.
  - `_check_retraining_trigger()`: Checks if feedback count >= 50 (Mock trigger).
- **Issues:** Blocking I/O (file copy), JSON storage is not concurrent-safe.

### `satellite_service.py`

- **Purpose:** Fetches NDVI data from AgroMonitoring API.
- **Logic:**
  - Defines a 50m polygon around the user's location.
  - Searches last 30 days for cloud-free images (<15% clouds).
  - Returns the most recent valid NDVI value.
- **Issues:**
  - Uses `requests` (synchronous) inside async routes, blocking the event loop.
  - Hardcoded API key handling (though user provides one, fallback exists).
  - No caching of Polygon IDs (re-registers polygon on every call).

### `ai_assistant.py`

- **Purpose:** Hybrid Chatbot (Online LLM + Offline Rule-based).
- **Online:** OpenRouter API (Llama 3.3).
  - Injects system prompt with `FarmAI` persona and user context.
- **Offline:** Keyword matching against local JSONs (`chemical_compositions.json`, `pest_database.json`).
- **Issues:** Synchronous `requests` usage.

## 2. Data Processing (`src/data_processing/`)

### `image_processor.py`

- **Purpose:** Image loading, preprocessing, and augmentation.
- **Stack:** OpenCV, Albumentations.
- **Features:**
  - `assess_image_quality()`: Checks sharpness/brightness.
  - `get_disease_severity_indicators()`: Heuristics based on color masks (Brown/Green ratio).

### `weather_processor.py`

- **Purpose:** Weather data cleaning and feature engineering.
- **Stack:** Pandas.
- **Features:**
  - `_aggregate_to_daily()`: Converts hourly data to daily summaries.
  - `calculate_pest_risk()`: Applies biological thresholds to weather data.
  - `predict_upcoming_risk()`: Simple forecasting based on trends.

## 3. Utilities (`src/utils/`)

- **`logger.py`**: Standard Python logging configuration.
- **`config.py`**: Central configuration (Paths, Constants).
- **`helpers.py`**: Shared helper functions (Normalization, Risk calc).
