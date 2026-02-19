# FarmAI System Overview

## Application Purpose

FarmAI is an AI-powered pest management and disease detection system designed specifically for Custard Apple farming. It utilizes computer vision and machine learning to help farmers identify crop diseases, predict pest risks, and optimize spray schedules.

## Tech Stack

### Backend

- **Framework:** FastAPI (Python)
- **Server:** Uvicorn
- **Language:** Python 3.x
- **Key Libraries:**
  - `torch`, `torchvision` (PyTorch for ML)
  - `xgboost`, `scikit-learn` (Traditional ML)
  - `pydantic` (Data validation)
  - `numpy`, `pandas` (Data processing)

### Frontend

- **Framework:** React 19
- **Build Tool:** Vite
- **Styling:** TailwindCSS
- **State Management:** React Hooks / Local State
- **Routing:** React Router DOM (v7)
- **HTTP Client:** Axios
- **Mobile Support:** Capacitor (Android)

### Database

- **Type:** SQLite
- **Location:** `data/farm_data.db`
- **ORM/Access:** Raw SQL via `src/database/db_manager.py`

### Machine Learning

- **Deep Learning:** PyTorch (CNN for disease detection)
- **Forecasting:** XGBoost (Pest risk prediction)
- **Computer Vision:** OpenCV, PIL, Albumentations

## Architecture Pattern

The system follows a **Monolithic Client-Server Architecture**:

- **Rich Client (Frontend):** Handles UI, user interaction, and state.
- **Stateless API (Backend):** RESTful API providing data, auth, and ML inference.
- **Embedded Database:** SQLite database embedded within the backend application.

## System Components

1. **Backend API Layer (`src/api/`)**
    - Entry point: `main.py`
    - Directs traffic to specific routers (Auth, Disease, Weather, etc.)
    - Handles CORS and Middleware.

2. **Data Processing Layer (`src/data_processing/`)**
    - `image_processor.py`: Prepares images for the CNN model.
    - `weather_processor.py`: Fetches and formats weather data.

3. **ML Model Layer (`src/models/`)**
    - `disease_cnn_pytorch.py`: PyTorch model for disease classification.
    - `pest_forecasting_model.py`: XGBoost model for pest risk.
    - `ai_treatment_recommender.py`: Logic for treatment advice.

4. **Database Layer (`src/database/`)**
    - `db_manager.py`: Centralized class for all SQLite operations (User management, Records, Feedback).

5. **Services Layer (`src/services/`)**
    - `ai_assistant.py`: Logic for the AI chatbot assistant.
    - `system.py`: System health and status.

## Component Interaction Diagram

```mermaid
graph TD
    Client[Frontend (React/Vite)] <-->|HTTP/REST JSON| API[Backend API (FastAPI)]
    
    subgraph Backend System
        API --> Auth[Auth Router]
        API --> Disease[Disease Router]
        API --> Pest[Pest Router]
        
        Disease --> ImgProc[Image Processor]
        ImgProc --> CNN[PyTorch Model]
        
        Pest --> WeatherProc[Weather Processor]
        WeatherProc --> XGB[XGBoost Model]
        
        Auth --> DB[(SQLite Database)]
        Disease --> DB
        Pest --> DB
    end
    
    WeatherProc <-->|HTTPS| ExternalWeather[External Weather API]
```

## Data Persistence

- User profiles and authentication data are stored in the `users` table.
- Farming interventions (sprays/fertilizers) are logged in `spray_records`.
- Model feedback and correction data is stored in `feedback`.
