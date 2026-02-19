# Machine Learning Models Documentation

## Overview

FarmAI employs a hybrid AI system combining Computer Vision (CNNs), Structured Data Forecasting (XGBoost), and Neural Collaborative Filtering for treatment recommendations.

## 1. Disease Detection Model

**File:** `src/models/disease_cnn_pytorch.py`
**Class:** `DiseaseClassifier` -> `DiseaseCNN`

### Architecture

- **Backbone:** Supports ResNet18, ResNet50, EfficientNet-B0, MobileNetV2 (Transfer Learning).
- **Head:** Custom classifier with Dropout and Batch Normalization.
- **Classes:** 8 (Anthracnose, Mealy Bug, Diplodia Rot, etc.)

### Pipeline

1. **Input:** Image (Numpy array or PIL Image).
2. **Preprocessing:**
    - Resize to 224x224.
    - Normalization (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]).
    - Augmentations (Training only): Flip, Rotation, ColorJitter.
3. **Inference:** `model.predict(image)` returns class index and confidence.
4. **Heuristics:** `src/api/routes/disease.py` adds logic for "Physical Damage" based on wind speed and fruit density.

## 2. Pest Risk Forecasting (Twin Brain)

**File:** `src/models/pest_forecasting_model.py`
**Class:** `PestForecastingModel`

### Architecture

- **Algorithm:** XGBoost Regressor (`xgb.XGBRegressor`).
- **Target:** Mealy Bug Risk Score (0-1).

### Features

- **Weather:** Temperature, Humidity, Rainfall, Wind Speed (Rolling averages and lags).
- **Satellite:** RVI, VH, VV indices (Smoothed via Savitzky-Golay filter).
- **Derived:** `Bio_Heat_Index`, `Rain_Intensity_Index`.

### Pipeline

1. **Input:** Weather forecast dataframe + Satellite indices.
2. **Feature Engineering:** Generates rolling windows (14, 30, 60 days) and lags.
3. **Prediction:** Outputs numerical risk score.
4. **Correction:** The API layer (`src/api/routes/pest.py`) applies an **EnKF (Ensemble Kalman Filter)** to fuse this prediction with manual user observations.

## 3. AI Treatment Recommender

**File:** `src/models/ai_treatment_recommender.py`
**Class:** `AITreatmentRecommender` -> `TreatmentRecommenderNN`

### Architecture

- **Type:** Multi-task Neural Network (PyTorch).
- **Tasks:**
    1. **Classification:** Predict best treatment (CrossEntropyLoss).
    2. **Regression:** Predict effectiveness score (MSELoss).

### Features

- **Disease:** One-hot encoded disease type.
- **Context:** Confidence, Weather (Temp, Humidity), Soil Type, Agro-climatic Zone.
- **Seasonality:** Month (Cyclical encoding).
- **Growth Stage:** Normalized stage index.

### Online Learning

- The system supports `update_with_feedback()` to retrain the model on new data points collected from expert farmers (Active Learning).
