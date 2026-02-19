Mealybug Prediction System

Technical Documentation

Custard Apple Pest Management using Twin Brain Architecture

FarmAI Project

February 14, 2026

# Executive Summary

The FarmAI Mealybug Prediction System represents a sophisticated fusion of biological modeling and machine learning, designed specifically for custard apple pest management. The system employs a "Twin Brain" architecture that combines:

A mechanistic Biological Engine ("Teacher") based on Growing Degree Days (GDD)

An XGBoost Machine Learning Model ("Student") trained on historical patterns

Real-time satellite vegetation indices and manual observations

Ensemble Kalman Filter (EnKF) for data fusion and uncertainty quantification

Key Innovation: The system models complex ecological interactions including ant-mealybug mutualism, rain washout effects, and crop phenology-specific vulnerability windows.

# 1. System Architecture

## 1.1 Twin Brain Approach

The prediction pipeline integrates three independent "brains" that process different aspects of pest risk:

| Component | Technology | Purpose |
| --- | --- | --- |
| Brain 1 | Biological GDD Model (Teacher) | Mechanistic pest development based on thermal time |
| Brain 2 | XGBoost ML Model (Student) | Pattern recognition from historical weather-pest data |
| Brain 3 | Satellite + Manual RVI | Real-time crop health and user ground-truth observations |
| Fusion | Ensemble Kalman Filter (EnKF) | Weighted fusion with K=0.7 gain, prioritizing user observations |

# 2. Biological Model: The "Teacher"

File: src/models/biological_risk_model.py

## 2.1 Growing Degree Day (GDD) Model

The system uses a classical thermal-time accumulation model to track mealybug development:

| Parameter | Value / Description |
| --- | --- |
| Base Temperature (Tbase) | 15°C (minimum temperature for development) |
| Upper Threshold (Tupper) | 35°C (maximum effective temperature) |
| Daily DD Calculation | max(0, min(Tavg, 35) - 15) |
| Generation Time | 350 DD per complete generation |
| Max Generations per Season | 3.5 generations (used for risk normalization) |

## 2.2 Biofix (Season Start Detection)

The model determines when mealybug populations begin their seasonal activity:

Detection Window: June through August

Trigger Condition: Cumulative rainfall ≥ 10mm

Initial Population: System starts with 35 accumulated Degree Days

## 2.3 Environmental Modifiers

### Rain Washout Effect

Heavy rainfall physically removes mealybug crawlers (first instar nymphs) from host plants:

Threshold: Rainfall > 80mm in a single day

Effect: Accumulated DD reduced by 50% (population loss)

### Seasonal Peak Boost

September through November represents the optimal period for mealybug reproduction:

Risk Multiplier: 1.5× during peak months

Biological Rationale: Coincides with optimal temperature and fruit development

# 3. Ecological Interactions

## 3.1 Ant-Mealybug Mutualism

The system explicitly models the protective relationship between ants and mealybugs. Ants defend mealybugs from natural enemies in exchange for honeydew secretions.

### Ant Activity Requirements

| Environmental Factor | Optimal Range for Ant Activity |
| --- | --- |
| Temperature | 20°C to 35°C |
| Humidity | < 80% (ants prefer drier conditions) |
| Rainfall | No rain (rain disrupts foraging trails) |

### Soil-Based Risk Modification

Soil type affects ant nest suitability, which indirectly influences mealybug survival:

| Soil Type | Clay Content | Risk Multiplier |
| --- | --- | --- |
| Sandy / Loam | < 35% | 1.2× (Higher Risk) |
| Heavy Clay | > 35% | 0.9× (Lower Risk) |

Mechanism: Sandy and loamy soils provide better drainage and easier tunneling for ant colonies. Heavy clay soils restrict ant movement and nest construction, reducing their ability to protect mealybug populations.

# 4. Custard Apple Phenology Integration

File: src/api/routes/pest.py

Risk assessment is dynamically adjusted based on the crop's developmental stage, as vulnerability to mealybug infestation varies throughout the growing season.

| Growth Stage | Risk Multiplier | Expected RVI | Vulnerability |
| --- | --- | --- | --- |
| Dormant / Post-Harvest | 0.2× | 0.25 | Very Low |
| Vegetative (New Leaves) | 0.3× | 0.45 | Low |
| Flowering | 0.7× | 0.60 | Moderate |
| Fruiting (Fruit Set) | 1.0× | 0.80 | CRITICAL |
| Harvesting | 1.0× | 0.70 | High |

Note: The RVI (Relative Vegetation Index) values help detect discrepancies between satellite observations and expected crop development, alerting to potential data staleness or crop stress.

# 5. Machine Learning Model: The "Student"

File: src/models/pest_forecasting_model.py

## 5.1 Model Architecture

The system uses an XGBoost gradient boosting model that learns from historical weather-pest correlations. The model file models/CustardApple_Blind_Model.joblib is loaded at runtime.

## 5.2 Feature Engineering

Rather than using hard-coded threshold rules (e.g., "if temperature >25°C for 7+ days"), the ML model learns critical patterns through engineered features:

| Feature | Description |
| --- | --- |
| temp_roll_mean_14 | 14-day rolling average temperature |
| temp_roll_mean_30 | 30-day rolling average temperature |
| humid_roll_mean_14 | 14-day rolling average humidity |
| Bio_Heat_Index | Temperature × Humidity interaction term |

Key Insight: The XGBoost model implicitly learns critical exposure durations (e.g., sustained heat) from the rolling window features, rather than requiring explicit "7+ day" rules to be programmed.

# 6. Data Fusion with Ensemble Kalman Filter

File: src/api/routes/pest.py → run_enkf_correction()

The EnKF combines three independent risk estimates into a single, probabilistically weighted prediction:

AI/Biological Risk Score: From XGBoost model or GDD fallback

Satellite RVI: Remote sensing vegetation health index

Manual RVI: Ground-truth observation from user

## 6.1 Kalman Gain Configuration

The filter applies a Kalman gain of K = 0.7, which heavily weights user observations over model predictions and satellite data. This design choice reflects:

Farmers' local expertise and direct field observations are highly reliable

Satellite data can lag or be obscured by cloud cover

The system acts as a "decision support" rather than full automation

# 7. Chemical Intervention Modeling

File: src/api/routes/pest.py → calculate_pipeline_risk()

## 7.1 Protection Window

When chemical sprays are applied, the system models their residual effectiveness:

Base Protection Period: 14 days from application

Rain Washout: Rainfall > 10mm reduces validity by 3 days per event

Heat Degradation: Accounted for in residual efficacy calculation

## 7.2 Risk Adjustment Formula

The final risk score incorporates chemical protection:

Final Risk = (EnKF Fused Score × Stage Multiplier × Soil Multiplier) × (1 - Protection Factor)

# 8. System Outputs & API Response

The prediction API returns a comprehensive JSON object containing:

## 8.1 Response Structure

| Field | Description |
| --- | --- |
| Mealy Bug | Final risk score (0-100%) |
| ai_score | Raw XGBoost model probability |
| soil_multiplier | Ant-soil symbiosis adjustment (0.9× or 1.2×) |
| enkf_fused_score | Score after Kalman Filter fusion |
| factors | Detailed breakdown: stage_mod, clay_pct, twin_brain_rvi, spray_protection |
| forecast | 7-day risk projection |
| twin_brain_status | Data source: "Satellite" or "Estimate" |

# 9. Model Validation & Current Limitations

## 9.1 Implemented Metrics

File: src/models/weather_pest_model.py

The codebase includes instrumentation for calculating standard ML metrics:

Accuracy

Precision, Recall, F1-Score

ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

## 9.2 Data Availability

Status: Pre-computed validation results are not present in the current deployment. The training_results/ directory contains metrics only for the spray scheduler RL model, not the pest prediction system.

## 9.3 What is NOT Modeled

Discrete Life Stages: The system does not track egg, larva, pupa, or adult stages separately; risk is a continuous scalar

Sexual vs Asexual Reproduction: The model treats population growth as uniform, not distinguishing reproductive modes

Natural Enemy Dynamics: Predators and parasitoids are not explicitly modeled (only ant protection is considered)

Honeydew & Sooty Mold: Secondary effects like mold growth are not tracked

# 10. Code Architecture & File Map

| File Path | Responsibility |
| --- | --- |
| src/models/biological_risk_model.py | BiologicalRiskModel class - GDD, biofix, rain washout |
| src/models/pest_forecasting_model.py | PestForecastingModel class - XGBoost ML predictions |
| src/models/weather_pest_model.py | Random Forest classifier for pest outbreak probability |
| src/api/routes/pest.py | API orchestration - run_enkf_correction(), calculate_pipeline_risk() |
| src/utils/config.py | PEST_THRESHOLDS, CHEMICAL_DATABASE, configuration constants |
| src/api/dependencies.py | Dependency injection for data sources |
| models/CustardApple_Blind_Model.joblib | Pre-trained XGBoost model binary |

# Conclusion

The FarmAI Mealybug Prediction System demonstrates a sophisticated integration of biological knowledge, machine learning, and real-time observation. By combining mechanistic pest development models with data-driven pattern recognition and adaptive filtering, the system provides farmers with actionable, contextually-aware risk assessments.

The "Twin Brain" architecture ensures robustness: if satellite data is unavailable or AI predictions fail, the biological model serves as a scientifically-grounded fallback. Conversely, the ML model can capture complex, non-linear relationships that mechanistic models might miss.

Future enhancements could include more granular lifecycle modeling, explicit natural enemy dynamics, and expanded validation datasets to further refine prediction accuracy.

