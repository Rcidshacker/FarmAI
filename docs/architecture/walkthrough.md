# FarmAI Architecture Analysis Report

## Executive Summary

I have completed a comprehensive technical analysis of the FarmAI application. The system is a monolithic **FastAPI** backend coupled with a **React 19** frontend, using **SQLite** for persistence and **PyTorch/XGBoost** for intelligence.

## 1. System Components

- **Backend:** FastAPI application with a modular router structure.
- **Database:** SQLite (`farmai.db`) with inline schema management.
- **AI Core:**
  - **Vision:** PyTorch CNN for Disease Detection.
  - **Forecasting:** XGBoost for Pest Risk (Twin Brain).
  - **Chat:** Llama 3.3 (via OpenRouter) + Offline Fallback.
- **Frontend:** React + Vite, using specific "pages" for each feature and a centralized `api.js` service.

## 2. Key Findings

- **Twin Brain Logic:** The pest prediction system cleverly fuses satellite data (NDVI) with biological models using an Ensemble Kalman Filter (EnKF).
- **Offline First:** The AI Assistant has a dedicated offline mode using local JSON knowledge bases.
- **Performance Bottlenecks:** Critical external API calls (Satellite, LLM) are synchronous and blocking.
- **Security Risks:** API keys are hardcoded in source files.

## 3. Artifacts Created

I have generated the following detailed documentation for your review:

1. **[System Overview](file:///C:/Users/Ruchit/.gemini/antigravity/brain/13d1a679-3bab-41d4-9c60-2de4d52f3cf6/system_overview.md)**: High-level architecture.
2. **[API Documentation](file:///C:/Users/Ruchit/.gemini/antigravity/brain/13d1a679-3bab-41d4-9c60-2de4d52f3cf6/api_documentation.md)**: Endpoint details.
3. **[Database Schema](file:///C:/Users/Ruchit/.gemini/antigravity/brain/13d1a679-3bab-41d4-9c60-2de4d52f3cf6/database_schema.md)**: Tables and relationships.
4. **[ML Models](file:///C:/Users/Ruchit/.gemini/antigravity/brain/13d1a679-3bab-41d4-9c60-2de4d52f3cf6/ml_models_documentation.md)**: Deep dive into CNN and XGBoost pipelines.
5. **[Frontend Architecture](file:///C:/Users/Ruchit/.gemini/antigravity/brain/13d1a679-3bab-41d4-9c60-2de4d52f3cf6/frontend_architecture.md)**: UI structure and data flow.
6. **[Data Flow & Services](file:///C:/Users/Ruchit/.gemini/antigravity/brain/13d1a679-3bab-41d4-9c60-2de4d52f3cf6/data_flow_documentation.md)**: Tracing user journeys.
7. **[Services & Utilities](file:///C:/Users/Ruchit/.gemini/antigravity/brain/13d1a679-3bab-41d4-9c60-2de4d52f3cf6/services_and_utilities.md)**: Helper modules.
8. **[Upgrade Recommendations](file:///C:/Users/Ruchit/.gemini/antigravity/brain/13d1a679-3bab-41d4-9c60-2de4d52f3cf6/upgrade_recommendations.md)**: Actionable plan for improvements.

## 4. Next Steps

- Review the **Upgrade Recommendations** to prioritize fixes (Blocking I/O and Secrets should be first).
- Begin implementing the "Services Layer Pattern" refactor to clean up API routes.
