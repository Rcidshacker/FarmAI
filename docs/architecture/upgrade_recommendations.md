# Upgrade Recommendations

## 1. Security & Configuration

- **[CRITICAL] Externalize Secrets:** API keys (e.g., AgroMonitoring key in `src/api/routes/pest.py`) are hardcoded. Move all keys to `.env` and load via `pydantic-settings`.
- **Input Validation:** Ensure all file uploads (images) are strictly validated for type and size (currently relies on `UploadFile` but explicit checks are better).

## 2. Performance & Scalability

- **[HIGH] Asynchronous I/O:**
  - `satellite_service.py` and `ai_assistant.py` use `requests` (synchronous) inside `async` API routes. This allows a single slow external API call to block the entire application.
  - **Fix:** Replace `requests` with `httpx` (asynchronous client).
- **Database Migrations:**
  - Current "migrations" are inline SQL checks in `db_manager.py`.
  - **Fix:** Implement **Alembic** for robust, version-controlled schema migrations.
- **Active Learning Storage:**
  - Currently stores metadata in a local JSON file (`feedback_metadata.json`). This is not thread-safe or scalable.
  - **Fix:** Move feedback storage to a dedicated SQL table (`feedback` table already exists, ensure it captures all metadata).

## 3. Architecture & Code Quality

- **Service Layer Pattern:**
  - Some logic is mixed into API routes (e.g., Heuristics in `disease.py`).
  - **Fix:** Move all business logic to dedicated Service classes (e.g., `DiseaseService`).
- **Dependency Injection:**
  - Services are instantiated directly in routes. Use `FastAPI.Depends` for better testing and modularity.

## 4. Frontend Improvements

- **State Management:**
  - Currently uses local `useState` and complex `useEffect` chains.
  - **Fix:** Adopt **TanStack Query (React Query)** for efficient data fetching, caching, and state management.
- **Error Handling:**
  - Add **Error Boundaries** to prevent the entire app from crashing on component errors.

## 5. ML Pipeline

- **Model Versioning:**
  - Models are loaded from specific paths.
  - **Fix:** Implement a model registry or simple versioning scheme to roll back if a new model underperforms.
