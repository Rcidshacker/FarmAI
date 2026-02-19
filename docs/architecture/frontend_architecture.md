# Frontend Architecture Documentation

## Tech Stack

- **Framework:** React 19
- **Build Tool:** Vite
- **Styling:** TailwindCSS with custom utility components (e.g., `NeonButton`).
- **Routing:** React Router DOM v6/v7 (`BrowserRouter`).
- **HTTP Client:** Axios.
- **Icons:** Lucide React.
- **Internationalization:** `i18next` / `react-i18next`.

## Project Structure (`frontend/src/`)

### 1. App Entry (`App.jsx`)

- Handles routing definition.
- Includes `Navbar` (conditionally rendered, hidden on `/auth`).
- Implements a splash screen logic on startup.

### 2. Pages (`pages/`)

- **`DiseaseDetection.jsx`**:
  - **Core Feature:** Image capture/upload for disease analysis.
  - **State:** Manages camera stream, file preview, analysis steps animation, and results.
  - **Integration:** Calls `api.detectDisease` and `api.predictPestRisk` (for weather context).
  - **UI:** Custom camera interface with overlay, animated progress steps.
- **`PestRisk.jsx`**: Forecasting dashboard (Pest prediction visualizations).
- **`AIAssistant.jsx`**: Chat interface for the AI farming assistant.
- **`SpraySchedule.jsx`**: Management of spray calendars.
- **`AuthPage.jsx`**: Login and Registration handling.
- **`ProfilePage.jsx`**: User settings and farm profile.
- **`SplashScreen.jsx`**: Initial loading screen.

### 3. Services (`services/`)

- **`api.js`**: Centralized Axios instance.
  - Configures `baseURL` from env or default `localhost:8000`.
  - Exports typed functions for all backend endpoints (e.g., `detectDisease`, `predictPestRisk`, `sendOtp`).

### 4. Components (`components/`)

- **`Navbar.jsx`**: Main navigation.
- **`ui/NeonButton.jsx`**: Custom styled button component used across the app.

## Data Flow Pattern

1. **User Action:** User interacts with UI (e.g., captures photo).
2. **Component State:** React state updates (e.g., `setFile`, `setLoading`).
3. **Service Call:** Component calls function from `api.js` (e.g., `detectDisease(formData)`).
4. **API Request:** Axios sends HTTP request to FastAPI backend.
5. **Response Handling:**
    - Success: Updates state with result data (e.g., `setResult(response.data)`).
    - Error: Logs error or shows toast notification.

## Key Features

- **Camera Integration:** Uses `navigator.mediaDevices.getUserMedia` for in-browser camera access (optimized for mobile).
- **Responsive Design:** specific "safe-area" padding and mobile-first layouts.
- **Animation:** Uses `framer-motion` (inferred) or CSS transitions for smooth UX (e.g., scanning animation).
