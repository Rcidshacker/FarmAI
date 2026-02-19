# API Endpoints Documentation

## Authentication Routes (`src/api/routes/auth.py`)

### POST /auth/send-otp

- **Purpose:** initiate login/registration via phone number.
- **Input:** `{ "phone": "string" }`
- **Output:** `{ "success": true, "message": "OTP sent", "debug_otp": "123456" }`
- **Note:** Currently mocks OTP sending. Returns fixed OTP "123456".

### POST /auth/verify-otp

- **Purpose:** Verify OTP and return auth token/user profile.
- **Input:** `{ "phone": "string", "otp": "string", "name": "optional" }`
- **Output:** `{ "success": true, "user": {...}, "token": "..." }`
- **Special Logic:**
  - Phone `1234567890` triggers a "Demo Mode" creating a dummy user "Rachit" with random land area.
  - Other numbers create a standard new user if not exists.

### POST /user/profile

- **Purpose:** Update user profile details.
- **Input:** `UserProfile` object (location, soil_type, variety, etc.)
- **Database:** Updates `users` table.

---

## Disease Detection Routes (`src/api/routes/disease.py`)

### POST /detect-disease

- **Purpose:** Analyze crop image for disease identification.
- **Input:**
  - `file`: Image (Multipart Form Data)
  - `wind_speed`: float (Optional, default 0.0)
  - `fruit_density`: string (Optional, default "Medium")
- **Processing Flow:**
  1. **Image Loading:** Converts uploaded file to RGB array.
  2. **Inference:** Calls `DiseaseClassifier.predict()`.
  3. **Heuristic Check:** If `wind_speed > 20` and `fruit_density == "High"`, increases probability of "Physical Damage" class.
- **Output:**

  ```json
  {
      "disease": "Leaf Blight",
      "confidence": 0.87,
      "quick_action": "Apply fungicide immediately",
      "rubbing_risk_warning": "High risk of Fruit Rubbing..."
  }
  ```

---

## Pest Prediction Routes (`src/api/routes/pest.py`)

### POST /predict-pest-risk

- **Purpose:** Predict pest outbreak risk (specifically Mealy Bug) using a "Twin Brain" approach (Satellite + Manual Input).
- **Input:** `{ "location": {...}, "crop_stage": "Fruiting", "manual_rvi": 0.5, "api_key": "..." }`
- **Logic:**
  1. **Satellite Data:** Fetches NDVI/RVI from satellite service (if available).
  2. **Soil Analysis:** Fetches soil clay percentage (affects Mealy Bug habitat).
  3. **Weather Analysis:** Fetches current and forecast weather.
  4. **EnKF Correction:** Fuses biological model risk with user observations (Kalman Filter).
  5. **Intervention Check:** Reduces risk if recent sprays are recorded in DB.
- **Output:** Detailed risk analysis including "Twin Brain" status and factors.

### POST /forecast/risk

- **Purpose:** 14-day pest risk forecast.
- **Logic:** Falls back to Biological Engine (GDD - Growing Degree Days) if AI Model fails.

---

## Weather & Environment Routes (`src/api/routes/weather.py`)

### GET /weather-forecast/{location}

- **Purpose:** Get 7-day weather forecast.
- **Source:** External Weather API via `DataIntegrator`.

### GET /soil/info

- **Purpose:** Get soil profile and fertilizer recommendations based on location.
- **Output:** `{ "soil_profile": {...}, "recommendations": [...] }`

### GET /satellite/data

- **Purpose:** Get vegetation indices (NDVI) for a specific lat/lon.

---

## AI Assistant Routes (`src/api/routes/assistant.py`)

### POST /assistant/chat

- **Purpose:** General AI queries about farming.
- **Context Injection:** Automatically injects:
  - Current Weather
  - Soil Profile
  - User Location
- **Input:** `{ "query": "What should I spray for mealy bugs?", "context": {...} }`

### GET /research-papers/{disease}

- **Purpose:** Search for latest academic papers on specific diseases.
- **Source:** External research paper fetcher.

---

## System Routes (`src/api/routes/system.py`)

*(Assumed based on file existence)*

- **GET /health** or **GET /status**: System health check.
