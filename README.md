# FarmAI - Custard Apple Pest Management System

FarmAI is an advanced, AI-powered platform designed to assist custard apple farmers in managing pest outbreaks, detecting diseases, and optimizing spray schedules. The system integrates real-time weather data, satellite imagery, and biological models to provide accurate, location-based recommendations.

## 🌟 Key Features

* **Disease Detection**: utilizing PyTorch-based CNNs to detect diseases from leaf images.
* **Pest Prediction**: sophisticated forecasting models (XGBoost) combined with biological risk assessments.
* **Automated Spray Scheduling**: Reinforcement Learning (RL) based scheduler for optimal chemical usage.
* **Digital Twin**: Integration of manual observations with satellite indices (NDVI) for precise canopy estimation.
* **Research Integration**: Automatic retrieval of relevant research papers for disease management.

## 🏗️ Architecture

The project is divided into two main components:

* **Backend**: A FastAPI-based server handling AI inference, data processing, and API endpoints.
* **Frontend**: A modern React application (Vite + TailwindCSS) providing the user interface.

## 🚀 Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

* **Python**: Version 3.10 or higher
* **Node.js**: Version 18 or higher
* **Git**: For version control

---

### 1. Backend Setup

The backend powers the AI and API services.

1. **Navigate to the project root:**

    ```bash
    cd FarmAI
    ```

2. **Create a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    * **Windows:**

        ```bash
        .\venv\Scripts\activate
        ```

    * **macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

4. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: If you have a GPU, ensure you install the CUDA version of PyTorch manually if needed.*

5. **Run the Server:**

    ```bash
    uvicorn src.api.app_v2:app --reload
    ```

    The API will be available at `http://localhost:8000`.
    You can view the API documentation at `http://localhost:8000/docs`.

---

### 2. Frontend Setup

The frontend provides the interactive dashboard.

1. **Navigate to the frontend directory:**

    ```bash
    cd frontend
    ```

2. **Install Node dependencies:**

    ```bash
    npm install
    ```

3. **Run the Development Server:**

    ```bash
    npm run dev
    ```

    The application will typically run at `http://localhost:5173` (check the console output).

---

## 📂 Project Structure

```
FarmAI/
├── builds/                 # APK/AAB Build Artifacts
├── frontend/               # React Frontend Code
├── scripts/                # Utility Scripts
│   └── data_extraction/    # Data Processing Scripts
├── src/                    # Backend Source Code
│   ├── api/                # FastAPI Endpoints and App
│   ├── automation/         # Spray Scheduling Logic
│   ├── models/             # PyTorch and XGBoost Models
│   ├── services/           # Business Logic Services
│   └── data_sources/       # Weather & Satellite Integration
├── knowledge_base/         # Research Papers / Docs
├── requirements.txt        # Python Dependencies
├── build-apk.bat           # Android Build Script (Windows)
└── setup-android.bat       # Android Environment Setup (Windows)
```

## 🤝 Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
