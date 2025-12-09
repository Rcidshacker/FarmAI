# FarmAI Frontend

A modern React-based frontend for the FarmAI Smart Agriculture System.

## Features

- **Disease Detection**: Upload leaf/fruit images for AI-powered disease identification
- **Pest Risk Forecasting**: Hybrid AI/Biological model for pest risk prediction
- **Smart Spray Schedule**: RL agent-optimized spray calendar with weather integration
- **AI Assistant**: Knowledge base-powered chat for farm queries

## Tech Stack

- **React 19** - UI framework
- **React Router** - Navigation
- **Axios** - API communication
- **Tailwind CSS** - Styling (via CDN)
- **Lucide React** - Icons
- **Vite** - Build tool

## Project Structure

```
src/
├── pages/
│   ├── DiseaseDetection.jsx  # Image upload & disease analysis
│   ├── PestRisk.jsx           # Pest risk prediction dashboard
│   ├── SpraySchedule.jsx      # RL-optimized spray calendar
│   └── AIAssistant.jsx        # Chat interface
├── services/
│   └── api.js                 # Backend API integration
├── App.jsx                    # Main app with routing
├── main.jsx                   # Entry point
└── index.css                  # Global styles
```

## Setup & Installation

### Prerequisites
- Node.js 16+ installed
- Backend FastAPI server running on `http://localhost:8000`

### Installation Steps

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Access the application**:
   Open your browser to `http://localhost:5173`

## API Configuration

The frontend connects to the backend at `http://localhost:8000`. To change this:

Edit `src/services/api.js`:
```javascript
const API_URL = 'http://your-backend-url:port';
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Backend Integration

This frontend requires the following backend endpoints:

- `POST /detect-disease` - Disease detection from images
- `POST /predict-pest-risk` - Pest risk prediction
- `POST /create-spray-schedule` - Generate spray schedule
- `POST /assistant/chat` - AI assistant queries
- `GET /health` - Health check
- `GET /system-stats` - System statistics

## Features by Page

### 1. Disease Detection (`/`)
- Image upload with preview
- Wind speed and fruit density inputs
- Confidence-based results
- Rubbing risk warnings
- Alternative disease possibilities

### 2. Pest Risk (`/risk`)
- Environmental context display
- Real-time weather integration
- Hybrid model predictions
- Risk visualization with color coding
- Detailed breakdown for Mealy Bug (AI + Biological scores)

### 3. Spray Schedule (`/schedule`)
- 30-day optimized calendar
- Weather-aware recommendations
- Cost and yield analysis
- Alert system for critical conditions
- Spray quality indicators

### 4. AI Assistant (`/chat`)
- Real-time chat interface
- Knowledge base integration
- Context-aware responses
- Query history

## Styling

The app uses Tailwind CSS via CDN for rapid development. Key design features:

- **Green theme** - Agricultural focus
- **Responsive layout** - Mobile-first design
- **Card-based UI** - Clean information hierarchy
- **Smooth animations** - Enhanced UX
- **Custom scrollbars** - Polished appearance

## Development Notes

### Hardcoded Values
- Location is set to "Pune" for pest risk prediction (matches backend logic)
- Spray schedule requests 30 days ahead
- Default pest pressure: 0.3

### Error Handling
All API calls include try-catch blocks with user-friendly error messages.

### State Management
Uses React hooks (useState, useEffect) for local state management.

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Troubleshooting

### Backend Connection Issues
- Ensure FastAPI server is running on port 8000
- Check CORS settings in backend
- Verify API endpoints match

### Build Issues
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Clear Vite cache: `rm -rf .vite`

### Styling Issues
- Ensure Tailwind CDN script is in `<head>` of index.html
- Check browser console for CSS errors

## Future Enhancements

- [ ] Add location selector (currently hardcoded to Pune)
- [ ] Implement user authentication
- [ ] Add data visualization charts
- [ ] Enable offline mode with service workers
- [ ] Add multi-language support
- [ ] Implement push notifications for alerts

## License

Part of the FarmAI Smart Agriculture System
