import axios from 'axios';

// Auto-detect API URL based on environment
// For development on mobile device, you may need to change 'localhost' to your computer's IP
// Get your computer IP: Windows: ipconfig, macOS/Linux: ifconfig
// Then update API_URL to 'http://YOUR_IP:8000'
let API_URL = 'http://localhost:8000';

// Check if running on mobile device
try {
    // If localStorage has a saved API URL (user configured), use it
    if (typeof window !== 'undefined' && localStorage.getItem('API_URL')) {
        API_URL = localStorage.getItem('API_URL');
    }
} catch (e) {
    // If localStorage access fails, continue with default
}

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Utility function to update API URL at runtime
export const setApiUrl = (url) => {
    // Ensure URL starts with http:// and has no trailing slash
    let cleanUrl = url.trim();
    if (!cleanUrl.startsWith('http')) {
        cleanUrl = `http://${cleanUrl}`;
    }
    if (cleanUrl.endsWith('/')) {
        cleanUrl = cleanUrl.slice(0, -1);
    }
    
    API_URL = cleanUrl;
    api.defaults.baseURL = cleanUrl;
    localStorage.setItem('API_URL', cleanUrl);
    console.log("API URL updated to:", cleanUrl);
};

export const getApiUrl = () => API_URL;

export const checkHealth = () => api.get('/health');
export const getSystemStats = () => api.get('/system-stats');

// Authentication (Phone + OTP)
export const sendOtp = (phone) => api.post('/auth/send-otp', { phone });
export const verifyOtp = (phone, otp, name = null) => api.post('/auth/verify-otp', { phone, otp, name });

// Disease Detection
export const detectDisease = (formData) => api.post('/detect-disease', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
});

// Pest Prediction
// Pest Prediction
export const predictPestRisk = (data) => api.post('/predict-pest-risk', data);

// Spray Schedule
export const createSpraySchedule = (locationName, daysAhead = 14) => api.post('/create-spray-schedule', {
    location: { name: locationName },
    days_ahead: daysAhead,
    current_pest_pressure: 0.3
});

// AI Assistant
export const chatAssistant = (query) => api.post('/assistant/chat', { query });

// Feedback
export const submitFeedback = (data) => api.post('/submit-treatment-feedback', data);

// Spray Management
export const recordSpray = (data) => api.post('/record-spray', data);
export const resetSprayHistory = (userId = "default_user") => api.delete(`/reset-spray-history?user_id=${userId}`);

export default api;
