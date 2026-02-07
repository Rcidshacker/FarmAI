import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

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
// AI Assistant (Updated to support Context)
export const chatAssistant = (query, context = {}) => api.post('/assistant/chat', { query, context });

// Feedback
export const submitFeedback = (data) => api.post('/submit-treatment-feedback', data);

// Interventions
export const recordIntervention = (data) => api.post('/record-intervention', data);
export const resetIntervention = (userId) => api.post('/reset-intervention', { user_id: userId });

export default api;
