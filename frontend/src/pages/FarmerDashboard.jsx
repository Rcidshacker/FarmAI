import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../styles/farmer-dashboard.css';

/**
 * Specialized Farmer Features Dashboard
 * Provides AI-driven insights and recommendations tailored to each farmer
 */

const FarmerDashboard = () => {
  const { t, i18n } = useTranslation();
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const [activeTab, setActiveTab] = useState('overview');
  const [farmData, setFarmData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [alerts, setAlerts] = useState([]);
  const [communityTips, setCommunityTips] = useState([]);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const userId = localStorage.getItem('user_id') || 'guest';

      // Try to load from API
      try {
        const analyticsRes = await fetch(`${API_BASE_URL}/api/assistant/farm-analytics`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, days_back: 30 })
        });

        if (analyticsRes.ok) {
          const analyticsData = await analyticsRes.json();
          setFarmData(analyticsData);
        }
      } catch (e) {
        console.warn('Could not fetch from API, using mock data');
      }

      // Try to load tips
      try {
        const tipsRes = await fetch(`${API_BASE_URL}/api/assistant/community-tips`);
        if (tipsRes.ok) {
          const tipsData = await tipsRes.json();
          setCommunityTips(tipsData.tips || []);
        }
      } catch (e) {
        console.warn('Could not fetch tips');
      }

      // Load mock alerts
      loadAlerts();
    } catch (error) {
      console.error('Error loading dashboard:', error);
      loadAlerts(); // Load mock data
    } finally {
      setLoading(false);
    }
  };

  const loadAlerts = () => {
    setAlerts([
      {
        id: 1,
        type: 'pest_risk',
        severity: 'high',
        title: 'High Mealy Bug Risk Detected',
        description: 'Weather conditions favor mealy bug outbreak in the next 5 days',
        recommendation: 'Consider preventive spray with Imidacloprid',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000)
      },
      {
        id: 2,
        type: 'disease_risk',
        severity: 'medium',
        title: 'Powder Mildew Alert',
        description: 'Low humidity with warm temperatures detected',
        recommendation: 'Monitor plants closely for early symptoms',
        timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000)
      },
      {
        id: 3,
        type: 'spray_reminder',
        severity: 'low',
        title: 'Spray Interval Reminder',
        description: "It's been 10 days since last spray",
        recommendation: 'Consider scheduling next spray cycle',
        timestamp: new Date(Date.now() - 12 * 60 * 60 * 1000)
      }
    ]);
  };

  const formatTimeAgo = (date) => {
    const seconds = Math.floor((new Date() - date) / 1000);
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  const renderOverview = () => (
    <div className="overview-section">
      <h3>{t('dashboard.overview') || 'Farm Overview'}</h3>

      <div className="cards-grid">
        <div className="info-card">
          <div className="card-header">{t('dashboard.threats') || 'Current Threats'}</div>
          <div className="card-value high-threat">3</div>
          <div className="card-description">Active alerts in your farm</div>
        </div>

        <div className="info-card">
          <div className="card-header">{t('dashboard.lastSpray') || 'Last Spray'}</div>
          <div className="card-value">10 days</div>
          <div className="card-description">Schedule next spray soon</div>
        </div>

        <div className="info-card">
          <div className="card-header">{t('dashboard.health') || 'Crop Health'}</div>
          <div className="card-value health-good">85%</div>
          <div className="card-description">Overall farm health score</div>
        </div>

        <div className="info-card">
          <div className="card-header">{t('dashboard.yield') || 'Yield Prediction'}</div>
          <div className="card-value">+12%</div>
          <div className="card-description">Expected yield vs. last season</div>
        </div>
      </div>
    </div>
  );

  const renderAlerts = () => (
    <div className="alerts-section">
      <h3>{t('dashboard.alerts') || 'Active Alerts'}</h3>

      <div className="alerts-list">
        {alerts.map(alert => (
          <div key={alert.id} className={`alert-item alert-${alert.severity}`}>
            <div className="alert-icon">
              {alert.type === 'pest_risk' && '🐛'}
              {alert.type === 'disease_risk' && '🌿'}
              {alert.type === 'spray_reminder' && '💧'}
            </div>

            <div className="alert-content">
              <h4>{alert.title}</h4>
              <p>{alert.description}</p>

              <div className="alert-recommendation">
                <strong>{t('dashboard.recommendation') || 'Recommendation'}:</strong>
                <p>{alert.recommendation}</p>
              </div>

              <small className="alert-time">{formatTimeAgo(alert.timestamp)}</small>
            </div>

            <button className="alert-action">{t('dashboard.action') || 'Action'}</button>
          </div>
        ))}
      </div>
    </div>
  );

  const renderAnalytics = () => {
    const data = [
      { name: 'Day 1', pestRisk: 45, diseaseRisk: 30 },
      { name: 'Day 2', pestRisk: 52, diseaseRisk: 35 },
      { name: 'Day 3', pestRisk: 48, diseaseRisk: 32 },
      { name: 'Day 4', pestRisk: 61, diseaseRisk: 40 },
      { name: 'Day 5', pestRisk: 55, diseaseRisk: 38 },
    ];

    return (
      <div className="analytics-section">
        <h3>{t('dashboard.analytics') || 'Risk Analytics'}</h3>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="pestRisk" stroke="#ff0000" />
            <Line type="monotone" dataKey="diseaseRisk" stroke="#00aa00" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderTips = () => (
    <div className="tips-section">
      <h3>{t('dashboard.tips') || 'Community Tips'}</h3>

      <div className="tips-list">
        {communityTips.length > 0 ? (
          communityTips.map((tip, idx) => (
            <div key={idx} className="tip-item">
              <p>{tip}</p>
            </div>
          ))
        ) : (
          <div className="no-tips">
            <p>No community tips available yet</p>
          </div>
        )}
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="farmer-dashboard">
        <div className="loading">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="farmer-dashboard">
      <div className="dashboard-header">
        <h1>{t('dashboard.title') || 'Farm Dashboard'}</h1>
        <select value={i18n.language} onChange={(e) => i18n.changeLanguage(e.target.value)} className="language-select">
          <option value="en">English</option>
          <option value="hi">हिन्दी</option>
          <option value="mr">मराठी</option>
        </select>
      </div>

      <div className="dashboard-tabs">
        <button
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          {t('dashboard.overview') || 'Overview'}
        </button>
        <button
          className={`tab-btn ${activeTab === 'alerts' ? 'active' : ''}`}
          onClick={() => setActiveTab('alerts')}
        >
          {t('dashboard.alerts') || 'Alerts'}
        </button>
        <button
          className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          {t('dashboard.analytics') || 'Analytics'}
        </button>
        <button
          className={`tab-btn ${activeTab === 'tips' ? 'active' : ''}`}
          onClick={() => setActiveTab('tips')}
        >
          {t('dashboard.tips') || 'Tips'}
        </button>
      </div>

      <div className="dashboard-content">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'alerts' && renderAlerts()}
        {activeTab === 'analytics' && renderAnalytics()}
        {activeTab === 'tips' && renderTips()}
      </div>
    </div>
  );
};

export default FarmerDashboard;
