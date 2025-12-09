import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DiseaseDetection from './pages/DiseaseDetection';
import PestRisk from './pages/PestRisk';
import SpraySchedule from './pages/SpraySchedule';
import AIAssistant from './pages/AIAssistant';
import AuthPage from './pages/AuthPage';
import ProfilePage from './pages/ProfilePage';
import { Navbar } from './components/Navbar';

import { useLocation } from 'react-router-dom';

function Layout() {
  const location = useLocation();
  const isAuthPage = location.pathname === '/auth';

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Responsive Navbar - Hide on Auth Page */}
      {!isAuthPage && <Navbar />}

      {/* Main Content */}
      <main className={!isAuthPage ? "pb-24 md:pb-6" : ""}>
        <Routes>
          <Route path="/" element={<DiseaseDetection />} />
          <Route path="/auth" element={<AuthPage />} />
          <Route path="/pest-risk" element={<PestRisk />} />
          <Route path="/spray-schedule" element={<SpraySchedule />} />
          <Route path="/assistant" element={<AIAssistant />} />
          <Route path="/profile" element={<ProfilePage />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Layout />
    </Router>
  );
}

export default App;