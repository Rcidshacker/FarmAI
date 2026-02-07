import React from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthForm } from '../components/ui/AuthForm';

import logo from '../assets/logo.png';

export default function AuthPage() {
    const navigate = useNavigate();

    const handleSuccess = (user) => {
        // In a real app, you'd save the auth token here
        console.log('Logged in:', user);
        navigate('/pest-risk');
    };

    return (
        <div className="min-h-screen bg-gray-50 flex items-start pt-20 md:pt-0 md:items-center justify-center p-4">
            <div className="w-full max-w-md">
                {/* Logo or Brand Element */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center mb-4">
                        <img
                            src={logo}
                            alt="FarmAI Logo"
                            className="w-24 h-24 object-contain drop-shadow-xl hover:scale-110 transition-transform duration-300"
                        />
                    </div>
                    <h1 className="text-3xl font-bold text-gray-900 tracking-tight">FarmAI</h1>
                    <p className="text-gray-500 mt-2">Smart Decisions for Better Yields</p>
                </div>

                {/* Auth Form Component */}
                <AuthForm onSuccess={handleSuccess} />
            </div>
        </div>
    );
}
