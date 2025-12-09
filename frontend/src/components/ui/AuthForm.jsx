import React, { useState } from 'react';
import { cn } from "../../lib/utils";
import {
    Phone,
    Shield,
    AlertTriangle,
    Loader2,
    CheckCircle2,
    ArrowRight
} from 'lucide-react';
import { NeonButton } from '../ui/NeonButton';
import { sendOtp, verifyOtp } from '../../services/api';

export function AuthForm({
    onSuccess,
    className,
}) {
    // State: 'phone' | 'otp'
    const [step, setStep] = useState('phone');
    const [isLoading, setIsLoading] = useState(false);
    const [phone, setPhone] = useState('');
    const [otp, setOtp] = useState('');
    const [error, setError] = useState('');
    const [successMessage, setSuccessMessage] = useState('');

    // Handle Phone Submit
    const handlePhoneSubmit = async (e) => {
        e.preventDefault();
        setError('');

        // Basic validation
        if (!phone || phone.length < 10) {
            setError('Please enter a valid phone number');
            return;
        }

        setIsLoading(true);
        try {
            await sendOtp(phone);
            setSuccessMessage(`OTP sent to ${phone}`);
            setStep('otp');
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || 'Failed to send OTP. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    // Handle OTP Submit
    const handleOtpSubmit = async (e) => {
        e.preventDefault();
        setError('');

        if (!otp || otp.length < 4) {
            setError('Please enter valid OTP');
            return;
        }

        setIsLoading(true);
        try {
            const response = await verifyOtp(phone, otp);
            const { user, token } = response.data;

            // Save session
            localStorage.setItem('token', token);
            localStorage.setItem('user', JSON.stringify(user));

            setSuccessMessage('Login Successful!');
            onSuccess?.(user);

        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || 'Invalid OTP. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={cn("bg-white p-6 md:p-8 rounded-3xl shadow-xl border border-gray-100", className)}>

            {/* Header */}
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold mb-2 text-gray-900">
                    Welcome to FarmAI
                </h2>
                <p className="text-gray-500 text-sm">
                    {step === 'phone'
                        ? 'Enter your mobile number to continue'
                        : 'Enter the verification code sent to your phone'}
                </p>
            </div>

            {/* Error Message */}
            {error && (
                <div className="mb-6 p-3 bg-red-50 border border-red-200 rounded-xl flex items-center gap-2 animate-in fade-in-0 slide-in-from-top-2">
                    <AlertTriangle className="h-4 w-4 text-red-500 shrink-0" />
                    <span className="text-red-700 text-sm">{error}</span>
                </div>
            )}

            {/* Success Message */}
            {successMessage && !error && (
                <div className="mb-6 p-3 bg-green-50 border border-green-200 rounded-xl flex items-center gap-2 animate-in fade-in-0 slide-in-from-top-2">
                    <CheckCircle2 className="h-4 w-4 text-green-600 shrink-0" />
                    <span className="text-green-700 text-sm font-medium">{successMessage}</span>
                </div>
            )}

            {/* STEP 1: PHONE INPUT */}
            {step === 'phone' && (
                <form onSubmit={handlePhoneSubmit} className="space-y-6 animate-in fade-in-50 slide-in-from-right-4">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-700 ml-1">Mobile Number</label>
                        <div className="relative">
                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                <Phone className="h-5 w-5 text-gray-400" />
                            </div>
                            <input
                                type="tel"
                                placeholder="1234567890"
                                value={phone}
                                onChange={(e) => {
                                    const val = e.target.value.replace(/\D/g, ''); // Numeric only
                                    setPhone(val);
                                }}
                                className="w-full pl-10 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-xl placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500/20 text-lg tracking-wide transition-all"
                                maxLength={10}
                                autoFocus
                            />
                        </div>
                    </div>

                    <NeonButton
                        type="submit"
                        disabled={isLoading || phone.length < 10}
                        variant="solid"
                        className="w-full shadow-lg shadow-green-600/20"
                    >
                        <span className="flex items-center justify-center gap-2">
                            {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : (
                                <>
                                    Get OTP <ArrowRight className="h-4 w-4" />
                                </>
                            )}
                        </span>
                    </NeonButton>

                    <p className="text-xs text-center text-gray-400 mt-4">
                        By continuing, you agree to our Terms & Privacy Policy.
                    </p>
                </form>
            )}

            {/* STEP 2: OTP INPUT */}
            {step === 'otp' && (
                <form onSubmit={handleOtpSubmit} className="space-y-6 animate-in fade-in-50 slide-in-from-right-4">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-700 ml-1">One Time Password (OTP)</label>
                        <div className="relative">
                            <Shield className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400 pointer-events-none" />
                            <input
                                type="text"
                                placeholder="• • • • • •"
                                value={otp}
                                onChange={(e) => setOtp(e.target.value)}
                                className="w-full pl-10 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-xl placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500/20 text-center text-2xl tracking-[0.5em] font-mono transition-all"
                                maxLength={6}
                                autoFocus
                            />
                        </div>
                        <div className="flex justify-end">
                            <button
                                type="button"
                                onClick={() => setStep('phone')}
                                className="text-xs text-green-600 font-medium hover:underline"
                            >
                                Change Phone Number?
                            </button>
                        </div>
                    </div>

                    <NeonButton
                        type="submit"
                        disabled={isLoading || otp.length < 4}
                        variant="solid"
                        className="w-full shadow-lg shadow-green-600/20"
                    >
                        <span className="flex items-center justify-center gap-2">
                            {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : "Verify & Login"}
                        </span>
                    </NeonButton>

                    <p className="text-xs text-center text-gray-400">
                        Did not receive code? <button type="button" className="text-green-600 font-medium hover:underline">Resend</button>
                    </p>
                </form>
            )}
        </div>
    );
}
