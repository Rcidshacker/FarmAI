import React, { useState } from 'react';
import { detectDisease, predictPestRisk } from '../services/api';
import {
    Camera,
    Upload,
    AlertTriangle,
    CheckCircle,
    Wind,
    Layers,
    X,
    ScanLine,
    Activity,
    Loader2
} from 'lucide-react';
import { NeonButton } from '../components/ui/NeonButton';
import { cn } from "../lib/utils";
import { useEffect, useRef } from 'react';
import { Camera as CapacitorCamera, CameraResultType, CameraSource } from '@capacitor/camera';
import { useTranslation } from 'react-i18next';

const DiseaseDetection = () => {
    const { t } = useTranslation();
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [windSpeed, setWindSpeed] = useState(5.0);
    const [fruitDensity, setFruitDensity] = useState('Medium');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [analysisStep, setAnalysisStep] = useState(0);
    const [isButtonVisible, setIsButtonVisible] = useState(true);

    // Scroll Direction Logic
    useEffect(() => {
        let lastScrollY = window.scrollY;

        const handleScroll = () => {
            const currentScrollY = window.scrollY;

            // Logic: Scroll DOWN -> Show Button (return true). Scroll UP -> Hide Button (return false)
            // User request: "when scroll down the button scroll up and shows itself and when scroll up the whole page the button scroll down nd hides"
            // This is inverse of standard "hide on scroll down".

            if (currentScrollY > lastScrollY) {
                // Scrolling DOWN
                setIsButtonVisible(true);
            } else {
                // Scrolling UP
                setIsButtonVisible(false);
            }
            lastScrollY = currentScrollY;
        };

        window.addEventListener("scroll", handleScroll, { passive: true });
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    // Camera Refs & State
    const videoRef = React.useRef(null);
    const canvasRef = React.useRef(null);
    const fileInputRef = React.useRef(null);
    const [isCameraActive, setIsCameraActive] = useState(false);
    const [cameraError, setCameraError] = useState(null);

    const startCamera = async () => {
        setCameraError(null);
        try {
            // Try Capacitor camera first (for mobile)
            if (window.Capacitor && window.Capacitor.isNativePlatform()) {
                const image = await CapacitorCamera.getPhoto({
                    quality: 90,
                    allowEditing: false,
                    resultType: CameraResultType.Uri,
                    source: CameraSource.Camera,
                    correctOrientation: true,
                });

                // Convert URI to Blob
                const response = await fetch(image.webPath);
                const blob = await response.blob();
                const fileName = image.path || "captured_image.jpg";
                const file = new File([blob], fileName, { type: blob.type });
                
                setFile(file);
                setPreview(image.webPath);
                setResult(null);
            } else {
                // Web fallback: Use getUserMedia
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment' } // Prefer back camera on mobile
                });
                setIsCameraActive(true);
                // Small delay to ensure ref is mounted
                setTimeout(() => {
                    if (videoRef.current) {
                        videoRef.current.srcObject = stream;
                    }
                }, 100);
            }
        } catch (err) {
            console.error("Camera Access Error:", err);
            setCameraError("Camera permission denied or not available. Please upload a file instead.");
        }
    };

    const stopCamera = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        setIsCameraActive(false);
    };

    const capturePhoto = () => {
        if (videoRef.current && canvasRef.current) {
            const video = videoRef.current;
            const canvas = canvasRef.current;

            // Match canvas size to video stream
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // Draw
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert to Blob/File
            canvas.toBlob((blob) => {
                const file = new File([blob], "captured_image.jpg", { type: "image/jpeg" });
                setFile(file);
                // Create preview URL
                setPreview(URL.createObjectURL(file));
                setResult(null);
                stopCamera(); // Close camera after capture
            }, 'image/jpeg', 0.95);
        }
    };

    // Auto-fetch weather data for background context
    React.useEffect(() => {
        const fetchWeather = async () => {
            try {
                // Using existing API to get current weather
                const response = await predictPestRisk({
                    location: { name: "Pune" },
                    use_realtime: true
                });

                if (response.data?.current_weather) {
                    // Prioritize wind_speed from API, fallback to default
                    const speed = response.data.current_weather.wind_speed || response.data.current_weather.windspeed || 5.0;
                    setWindSpeed(parseFloat(speed).toFixed(1));
                }
            } catch (error) {
                console.warn("Could not fetch live weather for wind speed", error);
            }
        };
        fetchWeather();
    }, []);

    const handleFileChange = (e) => {
        const selected = e.target.files[0];
        if (selected) {
            setFile(selected);
            setPreview(URL.createObjectURL(selected));
            setResult(null);
        }
    };

    const handleRemoveImage = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) return;

        setLoading(true);
        setResult(null);
        setAnalysisStep(0);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('wind_speed', windSpeed);
        formData.append('fruit_density', fruitDensity);

        // Run animation sequence in parallel with API call
        const animationSequence = async () => {
            setAnalysisStep(1);
            await new Promise(r => setTimeout(r, 1000));
            setAnalysisStep(2);
            await new Promise(r => setTimeout(r, 1000));
            setAnalysisStep(3);
            await new Promise(r => setTimeout(r, 1000));
            setAnalysisStep(4);
            await new Promise(r => setTimeout(r, 500)); // Short final pause
        };

        try {
            // Wait for BOTH the API and the Animation to finish
            const [response] = await Promise.all([
                detectDisease(formData),
                animationSequence()
            ]);
            setResult(response.data);
        } catch (error) {
            console.error("Error detecting disease", error);
            // In a real app, show a toast here
        } finally {
            setLoading(false);
            setAnalysisStep(0);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50/50 p-6 pb-24">
            <div className="max-w-6xl mx-auto">
                <div className="text-center mb-10">
                    <h1 className="text-3xl font-bold text-green-800 mb-2">{t('diseaseDetection.title')}</h1>
                    <p className="text-gray-500 max-w-lg mx-auto">
                        {t('diseaseDetection.description')}
                    </p>
                </div>

                <div className="grid md:grid-cols-2 gap-8 items-start">

                    {/* Input Section */}
                    <div className="bg-white rounded-3xl shadow-sm border border-green-100 overflow-hidden">
                        <div className="p-1 bg-green-50/50 border-b border-green-100">
                            <div className="flex items-center gap-2 px-4 py-2 text-green-700 font-medium text-sm">
                                <Camera className="h-4 w-4" />
                                <span>{t('diseaseDetection.imageAnalysis')}</span>
                            </div>
                        </div>

                        <div className="p-6 md:p-8 space-y-6">
                            <form onSubmit={handleSubmit}>
                                {/* Camera & Upload Section */}
                                <div className="group relative">
                                    {preview ? (
                                        <div className="relative rounded-2xl overflow-hidden bg-gray-900 aspect-video shadow-inner">
                                            <img
                                                src={preview}
                                                alt="Preview"
                                                className="w-full h-full object-contain"
                                            />
                                            <button
                                                type="button"
                                                onClick={handleRemoveImage}
                                                className="absolute top-3 right-3 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors backdrop-blur-sm shadow-sm"
                                            >
                                                <X className="h-4 w-4" />
                                            </button>
                                        </div>
                                    ) : isCameraActive ? (
                                        <div className="relative rounded-2xl overflow-hidden bg-black aspect-[4/3] shadow-inner">
                                            <video
                                                ref={videoRef}
                                                autoPlay
                                                playsInline
                                                className="w-full h-full object-cover"
                                            />
                                            {/* Camera Overlay */}
                                            <div className="absolute inset-0 border-[3px] border-white/20 m-6 rounded-xl pointer-events-none">
                                                <div className="absolute top-0 left-0 w-4 h-4 border-t-4 border-l-4 border-white/80 -mt-1 -ml-1"></div>
                                                <div className="absolute top-0 right-0 w-4 h-4 border-t-4 border-r-4 border-white/80 -mt-1 -mr-1"></div>
                                                <div className="absolute bottom-0 left-0 w-4 h-4 border-b-4 border-l-4 border-white/80 -mb-1 -ml-1"></div>
                                                <div className="absolute bottom-0 right-0 w-4 h-4 border-b-4 border-r-4 border-white/80 -mb-1 -mr-1"></div>
                                            </div>

                                            {/* Camera Controls */}
                                            <div className="absolute bottom-4 left-0 right-0 flex justify-center items-center gap-6 z-20">
                                                <button
                                                    type="button"
                                                    onClick={stopCamera}
                                                    className="p-3 bg-red-500/80 hover:bg-red-600 text-white rounded-full backdrop-blur-sm transition-transform active:scale-95"
                                                    title={t('diseaseDetection.cancelCamera')}
                                                >
                                                    <X className="h-6 w-6" />
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={capturePhoto}
                                                    className="h-16 w-16 bg-white rounded-full border-4 border-gray-300 ring-4 ring-white/50 active:scale-90 transition-all flex items-center justify-center shadow-lg"
                                                    title={t('diseaseDetection.takePhoto')}
                                                >
                                                    <div className="h-12 w-12 bg-gray-100 rounded-full border-2 border-gray-300"></div>
                                                </button>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="relative">
                                            {/* Hidden File Input */}
                                            <input
                                                type="file"
                                                ref={fileInputRef}
                                                onChange={handleFileChange}
                                                accept="image/*"
                                                className="hidden"
                                            />

                                            <div className="border-2 border-dashed border-green-200 rounded-3xl p-8 text-center bg-green-50/20 flex flex-col items-center justify-center min-h-[300px]">
                                                {/* Primary Action: Camera */}
                                                <div className="mb-6">
                                                    <button
                                                        type="button"
                                                        onClick={startCamera}
                                                        className="h-20 w-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4 hover:bg-green-200 hover:scale-105 transition-all duration-300 shadow-sm"
                                                    >
                                                        <Camera className="h-10 w-10 text-green-700" />
                                                    </button>
                                                    <h3 className="text-xl font-bold text-gray-800 mb-1">{t('diseaseDetection.takePhotoTitle')}</h3>
                                                    <p className="text-sm text-gray-500">{t('diseaseDetection.takePhotoDesc')}</p>
                                                </div>

                                                <div className="flex items-center gap-3 w-full max-w-xs mb-6">
                                                    <div className="h-px bg-green-200 flex-1"></div>
                                                    <span className="text-xs font-semibold text-green-400 uppercase tracking-widest">{t('diseaseDetection.or')}</span>
                                                    <div className="h-px bg-green-200 flex-1"></div>
                                                </div>

                                                {/* Secondary Action: Upload */}
                                                <button
                                                    type="button"
                                                    onClick={() => fileInputRef.current?.click()}
                                                    className="w-full flex flex-col items-center justify-center p-4 bg-white border-2 border-dashed border-green-200 rounded-2xl hover:bg-green-50 hover:border-green-300 transition-all shadow-sm active:scale-95"
                                                >
                                                    <Upload className="h-6 w-6 text-green-600 mb-2" />
                                                    <span className="text-green-800 font-bold text-sm">{t('diseaseDetection.uploadGallery')}</span>
                                                </button>

                                                {cameraError && (
                                                    <div className="mt-4 p-3 bg-red-50 text-red-600 text-xs rounded-lg flex items-center gap-2 max-w-xs mx-auto">
                                                        <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                                                        {cameraError}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                    <canvas ref={canvasRef} className="hidden" />
                                </div>

                                {/* Parameters */}
                                <div className="grid grid-cols-2 gap-4 mt-6">
                                    <div className="space-y-2">
                                        <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider ml-1">
                                            {t('diseaseDetection.windSpeed')}
                                        </label>
                                        <div className="relative group">
                                            <Wind className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-green-600" />
                                            <div className="w-full pl-10 pr-12 py-3 bg-green-50/50 border border-green-200 rounded-xl font-medium text-gray-700 cursor-not-allowed flex items-center h-[46px]">
                                                {windSpeed}
                                            </div>
                                            <span className="absolute right-4 top-1/2 -translate-y-1/2 text-sm text-gray-500 font-medium">km/h</span>
                                        </div>
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider ml-1">
                                            {t('diseaseDetection.density')}
                                        </label>
                                        <div className="relative group">
                                            <Layers className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400 group-focus-within:text-green-500 transition-colors" />
                                            <select
                                                value={fruitDensity}
                                                onChange={(e) => setFruitDensity(e.target.value)}
                                                className="w-full pl-10 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500/20 focus:border-green-500 transition-all font-medium text-gray-700 appearance-none cursor-pointer"
                                            >
                                                <option>{t('diseaseDetection.low')}</option>
                                                <option>{t('diseaseDetection.medium')}</option>
                                                <option>{t('diseaseDetection.high')}</option>
                                            </select>
                                            <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none">
                                                <div className="border-t-4 border-l-4 border-transparent border-t-gray-400"></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className={cn(
                                    "mt-8 md:static fixed bottom-[80px] left-0 right-0 p-4 bg-white/90 backdrop-blur-md border-t border-gray-200 md:bg-transparent md:border-none md:p-0 z-30 pb-4 md:pb-0 safe-area-pb transition-transform duration-300 transform",
                                    isButtonVisible ? "translate-y-0" : "translate-y-[200%] md:translate-y-0"
                                )}>
                                    <NeonButton
                                        type="submit"
                                        disabled={loading || !file}
                                        variant="solid"
                                        className="w-full h-14 md:h-12 shadow-lg shadow-green-600/20 text-lg md:text-base rounded-2xl md:rounded-xl"
                                    >
                                        {loading ? (
                                            <div className="flex items-center gap-2 justify-center">
                                                <Loader2 className="h-5 w-5 animate-spin" />
                                                <span>{t('diseaseDetection.analyzing')}</span>
                                            </div>
                                        ) : (
                                            <div className="flex items-center gap-2 justify-center">
                                                <Activity className="h-5 w-5" />
                                                <span>{t('diseaseDetection.analyzeImage')}</span>
                                            </div>
                                        )}
                                    </NeonButton>
                                </div>
                            </form>
                        </div>
                    </div>

                    {/* Output Section */}
                    <div className="space-y-6">
                        {/* Output Section with Animation */}
                        {loading ? (
                            <div className="bg-white rounded-3xl shadow-lg border border-green-100 overflow-hidden flex flex-col items-center justify-center p-8 min-h-[400px]">
                                <div className="max-w-xs w-full space-y-6">
                                    <h3 className="text-xl font-bold text-gray-800 text-center mb-6">{t('diseaseDetection.analyzingHealth')}</h3>

                                    {/* Checklist Items */}
                                    <div className="space-y-4">
                                        {/* Step 1: Upload */}
                                        <div className="flex items-center gap-3 transition-all duration-300">
                                            {analysisStep > 1 ? (
                                                <div className="bg-green-100 p-1 rounded-full"><CheckCircle className="h-5 w-5 text-green-600" /></div>
                                            ) : (
                                                <Loader2 className="h-6 w-6 text-green-600 animate-spin" />
                                            )}
                                            <span className={cn("text-sm font-medium", analysisStep > 1 ? "text-gray-900" : "text-gray-500")}>
                                                {t('diseaseDetection.processingImage')}
                                            </span>
                                        </div>

                                        {/* Step 2: Scanning */}
                                        <div className={`flex items-center gap-3 transition-opacity duration-500 ${analysisStep >= 2 ? 'opacity-100' : 'opacity-40'}`}>
                                            {analysisStep > 2 ? (
                                                <div className="bg-green-100 p-1 rounded-full"><CheckCircle className="h-5 w-5 text-green-600" /></div>
                                            ) : analysisStep === 2 ? (
                                                <Loader2 className="h-6 w-6 text-green-600 animate-spin" />
                                            ) : (
                                                <div className="w-6 h-6 rounded-full border-2 border-gray-200"></div>
                                            )}
                                            <span className={cn("text-sm font-medium", analysisStep > 2 ? "text-gray-900" : "text-gray-500")}>
                                                {t('diseaseDetection.scanning')}
                                            </span>
                                        </div>

                                        {/* Step 3: Weather */}
                                        <div className={`flex items-center gap-3 transition-opacity duration-500 ${analysisStep >= 3 ? 'opacity-100' : 'opacity-40'}`}>
                                            {analysisStep > 3 ? (
                                                <div className="bg-green-100 p-1 rounded-full"><CheckCircle className="h-5 w-5 text-green-600" /></div>
                                            ) : analysisStep === 3 ? (
                                                <Loader2 className="h-6 w-6 text-green-600 animate-spin" />
                                            ) : (
                                                <div className="w-6 h-6 rounded-full border-2 border-gray-200"></div>
                                            )}
                                            <span className={cn("text-sm font-medium", analysisStep > 3 ? "text-gray-900" : "text-gray-500")}>
                                                {t('diseaseDetection.correlating')}
                                            </span>
                                        </div>

                                        {/* Step 4: Diagnosis */}
                                        <div className={`flex items-center gap-3 transition-opacity duration-500 ${analysisStep >= 4 ? 'opacity-100' : 'opacity-40'}`}>
                                            {analysisStep > 4 ? (
                                                <div className="bg-green-100 p-1 rounded-full"><CheckCircle className="h-5 w-5 text-green-600" /></div>
                                            ) : analysisStep === 4 ? (
                                                <Loader2 className="h-6 w-6 text-green-600 animate-spin" />
                                            ) : (
                                                <div className="w-6 h-6 rounded-full border-2 border-gray-200"></div>
                                            )}
                                            <span className={cn("text-sm font-medium", analysisStep > 4 ? "text-gray-900" : "text-gray-500")}>
                                                {t('diseaseDetection.generating')}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : result ? (
                            <div className="bg-white rounded-3xl shadow-lg border border-green-100 overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500">
                                {/* Diagnosis Header */}
                                <div className="bg-green-600 p-6 text-white relative overflow-hidden">
                                    <div className="absolute top-0 right-0 p-4 opacity-10">
                                        <Activity className="h-32 w-32" />
                                    </div>
                                    <div className="relative z-10">
                                        <div className="flex justify-between items-start mb-2">
                                            <span className="bg-green-500/30 border border-green-400/30 px-3 py-1 rounded-full text-xs font-semibold tracking-wide uppercase">
                                                {t('diseaseDetection.diagnosisReport')}
                                            </span>
                                            <span className="font-mono font-bold text-2xl">
                                                {(result.confidence * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                        <h2 className="text-3xl font-bold mb-1">{result.disease}</h2>
                                        <p className="text-green-100 text-sm">{t('diseaseDetection.highConfidence')}</p>
                                    </div>
                                </div>

                                <div className="p-6 space-y-6">
                                    {/* Action Card */}
                                    <div className="bg-green-50 rounded-2xl p-5 border border-green-100">
                                        <h3 className="flex items-center gap-2 font-bold text-green-900 mb-2">
                                            <CheckCircle className="h-5 w-5 text-green-600" />
                                            {t('diseaseDetection.recommendedAction')}
                                        </h3>
                                        <p className="text-green-800 leading-relaxed text-sm">
                                            {result.quick_action}
                                        </p>
                                    </div>

                                    {/* Warnings */}
                                    {result.rubbing_risk_warning && (
                                        <div className="bg-amber-50 rounded-2xl p-5 border border-amber-100">
                                            <h3 className="flex items-center gap-2 font-bold text-amber-900 mb-2">
                                                <AlertTriangle className="h-5 w-5 text-amber-600" />
                                                {t('diseaseDetection.riskAlert')}
                                            </h3>
                                            <p className="text-amber-800 leading-relaxed text-sm">
                                                {result.rubbing_risk_warning}
                                            </p>
                                        </div>
                                    )}

                                    {/* Other Possibilities */}
                                    <div>
                                        <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
                                            {t('diseaseDetection.alternativeDiagnoses')}
                                        </h4>
                                        <div className="space-y-3">
                                            {result.all_predictions.slice(1, 4).map((pred, idx) => (
                                                <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors">
                                                    <span className="text-sm font-medium text-gray-700">{pred.class}</span>
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-24 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                                            <div
                                                                className="h-full bg-green-500 rounded-full"
                                                                style={{ width: `${pred.confidence * 100}%` }}
                                                            />
                                                        </div>
                                                        <span className="text-xs font-mono text-gray-500 w-10 text-right">
                                                            {(pred.confidence * 100).toFixed(1)}%
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            // Placeholder State for Right Column
                            <div className="h-full min-h-[400px] border-2 border-dashed border-gray-200 rounded-3xl flex flex-col items-center justify-center text-gray-400 p-8 text-center bg-gray-50/50">
                                <Activity className="h-16 w-16 mb-4 opacity-20" />
                                <h3 className="text-lg font-semibold text-gray-500 mb-2">{t('diseaseDetection.readyToAnalyze')}</h3>
                                <p className="text-sm max-w-xs mx-auto">
                                    {t('diseaseDetection.readyDesc')}
                                </p>
                            </div>
                        )}
                    </div>

                </div>
            </div>
        </div>
    );
};

export default DiseaseDetection;
