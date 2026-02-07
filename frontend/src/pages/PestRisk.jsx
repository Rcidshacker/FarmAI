import React, { useState, useEffect } from 'react';
import { predictPestRisk } from '../services/api';
import { MapPin, CloudRain, Thermometer, Droplets, Info, Shield, Eye, Bug, AlertTriangle, Moon, Leaf, Flower, Apple, Scissors, Sun, Wind } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { LumaSpin } from '../components/ui/LumaSpin';
import { NeonButton } from '../components/ui/NeonButton';
import Spline from '@splinetool/react-spline';
import { useTranslation } from 'react-i18next';

const PestRisk = () => {
    const { t } = useTranslation();
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState(null);
    const [cropStage, setCropStage] = useState("Fruiting (Fruit Set)");

    const handlePredict = async () => {
        setLoading(true);
        try {
            // Using Pune as hardcoded in backend logic, passing new interactive parameters
            const response = await predictPestRisk({
                location: { name: "Pune" }, // Updated structure to match backend Pydantic model
                use_realtime: true,
                crop_stage: cropStage
            });
            setData(response.data); // Axios response.data
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    // Auto-calculate when inputs change with debounce & Sync to Chatbot Context
    useEffect(() => {
        const timer = setTimeout(() => {
            handlePredict();
        }, 500); // Debounce delay 500ms

        // SYNC WITH CHATBOT: Save context silently
        try {
            const currentContext = JSON.parse(localStorage.getItem('farm_context') || '{}');
            currentContext.crop_stage = cropStage;
            localStorage.setItem('farm_context', JSON.stringify(currentContext));
        } catch (e) {
            console.warn("Context sync failed", e);
        }

        return () => clearTimeout(timer);
    }, [cropStage]);

    // ... (rest of code) ...

    const getRiskColor = (score) => {
        if (score > 70) return "#ef4444"; // red-500
        if (score > 30) return "#eab308"; // yellow-500
        return "#22c55e"; // green-500
    };

    const GaugeChart = ({ value }) => {
        const data = [
            { name: 'Risk', value: value },
            { name: 'Remaining', value: 100 - value }
        ];
        const COLORS = [getRiskColor(value), '#e5e7eb'];

        return (
            <div className="relative h-48 w-full flex justify-center items-center">
                <PieChart width={200} height={200}>
                    <Pie
                        data={data}
                        cx={100}
                        cy={100}
                        startAngle={180}
                        endAngle={0}
                        innerRadius={60}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                    >
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                    </Pie>
                </PieChart>
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1 mt-4 text-center">
                    <p className="text-3xl font-bold" style={{ color: getRiskColor(value) }}>{value.toFixed(1)}%</p>
                    <p className="text-xs text-gray-500 font-medium uppercase">{t('pestRisk.riskScore')}</p>
                </div>
            </div>
        );
    };

    return (
        <div className="p-6 max-w-6xl mx-auto space-y-6">
            <div className="flex flex-col md:flex-row justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-gray-100 mb-6">
                <h1 className="text-3xl font-bold text-green-800 tracking-tight text-center md:text-left mb-4 md:mb-0">{t('pestRisk.title')}</h1>
                <div className="flex items-center gap-4">
                    <span className="text-sm font-medium text-gray-500 bg-gray-100 px-3 py-1 rounded-full">{data ? t('common.modelActive') : t('common.readyToForecast')}</span>
                    <NeonButton
                        onClick={handlePredict}
                        disabled={loading}
                        variant="solid"
                    >
                        {loading ? t('common.calculating') : t('common.runPrediction')}
                    </NeonButton>
                </div>
            </div>

            {/* Controls Section */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-3">{t('pestRisk.selectStage')}</label>
                    <div className="flex overflow-x-auto pb-4 gap-3 md:grid md:grid-cols-5 md:gap-3 snap-x no-scrollbar md:pb-0">
                        {[
                            { id: "Dormant / Post-Harvest", label: t('pestRisk.stages.dormant'), icon: Moon, desc: t('pestRisk.stageDesc.dormant') },
                            { id: "Vegetative (New Leaves)", label: t('pestRisk.stages.vegetative'), icon: Leaf, desc: t('pestRisk.stageDesc.vegetative') },
                            { id: "Flowering", label: t('pestRisk.stages.flowering'), icon: Flower, desc: t('pestRisk.stageDesc.flowering') },
                            { id: "Fruiting (Fruit Set)", label: t('pestRisk.stages.fruiting'), icon: Apple, desc: t('pestRisk.stageDesc.fruiting') },
                            { id: "Harvesting", label: t('pestRisk.stages.harvest'), icon: Scissors, desc: t('pestRisk.stageDesc.harvest') }
                        ].map((stage) => {
                            const Icon = stage.icon;
                            const isSelected = cropStage === stage.id;
                            return (
                                <button
                                    key={stage.id}
                                    onClick={() => setCropStage(stage.id)}
                                    className={`flex-shrink-0 w-[120px] md:w-auto snap-center flex flex-col items-center justify-center p-3 rounded-xl border transition-all duration-200 ${isSelected
                                        ? 'bg-green-50 border-green-500 ring-1 ring-green-500 shadow-sm'
                                        : 'bg-white border-gray-200 hover:border-green-200 hover:bg-green-50/50'
                                        }`}
                                >
                                    <div className={`p-2 rounded-full mb-2 ${isSelected ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                                        <Icon className="w-6 h-6" />
                                    </div>
                                    <span className={`text-sm font-bold ${isSelected ? 'text-green-800' : 'text-gray-700'}`}>
                                        {stage.label}
                                    </span>
                                    <span className="text-xs text-gray-400 mt-0.5 text-center leading-tight">
                                        {stage.desc}
                                    </span>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* System Intelligence Display (Replaces Slider) */}
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                    <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
                        <span className="mr-2">📡</span> {t('pestRisk.systemIntelligence')}
                    </h3>

                    {data && data.twin_brain_status ? (
                        <div>
                            <p className={`text-md font-medium`} style={{ color: data.twin_brain_status.color }}>
                                {data.twin_brain_status.message}
                            </p>
                            <p className="text-xs text-gray-500 mt-1">
                                {t('pestRisk.calculatedDensity')}: {data.twin_brain_status.rvi.toFixed(2)}
                            </p>
                        </div>
                    ) : (
                        <p className="text-sm text-gray-400 italic">{t('pestRisk.calibratedVegetation')}</p>
                    )}
                </div>
            </div>

            {/* Loading Indicator */}
            {loading && (
                <div className="flex justify-center items-center py-12">
                    <LumaSpin />
                    <p className="ml-4 text-green-700 font-medium animate-pulse">{t('pestRisk.analyzing')}</p>
                </div>
            )}

            {!loading && data && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Environmental Context */}
                    {/* Environmental Context (Visual Redesign) */}
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 lg:col-span-1 space-y-6">

                        {/* Weather Station Widget */}
                        <div className="relative overflow-hidden rounded-2xl bg-green-600 bg-gradient-to-br from-green-600 to-emerald-500 p-6 text-white shadow-md">
                            <div className="absolute top-0 right-0 p-4 opacity-20">
                                <CloudRain className="h-24 w-24" />
                            </div>

                            <div className="relative z-10">
                                <div className="flex items-center gap-2 mb-4 opacity-90">
                                    <MapPin className="h-4 w-4" />
                                    <span className="text-sm font-semibold tracking-wide uppercase">{data.location} {t('pestRisk.station')}</span>
                                </div>

                                <div className="flex justify-between items-end mb-6">
                                    <div>
                                        <div className="text-5xl font-bold tracking-tighter">{data.current_weather.temperature}°</div>
                                        <div className="text-sm font-medium opacity-80 mt-1">{t('pestRisk.temperature')}</div>
                                    </div>
                                    <div className="text-right">
                                        <div className="flex items-center justify-end gap-1 text-xl font-bold">
                                            <Droplets className="h-5 w-5" />
                                            {data.current_weather.humidity}%
                                        </div>
                                        <div className="text-sm font-medium opacity-80 mt-1">{t('pestRisk.humidity')}</div>
                                    </div>
                                </div>

                                {/* Mini Gauges - Collapsible on small screens if needed, but keeping 2 basic ones is fine */}
                                <div className="grid grid-cols-2 gap-2">
                                    <div className="bg-white/20 backdrop-blur-sm rounded-lg p-2 text-center">
                                        <div className="text-xs opacity-75 mb-1">{t('pestRisk.windSpeed')}</div>
                                        <div className="font-bold flex items-center justify-center gap-1">
                                            <Wind className="h-3 w-3" /> 5 km/h
                                        </div>
                                    </div>
                                    <div className="bg-white/20 backdrop-blur-sm rounded-lg p-2 text-center">
                                        <div className="text-xs opacity-75 mb-1">{t('pestRisk.condition')}</div>
                                        <div className="font-bold text-xs truncate">Partly Cloudy</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Soil DNA Widget */}
                        <div>
                            <h3 className="font-semibold text-gray-500 mb-4 uppercase tracking-wider text-xs flex items-center gap-2">
                                <span className="w-1.5 h-1.5 rounded-full bg-amber-600"></span>
                                {t('pestRisk.soilAnalysis')}
                            </h3>
                            <div className="bg-amber-50/50 p-5 rounded-2xl border border-amber-100 space-y-4">
                                <div className="flex justify-between items-center">
                                    <span className="font-bold text-amber-900">{data.soil_info.type}</span>
                                    <span className="text-xs bg-amber-100 text-amber-800 px-2 py-1 rounded-full border border-amber-200">
                                        {t('pestRisk.sampleId')}: #8821
                                    </span>
                                </div>

                                <div>
                                    <div className="flex justify-between text-xs font-medium text-gray-500 mb-1">
                                        <span>{t('pestRisk.soilComponents')}</span>
                                        <span>{data.soil_info.clay_percent}% {t('pestRisk.clay')}</span>
                                    </div>
                                    <div className="h-4 w-full bg-stone-200 rounded-full overflow-hidden flex shadow-inner">
                                        <div
                                            className="h-full bg-amber-600 pattern-diagonal-lines"
                                            style={{ width: `${data.soil_info.clay_percent}%` }}
                                        ></div>
                                        <div className="h-full bg-stone-300 flex-1"></div>
                                    </div>
                                </div>

                                {data.soil_info.clay_percent > 25 && (
                                    <div className="bg-white p-3 rounded-xl border border-amber-200 shadow-sm flex gap-3 items-start">
                                        <div className="bg-red-100 p-2 rounded-full text-red-600 flex-shrink-0">
                                            <Bug className="h-5 w-5" />
                                        </div>
                                        <div>
                                            <h4 className="text-sm font-bold text-gray-800">{t('pestRisk.highAntRisk')}</h4>
                                            <p className="text-xs text-gray-500 leading-relaxed mt-1">
                                                {t('pestRisk.antRiskDesc')}
                                            </p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Risk & Forecast */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Current Risk Gauge & 3D Model */}
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 text-center relative overflow-hidden">
                            <h3 className="font-bold text-gray-800 mb-4 z-10 relative">{t('pestRisk.riskVisualizer')}</h3>

                            <div className="h-[300px] md:h-[400px] w-full rounded-2xl overflow-hidden shadow-inner bg-gray-900 border border-gray-700 relative mb-6">
                                <div className="absolute top-4 right-4 z-10 bg-black/50 backdrop-blur-md px-3 py-1 rounded-full text-xs font-mono text-white border border-white/20">
                                    {t('pestRisk.interactive3d')}
                                </div>

                                <Spline
                                    scene={
                                        data.pest_predictions['Mealy Bug'] > 70
                                            ? "https://prod.spline.design/7gVOeTrEuxBVtk0s/scene.splinecode" // Extreme
                                            : data.pest_predictions['Mealy Bug'] > 30
                                                ? "https://prod.spline.design/fkJydY3xbEfrdc2C/scene.splinecode" // Mild
                                                : "https://prod.spline.design/nSh716YCqf9icHlg/scene.splinecode" // Healthy
                                    }
                                    className="w-full h-full cursor-grab active:cursor-grabbing"
                                    onLoad={(spline) => {
                                        console.log('🎨 Spline scene loaded');

                                        let rotationActive = false;

                                        const tryRotateScene = () => {
                                            if (rotationActive) return;

                                            try {
                                                // Log all properties to find the scene
                                                console.log('🔍 Exploring spline properties...');
                                                for (const key in spline) {
                                                    if (spline[key] && typeof spline[key] === 'object') {
                                                        console.log(`  - ${key}:`, spline[key]);

                                                        // Check if this property has a traverse method (Three.js scene characteristic)
                                                        if (spline[key].traverse && typeof spline[key].traverse === 'function') {
                                                            console.log(`✅ Found scene-like object at: ${key}`);

                                                            const meshes = [];
                                                            spline[key].traverse((child) => {
                                                                if (child.isMesh) {
                                                                    console.log('🔍 Found mesh:', child.name || 'unnamed', child);
                                                                    meshes.push(child);
                                                                }
                                                            });

                                                            console.log(`📊 Total meshes found: ${meshes.length}`);

                                                            if (meshes.length > 0) {
                                                                rotationActive = true;

                                                                const rotateMeshes = () => {
                                                                    meshes.forEach(mesh => {
                                                                        if (mesh && mesh.rotation) {
                                                                            mesh.rotation.z += 0.01;
                                                                        }
                                                                    });
                                                                    requestAnimationFrame(rotateMeshes);
                                                                };

                                                                console.log('🚀 Starting mesh rotation...');
                                                                requestAnimationFrame(rotateMeshes);
                                                                return; // Exit once we found and started rotation
                                                            }
                                                        }
                                                    }
                                                }

                                                if (!rotationActive) {
                                                    console.warn('⚠️ Could not find any scene object with meshes');
                                                }
                                            } catch (e) {
                                                console.error('❌ Error accessing scene:', e);
                                            }
                                        };

                                        setTimeout(tryRotateScene, 1500);
                                    }}
                                />
                            </div>

                            <div className="flex flex-col md:flex-row items-center justify-center gap-8 z-10 relative">
                                <GaugeChart value={data.pest_predictions['Mealy Bug']} />
                                <div className="text-left space-y-2 text-sm max-w-xs">
                                    <h4 className="font-semibold text-gray-700 border-b pb-1">{t('pestRisk.riskBreakdown')}</h4>
                                    <div className="flex justify-between">
                                        <span className="text-gray-500">{t('pestRisk.aiRawScore')}:</span>
                                        <span className="font-medium">{(data.pest_predictions['Mealy Bug_details'].ai_score).toFixed(1)}%</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-500">{t('pestRisk.soilMultiplier')}:</span>
                                        <span className="font-medium">x{data.pest_predictions['Mealy Bug_details'].soil_multiplier.toFixed(2)}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-500">{t('pestRisk.enkfCorrection')}:</span>
                                        <span className="font-medium text-blue-600">
                                            {(data.pest_predictions['Mealy Bug_details'].enkf_fused_score).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="flex justify-between border-t pt-1 mt-1">
                                        <span className="text-gray-800 font-semibold">{t('pestRisk.finalAdjusted')}:</span>
                                        <span className="font-bold" style={{ color: getRiskColor(data.pest_predictions['Mealy Bug']) }}>
                                            {data.pest_predictions['Mealy Bug'].toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Forecast Cards (Farmer-First Redesign - Vertical on Mobile) */}
                        {data.forecast && data.forecast.dates.length > 0 && (
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                                <h3 className="font-bold text-gray-800 mb-4">{t('pestRisk.actionPlan')}</h3>
                                <div className="flex flex-col md:grid md:grid-cols-4 lg:grid-cols-7 gap-3">
                                    {data.forecast.dates.map((date, i) => {
                                        const risk = data.forecast.risks[i];
                                        let status, Icon, colorClass, bgColorClass, actionText, tip;

                                        if (risk < 30) {
                                            status = "Safe";
                                            Icon = Shield;
                                            colorClass = "text-green-600";
                                            bgColorClass = "bg-green-50 border-green-200";
                                            actionText = t('pestRisk.actions.safe');
                                            tip = t('pestRisk.tips.safe');
                                        } else if (risk < 70) {
                                            status = "Warning";
                                            Icon = Eye;
                                            colorClass = "text-yellow-600";
                                            bgColorClass = "bg-yellow-50 border-yellow-200";
                                            actionText = t('pestRisk.actions.warning');
                                            tip = t('pestRisk.tips.warning');
                                        } else {
                                            status = "Danger";
                                            Icon = Bug;
                                            colorClass = "text-red-600";
                                            bgColorClass = "bg-red-50 border-red-200";
                                            actionText = t('pestRisk.actions.danger');
                                            tip = t('pestRisk.tips.danger');
                                        }

                                        return (
                                            <div
                                                key={i}
                                                className={`flex md:flex-col items-center justify-between p-3 rounded-xl border ${bgColorClass} cursor-pointer transition-transform hover:scale-[1.02] md:hover:scale-105 h-auto md:h-32 group`}
                                                title={tip}
                                            >
                                                <div className="flex items-center gap-3 md:block">
                                                    <span className="text-xs font-bold text-gray-500 uppercase tracking-widest min-w-[3rem] md:mb-2">
                                                        {new Date(date).toLocaleDateString(undefined, { weekday: 'short' })}
                                                    </span>

                                                    <div className={`p-2 rounded-full bg-white shadow-sm ${colorClass} shrink-0`}>
                                                        <Icon className="w-5 h-5 md:w-6 md:h-6" />
                                                    </div>
                                                </div>

                                                <div className="flex-1 ml-4 md:ml-0 md:w-full text-right md:text-center">
                                                    <span className={`text-sm md:text-xs font-bold ${colorClass} block mb-1`}>{actionText}</span>
                                                    <div className="w-full bg-white/50 rounded-full h-1.5 md:h-1">
                                                        <div
                                                            className={`h-full rounded-full ${risk > 70 ? 'bg-red-500' : risk > 30 ? 'bg-yellow-500' : 'bg-green-500'}`}
                                                            style={{ width: `${risk}%` }}
                                                        ></div>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default PestRisk;
