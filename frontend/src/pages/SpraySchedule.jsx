import React, { useEffect, useState } from 'react';
import { createSpraySchedule, recordSpray } from '../services/api';
import {
    Calendar,
    AlertTriangle,
    Droplets,
    Thermometer,
    Wind,
    Leaf,
    DollarSign,
    TrendingUp,
    SprayCan,
    Loader2,
    CheckCircle2,
    AlertCircle
} from 'lucide-react';
import { cn } from "../lib/utils";
import { LumaSpin } from '../components/ui/LumaSpin';
import { useTranslation } from 'react-i18next';
import { NeonButton } from '../components/ui/NeonButton';

const SpraySchedule = () => {
    const { t } = useTranslation();
    const [schedule, setSchedule] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isSpraying, setIsSpraying] = useState(false);
    const [spraySuccess, setSpraySuccess] = useState(false);

    const handleSpray = async () => {
        setIsSpraying(true);
        try {
            await recordSpray({
                date: new Date().toISOString(),
                name: "Manual Spray",
                notes: "Triggered from Spray Schedule"
            });
            setSpraySuccess(true);
            setTimeout(() => setSpraySuccess(false), 3000);
        } catch (error) {
            console.error("Spray failed", error);
        } finally {
            setIsSpraying(false);
        }
    };

    const fetchSchedule = async () => {
        setLoading(true);
        setError(null);
        try {
            // Requesting 30 days ahead for Pune
            const response = await createSpraySchedule("Pune", 30);
            setSchedule(response.data);
        } catch (e) {
            console.error(e);
            setError(e.message || "Failed to load schedule");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchSchedule();
    }, []);

    if (loading) {
        return (
            <div className="min-h-[60vh] flex flex-col items-center justify-center p-6 text-center">
                <LumaSpin />
                <h3 className="text-xl font-semibold text-gray-800 mt-6">{t('dashboard.optimizingSchedule')}</h3>
                <p className="text-gray-500 mt-2 max-w-sm">
                    {t('dashboard.analyzing')}
                </p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-[60vh] flex flex-col items-center justify-center p-6 text-center">
                <div className="bg-red-50 p-4 rounded-full mb-4">
                    <AlertTriangle className="h-8 w-8 text-red-500" />
                </div>
                <h3 className="text-xl font-semibold text-gray-800">Connection Error</h3>
                <p className="text-gray-500 mt-2 max-w-sm mb-6">
                    {error}. Please check your API settings in Profile.
                </p>
                <NeonButton onClick={fetchSchedule} neon={false} className="bg-indigo-600 text-white">
                    Retry Connection
                </NeonButton>
            </div>
        );
    }

    if (!schedule) return null;

    return (
        <div className="min-h-screen bg-gray-50/50 pb-20">
            {/* Header Section */}
            <div className="bg-white border-b border-gray-100 px-6 py-8 md:py-10 mb-8">
                <div className="max-w-5xl mx-auto">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                        <div>
                            <h1 className="text-3xl font-bold text-gray-900 tracking-tight mb-2">
                                {t('dashboard.smartSpraySchedule')}
                            </h1>
                            <div className="flex items-center gap-2 text-gray-500 text-sm">
                                <Calendar className="h-4 w-4" />
                                <span>{t('dashboard.next30Days')}</span>
                                <span className="w-1 h-1 rounded-full bg-gray-300"></span>
                                <span className="text-green-600 font-medium">Pune Region</span>
                            </div>
                        </div>

                        {/* Summary Cards & Action */}
                        <div className="flex flex-col md:flex-row gap-4 items-stretch">
                            <div className="flex gap-4">
                                <div className="bg-green-50 rounded-2xl p-4 flex-1 md:min-w-[160px] border border-green-100">
                                    <div className="flex items-center gap-2 text-green-700 mb-1">
                                        <DollarSign className="h-4 w-4" />
                                        <span className="text-xs font-bold uppercase tracking-wider">{t('dashboard.estCost')}</span>
                                    </div>
                                    <div className="text-2xl font-bold text-gray-900">
                                        ₹{schedule.summary.total_cost}
                                    </div>
                                </div>
                                <div className="bg-blue-50 rounded-2xl p-4 flex-1 md:min-w-[160px] border border-blue-100">
                                    <div className="flex items-center gap-2 text-blue-700 mb-1">
                                        <TrendingUp className="h-4 w-4" />
                                        <span className="text-xs font-bold uppercase tracking-wider">{t('dashboard.yieldSaved')}</span>
                                    </div>
                                    <div className="text-2xl font-bold text-gray-900">
                                        ₹{10000 - schedule.summary.estimated_yield_loss}
                                    </div>
                                </div>
                            </div>

                            {/* Spray Button */}
                            <NeonButton
                                onClick={handleSpray}
                                disabled={isSpraying || spraySuccess}
                                className={cn(
                                    "h-auto min-h-[80px] md:w-[140px] rounded-2xl flex flex-col items-center justify-center gap-2 transition-all",
                                    spraySuccess ? "bg-green-500 border-green-600" : "bg-gradient-to-br from-indigo-500 to-purple-600 border-indigo-400"
                                )}
                                neon={!spraySuccess}
                            >
                                {isSpraying ? (
                                    <>
                                        <Loader2 className="h-6 w-6 animate-spin text-white" />
                                        <span className="text-white text-xs font-bold">Spraying...</span>
                                    </>
                                ) : spraySuccess ? (
                                    <>
                                        <CheckCircle2 className="h-8 w-8 text-white" />
                                        <span className="text-white text-xs font-bold">Done!</span>
                                    </>
                                ) : (
                                    <>
                                        <SprayCan className="h-6 w-6 text-white" />
                                        <span className="text-white text-xs font-bold">Spray Now</span>
                                    </>
                                )}
                            </NeonButton>
                        </div>
                    </div>
                </div>
            </div>

            <div className="max-w-5xl mx-auto px-4 md:px-6 space-y-8">

                {/* Active Alerts */}
                {schedule.alerts && schedule.alerts.length > 0 && (
                    <div className="bg-amber-50/80 backdrop-blur-sm border border-amber-200 rounded-2xl p-6 relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-4 opacity-10">
                            <AlertTriangle className="h-32 w-32 text-amber-500" />
                        </div>
                        <div className="relative z-10">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="bg-amber-100 p-2 rounded-xl">
                                    <AlertCircle className="h-6 w-6 text-amber-600" />
                                </div>
                                <h3 className="text-lg font-semibold text-amber-900">{t('dashboard.activeAdvisories')}</h3>
                            </div>
                            <div className="space-y-3">
                                {schedule.alerts.map((alert, idx) => (
                                    <div key={idx} className="flex items-start gap-3 bg-white/60 p-3 rounded-xl border border-amber-100/50">
                                        <div className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-2 shrink-0" />
                                        <p className="text-amber-800 text-sm leading-relaxed">{alert}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {/* Timeline */}
                <div>
                    <div className="flex items-center gap-3 mb-6">
                        <Leaf className="h-5 w-5 text-green-600" />
                        <h2 className="text-lg font-bold text-gray-900">{t('dashboard.recommendedTimeline')}</h2>
                    </div>

                    <div className="relative space-y-8 md:space-y-6 before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-gray-200 before:via-gray-200 before:to-transparent">
                        {schedule.schedule.map((event, index) => {
                            const date = new Date(event.date);
                            return (
                                <div key={index} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group">
                                    {/* Timeline Dot */}
                                    <div className="absolute left-0 md:left-1/2 w-10 h-10 bg-white border-4 border-gray-100 rounded-full flex items-center justify-center shrink-0 z-10 md:-translate-x-1/2 shadow-sm group-hover:scale-110 transition-transform duration-300">
                                        <div className="w-3 h-3 bg-green-500 rounded-full" />
                                    </div>

                                    {/* Card */}
                                    <div className="w-full md:w-[calc(50%-2.5rem)] ml-14 md:ml-0 bg-white rounded-2xl p-5 shadow-sm border border-gray-100 hover:shadow-md transition-all active:scale-[0.98] active:bg-gray-50">

                                        <div className="flex items-start justify-between mb-4">
                                            {/* Date Badge */}
                                            <div className="bg-gray-50 border border-gray-200 rounded-xl px-3 py-1.5 text-center min-w-[60px]">
                                                <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                                                    {date.toLocaleDateString('en-US', { month: 'short' })}
                                                </div>
                                                <div className="text-xl font-bold text-gray-900 leading-none mt-0.5">
                                                    {date.getDate()}
                                                </div>
                                            </div>

                                            {/* Quality Badge */}
                                            <div className={cn(
                                                "flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border",
                                                event.spray_quality === 'Good'
                                                    ? "bg-green-50 text-green-700 border-green-200"
                                                    : "bg-red-50 text-red-700 border-red-200"
                                            )}>
                                                {event.spray_quality === 'Good' ? <CheckCircle2 className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
                                                {event.spray_quality} Quality
                                            </div>
                                        </div>

                                        <div className="mb-4">
                                            <h4 className="flex items-center gap-2 text-green-700 font-bold text-lg mb-1">
                                                <SprayCan className="h-5 w-5" />
                                                {event.chemical_name || "Scheduled Spray"}
                                            </h4>
                                            <div className="space-y-1.5">
                                                <p className="text-gray-700 text-sm font-medium flex items-center gap-2">
                                                    <span className="bg-green-100 text-green-800 text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wider font-bold">Dosage</span>
                                                    {event.dosage}
                                                </p>
                                                {event.active_ingredient && (
                                                    <p className="text-gray-500 text-xs">
                                                        <span className="font-semibold">Active:</span> {event.active_ingredient}
                                                    </p>
                                                )}
                                                {event.reasoning && (
                                                    <p className="text-gray-500 text-xs italic mt-2 border-l-2 border-gray-200 pl-2">
                                                        "{event.reasoning}"
                                                    </p>
                                                )}
                                            </div>
                                        </div>

                                        {/* Weather Context */}
                                        <div className="flex items-center gap-4 pt-4 border-t border-gray-100">
                                            <div className="flex items-center gap-1.5 text-xs font-medium text-gray-500 bg-gray-50 px-2 py-1 rounded-lg">
                                                <Thermometer className="h-3.5 w-3.5" />
                                                {event.weather.temp.toFixed(1)}°C
                                            </div>
                                            <div className="flex items-center gap-1.5 text-xs font-medium text-gray-500 bg-gray-50 px-2 py-1 rounded-lg">
                                                <Droplets className="h-3.5 w-3.5" />
                                                {event.weather.rainfall}mm Rain
                                            </div>
                                            <div className="flex items-center gap-1.5 text-xs font-medium text-gray-500 bg-gray-50 px-2 py-1 rounded-lg">
                                                <Wind className="h-3.5 w-3.5" />
                                                Low Wind
                                            </div>
                                        </div>

                                        {/* Reasoning */}
                                        <div className="mt-3 text-xs text-gray-400 italic">
                                            "{event.reasoning}"
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Bottom Spacer */}
                <div className="h-12"></div>
            </div>
        </div>
    );
};

export default SpraySchedule;
