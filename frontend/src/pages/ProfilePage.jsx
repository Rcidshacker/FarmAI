import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, MapPin, Ruler, Phone, LogOut, Edit2, Check, X, Camera, Plus, Trash2, Settings } from 'lucide-react';
import { NeonButton } from '../components/ui/NeonButton';
import { setApiUrl, getApiUrl, resetSprayHistory } from '../services/api';
import defaultProfile from '../assets/default_profile.png';
import LanguageSwitcher from '../components/LanguageSwitcher';
import { useTranslation } from 'react-i18next';

const ProfilePage = () => {
    const navigate = useNavigate();
    const { t } = useTranslation();
    const [isEditing, setIsEditing] = useState(false);
    const [showApiSettings, setShowApiSettings] = useState(false);
    const [apiUrl, setApiUrlState] = useState(getApiUrl());
    const [tempApiUrl, setTempApiUrl] = useState(apiUrl);
    const [profile, setProfile] = useState({
        name: "Guest Farmer",
        phone: "",
        location: "",
        landSize: "",
        farms: []
    });
    const [farms, setFarms] = useState([]);
    const [newFarm, setNewFarm] = useState({ name: "", lat: "", lon: "" });
    const [showAddFarm, setShowAddFarm] = useState(false);
    const [isResetting, setIsResetting] = useState(false);

    const handleResetHistory = async () => {
        if (window.confirm("Are you sure you want to reset all spray history? This cannot be undone.")) {
            setIsResetting(true);
            try {
                await resetSprayHistory();
                alert("Spray history has been reset.");
            } catch (error) {
                console.error("Failed to reset history", error);
                alert("Failed to reset history.");
            } finally {
                setIsResetting(false);
            }
        }
    };

    useEffect(() => {
        const storedUser = localStorage.getItem('user');
        if (storedUser) {
            try {
                const user = JSON.parse(storedUser);
                setProfile({
                    name: user.name || "Guest Farmer",
                    phone: user.phone || "",
                    location: user.location?.name || "",
                    landSize: user.land_area_acres || "",
                    farms: user.farms || []
                });
                setFarms(user.farms || []);
            } catch (e) {
                console.error("Failed to parse user profile", e);
            }
        }
    }, []);

    // Backup for cancel
    const [tempProfile, setTempProfile] = useState(profile);

    const handleEdit = () => {
        setTempProfile(profile);
        setIsEditing(true);
    };

    const handleSave = () => {
        // Save to localStorage
        const storedUser = localStorage.getItem('user');
        let user = storedUser ? JSON.parse(storedUser) : {};
        
        user = {
            ...user,
            name: profile.name,
            phone: profile.phone,
            location: { name: profile.location },
            land_area_acres: profile.landSize,
            farms: farms
        };
        
        localStorage.setItem('user', JSON.stringify(user));
        setIsEditing(false);
    };

    const handleCancel = () => {
        setProfile(tempProfile);
        setIsEditing(false);
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setProfile(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleAddFarm = () => {
        if (newFarm.name && newFarm.lat && newFarm.lon) {
            const farm = {
                id: Date.now(),
                name: newFarm.name,
                latitude: parseFloat(newFarm.lat),
                longitude: parseFloat(newFarm.lon),
                addedDate: new Date().toLocaleDateString()
            };
            const updatedFarms = [...farms, farm];
            setFarms(updatedFarms);
            setProfile(prev => ({ ...prev, farms: updatedFarms }));
            setNewFarm({ name: "", lat: "", lon: "" });
            setShowAddFarm(false);
        }
    };

    const handleDeleteFarm = (farmId) => {
        const updatedFarms = farms.filter(f => f.id !== farmId);
        setFarms(updatedFarms);
        setProfile(prev => ({ ...prev, farms: updatedFarms }));
    };

    const handleSaveApiUrl = () => {
        if (tempApiUrl.trim()) {
            setApiUrl(tempApiUrl);
            setApiUrlState(tempApiUrl);
            setShowApiSettings(false);
        }
    };

    const handleCancelApiSettings = () => {
        setTempApiUrl(apiUrl);
        setShowApiSettings(false);
    };

    return (
        <div className="min-h-screen bg-gray-50 pb-24 md:pb-8 pt-4 md:pt-8 px-4">
            <div className="max-w-md mx-auto bg-white rounded-3xl shadow-sm overflow-hidden min-h-[60vh] relative">
                {/* Header */}
                {/* Header */}
                <div className="relative pt-6 px-4 grid grid-cols-[48px_1fr_48px] items-center">
                    <div className="flex justify-start">
                        {isEditing ? (
                            <NeonButton onClick={handleCancel} variant="ghost" size="icon" className="text-gray-500 hover:text-red-500">
                                <X size={20} />
                            </NeonButton>
                        ) : (
                            <div className="w-9" />
                        )}
                    </div>

                    <div className="flex justify-center">
                        <h1 className="text-xl font-bold text-gray-800">{t('profile.title')}</h1>
                    </div>

                    <div className="flex justify-end">
                        {isEditing ? (
                            <NeonButton onClick={handleSave} variant="ghost" size="icon" className="text-green-600 hover:text-green-700">
                                <Check size={20} />
                            </NeonButton>
                        ) : (
                            <NeonButton onClick={handleEdit} variant="ghost" size="icon" className="text-green-700 bg-green-50 hover:bg-green-100 border border-green-200 shadow-sm w-10 h-10">
                                <Edit2 size={20} />
                            </NeonButton>
                        )}
                    </div>
                </div>

                {/* Profile Content */}
                <div className="flex flex-col items-center mt-8 px-6">

                    {/* Avatar Section */}
                    <div className="relative mb-8">
                        <div className="w-28 h-28 rounded-full p-1 ring-2 ring-green-100">
                            <img
                                src={defaultProfile}
                                alt="Profile"
                                className="w-full h-full rounded-full bg-gray-100 object-cover"
                            />
                        </div>
                        {isEditing && (
                            <NeonButton className="absolute bottom-1 right-1 bg-green-500 hover:bg-green-600 text-white shadow-md rounded-full" size="icon" neon={false}>
                                <Camera size={16} />
                            </NeonButton>
                        )}
                    </div>

                    {/* Form Fields */}
                    <div className="w-full space-y-5">
                        {/* Name */}
                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider ml-1">{t('profile.name')}</label>
                            {isEditing ? (
                                <div className="flex items-center gap-3 bg-gray-50 p-3 rounded-xl border border-gray-200 focus-within:border-green-500 focus-within:ring-1 focus-within:ring-green-500 transition-all">
                                    <User size={18} className="text-gray-400" />
                                    <input
                                        type="text"
                                        name="name"
                                        value={profile.name}
                                        onChange={handleChange}
                                        className="bg-transparent border-none outline-none w-full text-gray-800 font-medium placeholder-gray-400"
                                        placeholder="Enter your name"
                                    />
                                </div>
                            ) : (
                                <div className="flex items-center gap-3 p-3">
                                    <User size={18} className="text-green-600" />
                                    <span className="text-lg font-bold text-gray-800">{profile.name}</span>
                                </div>
                            )}
                        </div>

                        {/* Phone */}
                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider ml-1">{t('profile.phone')}</label>
                            {isEditing ? (
                                <div className="flex items-center gap-3 bg-gray-50 p-3 rounded-xl border border-gray-200 focus-within:border-green-500 focus-within:ring-1 focus-within:ring-green-500 transition-all">
                                    <Phone size={18} className="text-gray-400" />
                                    <input
                                        type="tel"
                                        name="phone"
                                        value={profile.phone}
                                        onChange={handleChange}
                                        className="bg-transparent border-none outline-none w-full text-gray-800 font-medium placeholder-gray-400"
                                        placeholder="Enter phone number"
                                    />
                                </div>
                            ) : (
                                <div className="flex items-center gap-3 p-3">
                                    <Phone size={18} className="text-green-600" />
                                    <span className="text-base font-medium text-gray-600">{profile.phone}</span>
                                </div>
                            )}
                        </div>

                        {/* Location */}
                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider ml-1">{t('profile.location')}</label>
                            {isEditing ? (
                                <div className="flex items-center gap-3 bg-gray-50 p-3 rounded-xl border border-gray-200 focus-within:border-green-500 focus-within:ring-1 focus-within:ring-green-500 transition-all">
                                    <MapPin size={18} className="text-gray-400" />
                                    <input
                                        type="text"
                                        name="location"
                                        value={profile.location}
                                        onChange={handleChange}
                                        className="bg-transparent border-none outline-none w-full text-gray-800 font-medium placeholder-gray-400"
                                        placeholder="City, Country"
                                    />
                                </div>
                            ) : (
                                <div className="flex items-center gap-3 p-3">
                                    <MapPin size={18} className="text-green-600" />
                                    <span className="text-base font-medium text-gray-600">{profile.location}</span>
                                </div>
                            )}
                        </div>

                        {/* Land Size */}
                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider ml-1">{t('profile.landSize')}</label>
                            {isEditing ? (
                                <div className="flex items-center gap-3 bg-gray-50 p-3 rounded-xl border border-gray-200 focus-within:border-green-500 focus-within:ring-1 focus-within:ring-green-500 transition-all">
                                    <Ruler size={18} className="text-gray-400" />
                                    <input
                                        type="text"
                                        name="landSize"
                                        value={profile.landSize}
                                        onChange={handleChange}
                                        className="bg-transparent border-none outline-none w-full text-gray-800 font-medium placeholder-gray-400"
                                        placeholder="e.g. 5.2"
                                    />
                                </div>
                            ) : (
                                <div className="flex items-center gap-3 p-3">
                                    <Ruler size={18} className="text-green-600" />
                                    <span className="text-base font-medium text-gray-600">{profile.landSize} Acres</span>
                                </div>
                            )}
                        </div>

                    </div>

                    <div className="w-full h-px bg-gray-100 my-8" />

                    {/* Farms Management Section */}
                    <div className="w-full">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-lg font-bold text-gray-800">{t('profile.farms')}</h2>
                            {isEditing && (
                                <NeonButton
                                    onClick={() => setShowAddFarm(!showAddFarm)}
                                    size="icon"
                                    className="bg-green-500 text-white rounded-full hover:bg-green-600"
                                    neon={false}
                                >
                                    <Plus size={16} />
                                </NeonButton>
                            )}
                        </div>

                        {showAddFarm && isEditing && (
                            <div className="bg-green-50 p-4 rounded-xl border border-green-200 mb-4 space-y-3">
                                <input
                                    type="text"
                                    placeholder="Farm name"
                                    value={newFarm.name}
                                    onChange={(e) => setNewFarm({ ...newFarm, name: e.target.value })}
                                    className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:border-green-500"
                                />
                                <div className="grid grid-cols-2 gap-2">
                                    <input
                                        type="number"
                                        placeholder="Latitude"
                                        value={newFarm.lat}
                                        onChange={(e) => setNewFarm({ ...newFarm, lat: e.target.value })}
                                        step="0.0001"
                                        className="p-2 border border-gray-300 rounded-lg focus:outline-none focus:border-green-500"
                                    />
                                    <input
                                        type="number"
                                        placeholder="Longitude"
                                        value={newFarm.lon}
                                        onChange={(e) => setNewFarm({ ...newFarm, lon: e.target.value })}
                                        step="0.0001"
                                        className="p-2 border border-gray-300 rounded-lg focus:outline-none focus:border-green-500"
                                    />
                                </div>
                                <NeonButton
                                    onClick={handleAddFarm}
                                    className="w-full bg-green-600 text-white rounded-lg hover:bg-green-700"
                                    neon={false}
                                >
                                    {t('profile.addFarm')}
                                </NeonButton>
                            </div>
                        )}

                        <div className="space-y-2 max-h-48 overflow-y-auto">
                            {farms.length > 0 ? (
                                farms.map((farm) => (
                                    <div key={farm.id} className="flex items-center justify-between bg-gray-50 p-3 rounded-lg border border-gray-200">
                                        <div className="flex-1">
                                            <p className="font-semibold text-gray-800">{farm.name}</p>
                                            <p className="text-xs text-gray-500">{farm.latitude.toFixed(4)}, {farm.longitude.toFixed(4)}</p>
                                        </div>
                                        {isEditing && (
                                            <NeonButton
                                                onClick={() => handleDeleteFarm(farm.id)}
                                                size="icon"
                                                className="bg-red-50 text-red-500 hover:bg-red-100 rounded-lg"
                                                neon={false}
                                            >
                                                <Trash2 size={16} />
                                            </NeonButton>
                                        )}
                                    </div>
                                ))
                            ) : (
                                <p className="text-center text-gray-400 py-4 text-sm">No farms added yet</p>
                            )}
                        </div>
                    </div>

                    <div className="w-full h-px bg-gray-100 my-8" />

                    {/* Language Settings */}
                    <div className="w-full mb-4">
                        <div className="flex items-center justify-between gap-3 p-4 bg-green-50 rounded-xl border border-green-200">
                            <div className="flex items-center gap-3">
                                <span className="text-green-700 font-semibold">{t('profile.language')}</span>
                            </div>
                            <LanguageSwitcher />
                        </div>
                    </div>

                    {/* API Configuration Section */}
                    <div className="w-full">
                        <button
                            onClick={() => {
                                setTempApiUrl(apiUrl);
                                setShowApiSettings(!showApiSettings);
                            }}
                            className="w-full flex items-center justify-between gap-3 p-4 bg-blue-50 hover:bg-blue-100 rounded-xl transition-all font-semibold mb-4 border border-blue-200"
                        >
                            <div className="flex items-center gap-3">
                                <Settings size={20} className="text-blue-600" />
                                <span className="text-blue-700">{t('profile.apiSettings')}</span>
                            </div>
                            <span className="text-xs bg-white px-2 py-1 rounded text-blue-600 font-mono">{apiUrl.split('://')[1] || apiUrl}</span>
                        </button>

                        {showApiSettings && (
                            <div className="w-full bg-gradient-to-b from-blue-50 to-white p-4 rounded-xl border border-blue-200 mb-4">
                                <label className="block text-sm font-semibold text-gray-700 mb-2">{t('profile.backendUrl')}</label>
                                <input
                                    type="text"
                                    value={tempApiUrl}
                                    onChange={(e) => setTempApiUrl(e.target.value)}
                                    placeholder="192.168.x.x:8000"
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm font-mono mb-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                                <p className="text-xs text-gray-600 mb-3">
                                    Enter IP:Port (e.g., 192.168.1.5:8000). No need for http://
                                </p>
                                <div className="flex gap-2">
                                    <NeonButton
                                        onClick={handleSaveApiUrl}
                                        className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg"
                                        neon={false}
                                    >
                                        <Check size={16} className="inline mr-1" /> {t('profile.save')}
                                    </NeonButton>
                                    <NeonButton
                                        onClick={handleCancelApiSettings}
                                        className="flex-1 bg-gray-300 hover:bg-gray-400 text-gray-800 font-semibold py-2 rounded-lg"
                                        neon={false}
                                    >
                                        <X size={16} className="inline mr-1" /> {t('profile.cancel')}
                                    </NeonButton>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="w-full h-px bg-gray-100 my-8" />

                    {/* Data Management */}
                    <div className="w-full mb-4">
                        <NeonButton
                            onClick={handleResetHistory}
                            disabled={isResetting}
                            className="w-full flex items-center justify-between gap-3 p-4 bg-amber-50 hover:bg-amber-100 rounded-xl transition-all font-semibold border border-amber-200 text-amber-700"
                            neon={false}
                        >
                            <div className="flex items-center gap-3">
                                <Trash2 size={20} className="text-amber-600" />
                                <span>Reset Spray History</span>
                            </div>
                            {isResetting && <span className="text-xs">Resetting...</span>}
                        </NeonButton>
                    </div>

                    <NeonButton
                        onClick={() => {
                            localStorage.removeItem('token');
                            localStorage.removeItem('user');
                            navigate('/auth');
                        }}
                        variant="ghost"
                        className="w-full flex items-center justify-center gap-3 p-4 text-red-500 bg-red-50 hover:bg-red-100 rounded-xl transition-all font-semibold group mb-6 hover:border-red-200"
                        neon={false}
                    >
                        <LogOut size={20} className="group-hover:scale-110 transition-transform" />
                        <span>{t('profile.logout')}</span>
                    </NeonButton>

                </div>
            </div >
        </div >
    );
};

export default ProfilePage;
