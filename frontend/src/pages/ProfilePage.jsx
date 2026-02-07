import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, MapPin, Ruler, Phone, LogOut, Edit2, Check, X, Camera } from 'lucide-react';
import { NeonButton } from '../components/ui/NeonButton';
import defaultProfile from '../assets/default_profile.png';
import { useTranslation } from 'react-i18next';

const ProfilePage = () => {
    const { t, i18n } = useTranslation();
    const navigate = useNavigate();
    const [isEditing, setIsEditing] = useState(false);
    const [profile, setProfile] = useState({
        name: "",
        phone: "",
        location: "",
        landSize: "",
    });

    useEffect(() => {
        const storedUser = localStorage.getItem('user');
        if (storedUser) {
            try {
                const user = JSON.parse(storedUser);
                setProfile({
                    name: user.name || "",
                    phone: user.phone || "",
                    location: user.location?.name || "",
                    landSize: user.land_area_acres || "",
                });
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
        // Here you would typically save to backend
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

    const handleResetSpray = async () => {
        if (window.confirm(t('profile.confirmReset'))) {
            try {
                const storedUser = localStorage.getItem('user');
                if (storedUser) {
                    const user = JSON.parse(storedUser);
                    const { resetIntervention } = await import('../services/api');
                    await resetIntervention(user.user_id || 'guest');
                    alert(t('profile.sprayHistoryReset'));
                }
            } catch (e) {
                console.error(e);
                alert(t('profile.sprayHistoryResetFailed'));
            }
        }
    };

    const changeLanguage = (lng) => {
        i18n.changeLanguage(lng);
    };

    return (
        <div className="min-h-screen bg-gray-50 pb-24 md:pb-8 pt-4 md:pt-8 px-4">
            <div className="max-w-md mx-auto bg-white rounded-3xl shadow-sm overflow-hidden min-h-[60vh] relative">
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

                    {/* Language Switcher */}
                    <div className="w-full mb-6">
                        <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider ml-1 mb-2 block">{t('profile.language')}</label>
                        <div className="flex gap-2 justify-center bg-gray-50 p-2 rounded-xl border border-gray-100">
                            <button
                                onClick={() => changeLanguage('en')}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${i18n.language === 'en' ? 'bg-white shadow-sm text-green-600 border border-green-100' : 'text-gray-500 hover:text-gray-700'}`}
                            >
                                English
                            </button>
                            <button
                                onClick={() => changeLanguage('hi')}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${i18n.language === 'hi' ? 'bg-white shadow-sm text-green-600 border border-green-100' : 'text-gray-500 hover:text-gray-700'}`}
                            >
                                हिंदी
                            </button>
                            <button
                                onClick={() => changeLanguage('mr')}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${i18n.language === 'mr' ? 'bg-white shadow-sm text-green-600 border border-green-100' : 'text-gray-500 hover:text-gray-700'}`}
                            >
                                मराठी
                            </button>
                        </div>
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
                                        placeholder={t('profile.enterName')}
                                    />
                                </div>
                            ) : (
                                <div className="flex items-center gap-3 p-3">
                                    <User size={18} className="text-green-600" />
                                    <span className="text-lg font-bold text-gray-800">
                                        {profile.name || t('profile.guestFarmer')}
                                    </span>
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
                                        placeholder={t('profile.enterPhone')}
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
                                        placeholder={t('profile.cityCountry')}
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
                                        placeholder={t('profile.acresExample')}
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

                    {/* Danger Zone */}
                    <NeonButton
                        onClick={handleResetSpray}
                        variant="ghost"
                        className="w-full flex items-center justify-center gap-3 p-4 text-orange-500 bg-orange-50 hover:bg-orange-100 rounded-xl transition-all font-semibold group mb-3 hover:border-orange-200"
                        neon={false}
                    >
                        <LogOut size={20} className="rotate-180 group-hover:rotate-0 transition-transform duration-500" />
                        <span>{t('profile.resetSpray')}</span>
                    </NeonButton>

                    {/* Logout Button */}
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
