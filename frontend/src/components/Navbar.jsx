import React from 'react';
import { Activity, Calendar, HelpCircle, User, Camera, MessageCircle, BarChart3 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { TubelightNavbar } from './ui/TubelightNavbar';
import { BottomNavbar } from './BottomNavbar';

export function Navbar() {
    const { t } = useTranslation();

    const items = [
        { name: t('nav.camera') || 'Camera', icon: Camera, url: "/" },
        { name: t('nav.pestRisk') || 'Pest Risk', icon: Activity, url: "/pest-risk" },
        { name: t('nav.schedule') || 'Schedule', icon: Calendar, url: "/spray-schedule" },
        { name: t('nav.chatbot') || 'Chat', icon: MessageCircle, url: "/chatbot" },
        { name: t('nav.dashboard') || 'Dashboard', icon: BarChart3, url: "/dashboard" },
        { name: t('nav.assistant') || 'Assistant', icon: HelpCircle, url: "/assistant" },
        { name: t('nav.profile') || 'Profile', icon: User, url: "/profile" },
    ];

    return (
        <>
            {/* Desktop Navbar - Hidden on Mobile */}
            <TubelightNavbar items={items} className="hidden md:flex" />

            {/* Mobile Bottom Navbar - Hidden on Desktop */}
            <BottomNavbar items={items} className="md:hidden" />

            {/* Spacer for content */}
            <div className="h-0 md:h-[72px]" />
            <div className="h-[64px] pb-[env(safe-area-inset-bottom)] md:h-0" />
        </>
    );
}
