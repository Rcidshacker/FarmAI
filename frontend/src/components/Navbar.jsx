import React from 'react';
import { Activity, Calendar, HelpCircle, User, Camera } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { TubelightNavbar } from './ui/TubelightNavbar';
import { BottomNavbar } from './BottomNavbar';

export function Navbar() {
    const { t } = useTranslation();

    const items = [
        { name: t('navbar.pestRisk'), icon: Activity, url: "/pest-risk" },
        { name: t('navbar.spraySchedule'), icon: Calendar, url: "/spray-schedule" },
        { name: t('navbar.camera'), icon: Camera, url: "/" },
        { name: t('navbar.aiAssistant'), icon: HelpCircle, url: "/assistant" },
        { name: t('navbar.profile'), icon: User, url: "/profile" },
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
