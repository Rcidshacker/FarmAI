import React from 'react';
import { Home, Activity, Calendar, HelpCircle, User, Camera } from 'lucide-react';
import { TubelightNavbar } from './ui/TubelightNavbar';
import { BottomNavbar } from './BottomNavbar';
import { useTranslation } from 'react-i18next';

export function Navbar() {
    const { t } = useTranslation();
    const navItems = [
        { name: t('nav.pestRisk'), icon: Activity, url: "/pest-risk" },
        { name: t('nav.schedule'), icon: Calendar, url: "/spray-schedule" },
        { name: t('nav.camera'), icon: Camera, url: "/" },
        { name: t('nav.assistant'), icon: HelpCircle, url: "/assistant" },
        { name: t('nav.profile'), icon: User, url: "/profile" },
    ];

    return (
        <>
            {/* Desktop Navbar - Hidden on Mobile */}
            <TubelightNavbar items={navItems} className="hidden md:flex" />

            {/* Mobile Bottom Navbar - Hidden on Desktop */}
            <BottomNavbar items={navItems} className="md:hidden" />

            {/* Spacer for content */}
            <div className="h-0 md:h-[72px]" />
            <div className="h-[64px] pb-[env(safe-area-inset-bottom)] md:h-0" />
        </>
    );
}
