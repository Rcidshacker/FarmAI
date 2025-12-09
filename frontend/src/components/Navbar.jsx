import React from 'react';
import { Home, Activity, Calendar, HelpCircle, User, Camera } from 'lucide-react';
import { TubelightNavbar } from './ui/TubelightNavbar';

import { BottomNavbar } from './BottomNavbar';

export function Navbar() {
    const navItems = [
        { name: "Pest Risk", icon: Activity, url: "/pest-risk" },
        { name: "Schedule", icon: Calendar, url: "/spray-schedule" },
        { name: "Camera", icon: Camera, url: "/" },
        { name: "Assistant", icon: HelpCircle, url: "/assistant" },
        { name: "Profile", icon: User, url: "/profile" },
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
