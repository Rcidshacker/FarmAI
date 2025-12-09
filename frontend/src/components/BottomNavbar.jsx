
import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { cn } from "../lib/utils";

export function BottomNavbar({ items, className }) {
    const navigate = useNavigate();
    const location = useLocation();

    return (
        <div className={cn(
            "fixed bottom-0 left-0 right-0 z-50 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800",
            "pb-[env(safe-area-inset-bottom)]",
            className
        )}>
            <div className="flex justify-around items-center h-16">
                {items.map((item) => {
                    const Icon = item.icon;
                    const isActive = location.pathname === item.url;

                    return (
                        <button
                            key={item.name}
                            onClick={() => navigate(item.url)}
                            className={cn(
                                "flex flex-col items-center justify-center w-full h-full space-y-1",
                                "text-gray-500 dark:text-gray-400",
                                isActive && "text-green-600 dark:text-green-500"
                            )}
                        >
                            <div className={cn(
                                "p-1 rounded-full transition-all duration-200",
                                isActive && "bg-green-50 dark:bg-green-900/20"
                            )}>
                                <Icon size={24} strokeWidth={isActive ? 2.5 : 2} />
                            </div>
                            <span className="text-[10px] font-medium">
                                {item.name}
                            </span>
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
