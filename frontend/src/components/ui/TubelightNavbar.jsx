import React, { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { motion } from "framer-motion";
import { useNavigate, useLocation } from "react-router-dom";
import { cn } from "../../lib/utils";

export function TubelightNavbar({ items, className }) {
    const navigate = useNavigate();
    const location = useLocation();
    const [activeTab, setActiveTab] = useState(items[0]?.name || "");
    const [isMobile, setIsMobile] = useState(false);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        const handleResize = () => {
            setIsMobile(window.innerWidth < 768);
        };

        handleResize();
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    // Update active tab based on current route
    useEffect(() => {
        const currentItem = items.find(item => item.url === location.pathname);
        if (currentItem) {
            setActiveTab(currentItem.name);
        }
    }, [location.pathname, items]);

    const handleClick = (item) => {
        setActiveTab(item.name);
        navigate(item.url);
    };

    if (!mounted) return null;

    return createPortal(
        <div
            className={cn(
                "fixed inset-0 z-50 flex justify-center pointer-events-none",
                "items-end pb-6 md:items-start md:pt-6",
                className,
            )}
        >
            <div className="pointer-events-auto flex items-center justify-center gap-3 bg-white/80 dark:bg-background/80 border border-gray-200 dark:border-border backdrop-blur-lg py-1 px-1 rounded-full shadow-lg">
                {items.map((item) => {
                    const Icon = item.icon;
                    const isActive = activeTab === item.name;

                    return (
                        <button
                            key={item.name}
                            onClick={() => handleClick(item)}
                            className={cn(
                                "relative cursor-pointer text-sm font-semibold px-6 py-2 rounded-full transition-colors flex items-center justify-center",
                                "text-gray-600 dark:text-foreground/80 hover:text-green-600 dark:hover:text-primary",
                                isActive && "bg-green-50 dark:bg-muted text-green-600 dark:text-primary",
                            )}
                        >
                            <span className="hidden md:inline">{item.name}</span>
                            <span className="md:hidden flex items-center justify-center">
                                <Icon size={18} strokeWidth={2.5} />
                            </span>
                            {isActive && (
                                <motion.div
                                    layoutId="lamp"
                                    className="absolute inset-0 w-full bg-green-100/50 dark:bg-primary/5 rounded-full -z-10"
                                    initial={false}
                                    transition={{
                                        type: "spring",
                                        stiffness: 300,
                                        damping: 30,
                                    }}
                                >
                                    <div className="absolute -top-2 left-0 right-0 mx-auto w-8 h-1 bg-green-600 dark:bg-primary rounded-t-full">
                                        <div className="absolute w-12 h-6 bg-green-400/30 dark:bg-primary/20 rounded-full blur-md -top-2 -left-2" />
                                        <div className="absolute w-8 h-6 bg-green-400/30 dark:bg-primary/20 rounded-full blur-md -top-1" />
                                        <div className="absolute w-4 h-4 bg-green-400/30 dark:bg-primary/20 rounded-full blur-sm top-0 left-2" />
                                    </div>
                                </motion.div>
                            )}
                        </button>
                    );
                })}
            </div>
        </div>,
        document.body
    );
}
