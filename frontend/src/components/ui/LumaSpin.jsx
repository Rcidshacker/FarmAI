import React from 'react';
import './LumaSpin.css';

export const LumaSpin = ({ className }) => {
    return (
        <div className={`relative w-[65px] aspect-square ${className || ''}`}>
            <span className="absolute rounded-[50px] animate-loaderAnim shadow-[inset_0_0_0_3px] shadow-black" />
            <span className="absolute rounded-[50px] animate-loaderAnim animation-delay shadow-[inset_0_0_0_3px] shadow-black" />
        </div>
    );
};
