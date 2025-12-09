import React from 'react';
import { useLottie } from 'lottie-react';
import animationData from '../Scene-1.json';

const SplashScreen = () => {
    const options = {
        animationData: animationData,
        loop: true,
        autoplay: true,
    };

    const { View } = useLottie(options);

    return (
        <div className="fixed inset-0 bg-white z-50 flex flex-col items-center justify-center p-4">
            <div className="w-full max-w-md md:max-w-lg lg:max-w-xl aspect-square">
                {View}
            </div>

            <div className="mt-8 text-center animate-fade-in">
                <h1 className="text-3xl md:text-4xl font-bold text-green-800 tracking-tight font-sans">
                    FarmAI
                </h1>
                <p className="text-lg md:text-xl text-green-600 mt-2 font-medium tracking-wide">
                    Guardian of the grove
                </p>
            </div>
        </div>
    );
};

export default SplashScreen;
