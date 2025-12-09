import React from 'react';
import { cn } from '../../lib/utils';
import { cva } from "class-variance-authority";

const buttonVariants = cva(
    "relative group border text-foreground mx-auto text-center rounded-full cursor-pointer",
    {
        variants: {
            variant: {
                default: "bg-green-500/5 hover:bg-green-500/0 border-green-500/20",
                solid: "bg-green-600 hover:bg-green-700 text-white border-transparent hover:border-foreground/50 transition-all duration-200",
                ghost: "border-transparent bg-transparent hover:border-green-600/20 hover:bg-green-500/5 hover:text-green-700",
            },
            size: {
                default: "px-7 py-1.5",
                sm: "px-4 py-0.5",
                lg: "px-10 py-2.5",
                icon: "h-10 w-10 px-0 flex items-center justify-center", // Added icon size for compatibility
            },
        },
        defaultVariants: {
            variant: "default",
            size: "default",
        },
    }
);

const NeonButton = React.forwardRef(
    ({ className, neon = true, size, variant, children, ...props }, ref) => {
        return (
            <button
                className={cn(buttonVariants({ variant, size }), className)}
                ref={ref}
                {...props}
            >
                <span className={cn("absolute h-px opacity-0 group-hover:opacity-100 transition-all duration-500 ease-in-out inset-x-0 inset-y-0 bg-gradient-to-r w-3/4 mx-auto from-transparent dark:via-green-500 via-green-600 to-transparent hidden", neon && "block")} />
                {children}
                <span className={cn("absolute group-hover:opacity-30 transition-all duration-500 ease-in-out inset-x-0 h-px -bottom-px bg-gradient-to-r w-3/4 mx-auto from-transparent dark:via-green-500 via-green-600 to-transparent hidden", neon && "block")} />
            </button>
        );
    }
);

NeonButton.displayName = 'NeonButton';

export { NeonButton, buttonVariants };
