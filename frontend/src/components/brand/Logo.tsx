import React from "react";
import { cn } from "@/lib/utils";

interface LogoProps {
  className?: string;
  variant?: "full" | "icon";
  theme?: "dark" | "light"; // dark = for dark navy background; light = for light background
  size?: "sm" | "md" | "lg";
}

/**
 * Custom Vector Logo Mark for Stock Market Predictor:
 * - Three ascending bars (teal -> blue gradient).
 * - A rising trendline slicing through.
 * - An upward arrow representing forecasting and positive alpha.
 */
export function LogoIcon({
  className,
  size = "md",
}: {
  className?: string;
  size?: "sm" | "md" | "lg" | "xl";
}) {
  const pixelSize = {
    sm: 24,
    md: 32,
    lg: 40,
    xl: 48,
  }[size];

  return (
    <svg
      width={pixelSize}
      height={pixelSize}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("shrink-0", className)}
      aria-label="Stock Market Predictor Logo"
    >
      <defs>
        {/* Background container gradient */}
        <linearGradient id="smp-bg-grad" x1="4" y1="4" x2="44" y2="44" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#0B132B" />
          <stop offset="100%" stopColor="#1E293B" />
        </linearGradient>

        {/* Bar 1 gradient (teal accent) */}
        <linearGradient id="bar1-grad" x1="12" y1="36" x2="12" y2="26" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#0D9488" />
          <stop offset="100%" stopColor="#14B8A6" />
        </linearGradient>

        {/* Bar 2 gradient (cyan-blue) */}
        <linearGradient id="bar2-grad" x1="22" y1="36" x2="22" y2="19" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#0284C7" />
          <stop offset="100%" stopColor="#38BDF8" />
        </linearGradient>

        {/* Bar 3 gradient (primary blue -> green) */}
        <linearGradient id="bar3-grad" x1="32" y1="36" x2="32" y2="12" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#2563EB" />
          <stop offset="100%" stopColor="#10B981" />
        </linearGradient>

        {/* Trendline glow */}
        <linearGradient id="trendline-grad" x1="8" y1="32" x2="38" y2="10" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#0D9488" />
          <stop offset="50%" stopColor="#38BDF8" />
          <stop offset="100%" stopColor="#22C55E" />
        </linearGradient>
      </defs>

      {/* Rounded square container */}
      <rect x="2" y="2" width="44" height="44" rx="10" fill="url(#smp-bg-grad)" stroke="#334155" strokeWidth="1.5" />

      {/* 3 Ascending Bars */}
      {/* Bar 1 */}
      <rect x="10" y="26" width="6" height="11" rx="2" fill="url(#bar1-grad)" />
      {/* Bar 2 */}
      <rect x="19" y="19" width="6" height="18" rx="2" fill="url(#bar2-grad)" />
      {/* Bar 3 */}
      <rect x="28" y="12" width="6" height="25" rx="2" fill="url(#bar3-grad)" />

      {/* Rising Stock-Market Trendline */}
      <path
        d="M8 32 L18 24 L27 26 L38 12"
        stroke="url(#trendline-grad)"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Upward forecasting arrow head */}
      <path
        d="M32 12 H38 V18"
        stroke="#22C55E"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Subtle data pulse dot */}
      <circle cx="38" cy="12" r="2" fill="#86EFAC" />
    </svg>
  );
}

export function Logo({
  className,
  variant = "full",
  theme = "dark",
  size = "md",
}: LogoProps) {
  if (variant === "icon") {
    return <LogoIcon size={size} className={className} />;
  }

  const isDark = theme === "dark";

  return (
    <div className={cn("flex items-center gap-3 select-none", className)}>
      <LogoIcon size={size} />
      <div className="flex flex-col">
        <span
          className={cn(
            "font-bold text-sm tracking-tight leading-tight font-sans",
            isDark ? "text-white" : "text-slate-900"
          )}
        >
          Stock Market Predictor
        </span>
        <span
          className={cn(
            "text-[10px] font-mono font-medium tracking-wider uppercase leading-tight",
            isDark ? "text-sky-400" : "text-sky-600"
          )}
        >
          Institutional Analytics Terminal
        </span>
      </div>
    </div>
  );
}
