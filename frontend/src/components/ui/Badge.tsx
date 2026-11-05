import React from "react";
import { cn } from "@/lib/utils";
import {
  Clock,
  TrendingUp,
  TrendingDown,
  Pause,
  MinusCircle,
  HelpCircle,
} from "lucide-react";

interface SignalBadgeProps {
  signal: string;
  className?: string;
  size?: "sm" | "md" | "lg";
  showDescription?: boolean;
}

/**
 * Consistent institutional signal design across all terminal modules.
 * CRITICAL RULE: HOLD and NO SIGNAL are distinct states and never interchangeable.
 */
export function SignalBadge({
  signal,
  className,
  size = "md",
  showDescription = false,
}: SignalBadgeProps) {
  const norm = (signal || "").trim().toUpperCase();

  const sizeClasses = {
    sm: "text-xs px-2 py-0.5 font-medium",
    md: "text-xs px-2.5 py-1 font-semibold",
    lg: "text-sm px-3.5 py-1.5 font-bold tracking-wide",
  }[size];

  // 1. BUY / UP Signal (Positive)
  if (norm === "BUY" || norm === "UP") {
    return (
      <div className="inline-flex flex-col">
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-md bg-emerald-50 text-emerald-800 border border-emerald-300 shadow-xs",
            sizeClasses,
            className
          )}
          title="BUY / UP: Model probability and expected returns exceed transaction costs."
        >
          <TrendingUp className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
          <span>{norm}</span>
        </span>
        {showDescription && (
          <span className="text-[10px] text-emerald-700 font-sans mt-0.5">
            Upward expected movement exceeds cost hurdle
          </span>
        )}
      </div>
    );
  }

  // 2. SELL / DOWN Signal (Negative / Exit)
  if (norm === "SELL" || norm === "DOWN") {
    return (
      <div className="inline-flex flex-col">
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-md bg-rose-50 text-rose-800 border border-rose-300 shadow-xs",
            sizeClasses,
            className
          )}
          title="SELL / DOWN: Downward probability indicates exiting positions or protecting capital."
        >
          <TrendingDown className="w-3.5 h-3.5 text-rose-600 shrink-0" />
          <span>{norm}</span>
        </span>
        {showDescription && (
          <span className="text-[10px] text-rose-700 font-sans mt-0.5">
            Downward expectation indicates capital preservation
          </span>
        )}
      </div>
    );
  }

  // 3. HOLD Signal (Deliberate pause / cost hurdle or indicator conflict)
  if (norm === "HOLD") {
    return (
      <div className="inline-flex flex-col">
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-md bg-amber-50 text-amber-800 border border-amber-300 shadow-xs",
            sizeClasses,
            className
          )}
          title="HOLD: Estimated expected return does not exceed transaction friction or technical indicators conflict. Stay in cash."
        >
          <Pause className="w-3.5 h-3.5 text-amber-600 shrink-0" />
          <span>HOLD</span>
        </span>
        {showDescription && (
          <span className="text-[10px] text-amber-700 font-sans mt-0.5">
            Hold cash: Returns do not overcome transaction friction
          </span>
        )}
      </div>
    );
  }

  // 4. NO SIGNAL (Model abstains due to low statistical conviction)
  return (
    <div className="inline-flex flex-col">
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-md bg-slate-100 text-slate-700 border border-slate-300 shadow-xs",
          sizeClasses,
          className
        )}
        title="NO SIGNAL: Conviction is below operating threshold. The model abstains rather than guessing."
      >
        <MinusCircle className="w-3.5 h-3.5 text-slate-500 shrink-0" />
        <span>NO SIGNAL</span>
      </span>
      {showDescription && (
        <span className="text-[10px] text-slate-500 font-sans mt-0.5">
          Model abstains: Insufficient conviction above noise
        </span>
      )}
    </div>
  );
}

interface FreshnessBadgeProps {
  isLive?: boolean;
  date?: string;
  className?: string;
}

export function FreshnessBadge({ isLive = false, date, className }: FreshnessBadgeProps) {
  if (isLive) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-700 border border-emerald-200",
          className
        )}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
        Live Feed
      </span>
    );
  }

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-800 border border-amber-300/80 shadow-xs",
        className
      )}
      title="Historical Research Snapshot. Evaluated on verified out-of-sample data. Not connected to a real-time live market feed."
    >
      <Clock className="w-3 h-3 text-amber-600" />
      <span>Research Snapshot {date ? `· ${date}` : ""}</span>
    </span>
  );
}

interface RegimeBadgeProps {
  regime: string;
  className?: string;
}

export function RegimeBadge({ regime, className }: RegimeBadgeProps) {
  const norm = (regime || "").toUpperCase();

  let style = "bg-slate-100 text-slate-700 border-slate-200";
  if (norm.includes("BULL")) {
    style = "bg-sky-50 text-sky-800 border-sky-300";
  } else if (norm.includes("BEAR")) {
    style = "bg-rose-50 text-rose-800 border-rose-300";
  } else if (norm.includes("VOLATILITY")) {
    style = "bg-amber-50 text-amber-800 border-amber-300";
  }

  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border uppercase tracking-wider font-mono",
        style,
        className
      )}
    >
      {regime.replace(/_/g, " ")}
    </span>
  );
}
