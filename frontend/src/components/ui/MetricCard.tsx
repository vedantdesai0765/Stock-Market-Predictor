import React from "react";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { InfoTooltip } from "@/components/ui/InfoTooltip";

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  change?: number;
  changeSuffix?: string;
  badge?: React.ReactNode;
  icon?: React.ReactNode;
  hint?: string;
  className?: string;
  signalTone?: "buy" | "sell" | "hold" | "neutral" | "default";
  // Plain-English intuitive helper
  infoTerm?: string;
  infoExplanation?: string;
  technicalNote?: string;
}

export function MetricCard({
  title,
  value,
  subtitle,
  change,
  changeSuffix = "",
  badge,
  icon,
  hint,
  className,
  signalTone = "default",
  infoTerm,
  infoExplanation,
  technicalNote,
}: MetricCardProps) {
  const isPositive = change !== undefined && change > 0;
  const isNegative = change !== undefined && change < 0;

  // Background tinting per signal design requirements
  const toneClasses = {
    buy: "bg-emerald-50/50 border-emerald-300 hover:border-emerald-400",
    sell: "bg-rose-50/50 border-rose-300 hover:border-rose-400",
    hold: "bg-amber-50/50 border-amber-300 hover:border-amber-400",
    neutral: "bg-slate-50/60 border-slate-200 hover:border-slate-300",
    default: "bg-white border-slate-200/90 hover:border-slate-300",
  }[signalTone];

  return (
    <div
      className={cn(
        "border rounded-lg p-4 shadow-subtle transition-all flex flex-col justify-between",
        toneClasses,
        className
      )}
      title={hint}
    >
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 font-sans">
            {title}
          </span>
          {infoTerm && infoExplanation && (
            <InfoTooltip
              term={infoTerm}
              explanation={infoExplanation}
              technicalNote={technicalNote}
            />
          )}
        </div>
        {icon && <div className="text-slate-400">{icon}</div>}
      </div>

      <div className="my-1 flex items-baseline justify-between gap-2">
        <span className="text-2xl font-bold text-slate-900 font-mono tracking-tight">
          {value}
        </span>
        {badge}
      </div>

      <div className="mt-1 flex items-center justify-between text-xs">
        {change !== undefined && (
          <span
            className={cn(
              "inline-flex items-center gap-0.5 font-medium font-mono",
              isPositive && "text-emerald-700",
              isNegative && "text-rose-700",
              !isPositive && !isNegative && "text-slate-500"
            )}
          >
            {isPositive && <TrendingUp className="w-3.5 h-3.5" />}
            {isNegative && <TrendingDown className="w-3.5 h-3.5" />}
            {!isPositive && !isNegative && <Minus className="w-3.5 h-3.5" />}
            {change > 0 ? `+${change}` : change}
            {changeSuffix}
          </span>
        )}

        {subtitle && (
          <span className="text-slate-500 truncate font-sans text-xs">
            {subtitle}
          </span>
        )}
      </div>
    </div>
  );
}
