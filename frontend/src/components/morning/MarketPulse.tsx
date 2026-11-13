import React from "react";
import { cn, formatSignedPct } from "@/lib/utils";
import { MorningMarketInput } from "@/types/api";

/** The overnight inputs: what the rest of the world did while the Indian market was shut. */
export function MarketPulse({ items }: { items: MorningMarketInput[] }) {
  if (!items.length) return null;
  return (
    <section aria-label="Overnight markets">
      <div className="flex items-baseline justify-between mb-2">
        <h2 className="text-sm font-semibold text-slate-900">What happened overnight</h2>
        <span className="hidden sm:inline text-[11px] text-slate-500">Known before the NSE opens at 9:15 am</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {items.map((m) => (
          <PulseTile key={m.key} item={m} />
        ))}
      </div>
    </section>
  );
}

function PulseTile({ item }: { item: MorningMarketInput }) {
  const isVix = item.key === "vix";
  const isUs = item.key === "sp500";
  const pct = item.change_pct;
  const up = pct !== null && pct > 0;
  const down = pct !== null && pct < 0;

  const caption = !item.available
    ? item.note || "No data"
    : isVix
    ? `${formatSignedPct(pct, 1)} vs previous day`
    : isUs
    ? "Overnight session"
    : "Opened this morning";

  return (
    <div className="bg-white border border-slate-200/90 rounded-lg px-3.5 py-3 shadow-subtle">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold text-slate-800 truncate">{item.label}</span>
        <span className="text-[10px] text-slate-400 truncate">{isVix ? "Fear index" : item.region}</span>
      </div>
      <div
        className={cn(
          "mt-1 text-xl font-bold font-mono tracking-tight",
          !item.available && "text-slate-300",
          item.available && isVix && "text-slate-900",
          item.available && !isVix && up && "text-emerald-600",
          item.available && !isVix && down && "text-rose-600",
          item.available && !isVix && !up && !down && "text-slate-700"
        )}
      >
        {!item.available ? "–" : isVix ? (item.level ?? 0).toFixed(1) : formatSignedPct(pct)}
      </div>
      <div className="mt-0.5 text-[11px] text-slate-500 truncate" title={caption}>
        {caption}
      </div>
    </div>
  );
}
