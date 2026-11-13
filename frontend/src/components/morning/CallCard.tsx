"use client";

import React, { useState } from "react";
import { ArrowDownRight, ArrowUpRight, CheckCircle2, ChevronDown, Clock, MinusCircle, XCircle } from "lucide-react";
import { cn, formatINR, formatShare, formatSignedPct } from "@/lib/utils";
import { MorningStockCall } from "@/types/api";

interface CallCardProps {
  call: MorningStockCall;
  /** true while today's session is running and the opening price has not been checked yet */
  awaitingResult: boolean;
  /** share of past mornings on which a stock opened higher (the model's starting point) */
  baseRateUp?: number | null;
}

export function CallCard({ call, awaitingResult, baseRateUp }: CallCardProps) {
  const [open, setOpen] = useState(false);

  if (!call.available || call.prob_up == null || !call.call) {
    return (
      <div className="bg-white border border-slate-200/90 rounded-lg p-4 shadow-subtle">
        <Title call={call} />
        <p className="mt-3 text-xs text-slate-500">
          No forecast for this stock today. {call.reason}
        </p>
      </div>
    );
  }

  const higher = call.call === "HIGHER";
  const chance = higher ? call.prob_up : 1 - call.prob_up;
  const main = call.drivers[0];

  return (
    <div className="bg-white border border-slate-200/90 rounded-lg shadow-subtle overflow-hidden flex flex-col">
      <div className="p-4 pb-3">
        <div className="flex items-start justify-between gap-3">
          <Title call={call} />
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-md px-2.5 py-1 text-xs font-bold tracking-wide border shrink-0",
              higher ? "bg-emerald-50 text-emerald-800 border-emerald-200" : "bg-rose-50 text-rose-800 border-rose-200"
            )}
          >
            {higher ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
            {higher ? "HIGHER" : "LOWER"}
          </span>
        </div>

        <ProbabilityBar probUp={call.prob_up} />

        <p className="mt-2 text-sm text-slate-800">
          <span className="font-semibold font-mono">{formatShare(chance, 0)}</span> chance of opening{" "}
          {higher ? "higher" : "lower"} than {call.prev_close != null ? formatINR(call.prev_close) : "the last close"}
        </p>
        {call.confidence?.tested_accuracy != null && (
          <p className="mt-0.5 text-xs text-slate-500">
            {call.confidence.level} confidence. Calls like this were right{" "}
            <span className="font-medium text-slate-700">{formatShare(call.confidence.tested_accuracy, 0)}</span> of the
            time in testing.
          </p>
        )}

        {main && (
          <button
            type="button"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            className="mt-3 w-full flex items-center justify-between gap-2 text-left text-xs text-slate-600 hover:text-slate-900 transition-colors"
          >
            <span className="flex items-center gap-1.5 min-w-0">
              <span className="text-slate-400 shrink-0">Biggest influence</span>
              {main.direction === "up" ? (
                <ArrowUpRight className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              ) : (
                <ArrowDownRight className="w-3.5 h-3.5 text-rose-600 shrink-0" />
              )}
              <span className="truncate">{main.detail}</span>
            </span>
            <span className="inline-flex items-center gap-0.5 text-sky-700 font-medium shrink-0">
              {open ? "Hide" : "Why?"}
              <ChevronDown className={cn("w-3.5 h-3.5 transition-transform", open && "rotate-180")} />
            </span>
          </button>
        )}

        {open && (
          <div className="mt-2.5 pt-2.5 border-t border-slate-100 space-y-2">
            {call.drivers.map((d) => (
              <div key={d.key} className="flex items-center gap-2.5 text-xs">
                <span
                  className={cn(
                    "w-5 h-5 rounded-full flex items-center justify-center shrink-0",
                    d.direction === "up" ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700"
                  )}
                  title={d.direction === "up" ? "Pushes the forecast higher" : "Pulls the forecast lower"}
                >
                  {d.direction === "up" ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="text-slate-800 font-medium truncate">{d.label}</div>
                  <div className="text-slate-500 truncate">{d.detail}</div>
                </div>
                <div className="w-16 shrink-0" title={`${Math.round(d.weight * 100)}% of the total push`}>
                  <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className={cn("h-full rounded-full", d.direction === "up" ? "bg-emerald-500" : "bg-rose-500")}
                      style={{ width: `${Math.max(6, Math.round(d.weight * 100))}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
            <p className="text-[11px] text-slate-400 leading-snug">
              {baseRateUp != null &&
                `The starting point is the usual pattern: stocks opened higher on ${formatShare(baseRateUp, 0)} of past mornings. `}
              Arrows show whether each input pushes this forecast higher or lower than on an average morning. Bars
              show how strongly.
            </p>
          </div>
        )}
      </div>

      <ResultStrip call={call} awaitingResult={awaitingResult} />
    </div>
  );
}

function Title({ call }: { call: MorningStockCall }) {
  return (
    <div className="min-w-0">
      <div className="font-mono text-sm font-bold text-slate-900">{call.ticker}</div>
      <div className="text-xs text-slate-500 truncate">{call.name}</div>
    </div>
  );
}

/** A bar centred on 50%: it grows right (green) for "higher" and left (red) for "lower". */
function ProbabilityBar({ probUp }: { probUp: number }) {
  const higher = probUp > 0.5;
  const width = Math.min(50, Math.abs(probUp - 0.5) * 100);
  return (
    <div className="mt-3.5" aria-hidden="true">
      <div className="relative h-2 rounded-full bg-slate-100">
        <div
          className={cn("absolute top-0 h-2", higher ? "bg-emerald-500 rounded-r-full" : "bg-rose-500 rounded-l-full")}
          style={higher ? { left: "50%", width: `${width}%` } : { left: `${50 - width}%`, width: `${width}%` }}
        />
        <div className="absolute top-[-3px] left-1/2 -translate-x-1/2 w-px h-3.5 bg-slate-400" />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-slate-400">
        <span>Lower</span>
        <span>50 / 50</span>
        <span>Higher</span>
      </div>
    </div>
  );
}

function ResultStrip({ call, awaitingResult }: CallCardProps) {
  const r = call.result;
  const base = "px-4 py-2 text-xs flex items-center gap-1.5 border-t mt-auto";

  if (r?.actual === "HIGHER" || r?.actual === "LOWER") {
    const right = r.correct === true;
    const move =
      r.open_price != null && call.prev_close ? ((r.open_price - call.prev_close) / call.prev_close) * 100 : null;
    return (
      <div
        className={cn(
          base,
          right ? "bg-emerald-50/70 border-emerald-100 text-emerald-800" : "bg-rose-50/70 border-rose-100 text-rose-800"
        )}
      >
        {right ? <CheckCircle2 className="w-3.5 h-3.5 shrink-0" /> : <XCircle className="w-3.5 h-3.5 shrink-0" />}
        <span className="font-semibold">{right ? "Right." : "Wrong."}</span>
        <span className="truncate">
          Opened at {r.open_price != null ? formatINR(r.open_price) : "–"}
          {move != null && ` (${formatSignedPct(move)})`}
        </span>
      </div>
    );
  }
  if (r?.actual === "FLAT") {
    return (
      <div className={cn(base, "bg-slate-50 border-slate-100 text-slate-600")}>
        <MinusCircle className="w-3.5 h-3.5 shrink-0" />
        Opened unchanged, so this call is not scored.
      </div>
    );
  }
  if (r?.actual === "VOID") {
    return (
      <div className={cn(base, "bg-slate-50 border-slate-100 text-slate-600")}>
        <MinusCircle className="w-3.5 h-3.5 shrink-0" />
        The market did not open, so this call is not scored.
      </div>
    );
  }
  if (awaitingResult) {
    return (
      <div className={cn(base, "bg-slate-50 border-slate-100 text-slate-500")}>
        <Clock className="w-3.5 h-3.5 shrink-0" />
        Result pending. Checked against the opening price after 9:50 am.
      </div>
    );
  }
  return null;
}
