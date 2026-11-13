import React from "react";
import { Check, Minus, X } from "lucide-react";
import { cn, formatDateShort, formatShare } from "@/lib/utils";
import { MorningScorecardResponse } from "@/types/api";

/** Two horizontal bars: the model against the simple guess, on the same calls. */
function Versus({ model, guess }: { model: number | null; guess: number | null }) {
  const rows = [
    { label: "Model", value: model, bar: "bg-sky-600", text: "text-slate-900 font-semibold" },
    { label: "Simple guess", value: guess, bar: "bg-slate-300", text: "text-slate-600" },
  ];
  return (
    <div className="space-y-1.5">
      {rows.map((r) => (
        <div key={r.label} className="flex items-center gap-2 text-xs">
          <span className="w-[84px] text-slate-500 shrink-0">{r.label}</span>
          <div className="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden">
            <div className={cn("h-full rounded-full", r.bar)} style={{ width: `${Math.round((r.value ?? 0) * 100)}%` }} />
          </div>
          <span className={cn("w-12 text-right font-mono", r.text)}>{formatShare(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

function Panel({ title, note, children }: { title: string; note: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-slate-200/90 rounded-lg p-4 shadow-subtle flex flex-col">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">{title}</h3>
      <div className="mt-3 flex-1">{children}</div>
      <p className="mt-3 pt-3 border-t border-slate-100 text-[11px] text-slate-500 leading-snug">{note}</p>
    </div>
  );
}

export function TrackRecord({ card }: { card: MorningScorecardResponse }) {
  const live = card.open;
  const hasLive = live.n > 0;
  const tested = card.tested;
  const unseen = card.after_study;

  return (
    <section aria-label="Track record">
      <div className="flex items-baseline justify-between mb-2">
        <h2 className="text-sm font-semibold text-slate-900">How reliable is it?</h2>
        <span className="hidden sm:inline text-[11px] text-slate-500">
          The simple guess always says &ldquo;higher&rdquo;, the more common outcome
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <Panel
          title="Live record"
          note={
            hasLive
              ? `Only forecasts published before 9:15 am count. Tracking since ${formatDateShort(card.since)}.`
              : "Only forecasts published before 9:15 am count. Each one is saved publicly before the market opens."
          }
        >
          {hasLive ? (
            <>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold font-mono tracking-tight text-slate-900">
                  {formatShare(live.accuracy, 0)}
                </span>
                <span className="text-xs text-slate-500">
                  {live.correct} of {live.n} calls right, over {card.mornings} morning{card.mornings === 1 ? "" : "s"}
                </span>
              </div>
              <div className="mt-3">
                <Versus model={live.accuracy} guess={live.baseline_accuracy} />
              </div>
              {card.days.length > 1 && (
                <div className="mt-3 flex items-end gap-1 h-8" aria-label="Share of calls right, by morning">
                  {card.days.map((d) => {
                    const share = d.n ? d.correct / d.n : 0;
                    return (
                      <div
                        key={d.date}
                        title={`${formatDateShort(d.date)}: ${d.correct} of ${d.n} right`}
                        className={cn(
                          "flex-1 max-w-3 rounded-sm",
                          share >= 0.6 ? "bg-emerald-500" : share >= 0.4 ? "bg-amber-400" : "bg-rose-400"
                        )}
                        style={{ height: `${Math.max(12, share * 100)}%` }}
                      />
                    );
                  })}
                </div>
              )}
            </>
          ) : (
            <div className="h-full flex flex-col justify-center">
              <div className="text-sm font-medium text-slate-800">No graded mornings yet</div>
              <p className="mt-1 text-xs text-slate-500">
                The record starts with the first forecast that is checked against a real opening price.
              </p>
            </div>
          )}
        </Panel>

        <Panel
          title="Tested on 13 years"
          note={
            tested
              ? `${tested.n.toLocaleString("en-IN")} calls, ${tested.years}. Each year was predicted by a model trained only on earlier years. It beat the simple guess in ${tested.years_beating_guess} of ${tested.years_tested} years.`
              : "Walk-forward test results are not available."
          }
        >
          {tested && (
            <>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold font-mono tracking-tight text-slate-900">
                  {formatShare(tested.accuracy)}
                </span>
                <span className="text-xs text-slate-500">
                  of calls right, {(tested.edge * 100).toFixed(1)} points above the simple guess
                </span>
              </div>
              <div className="mt-3">
                <Versus model={tested.accuracy} guess={tested.baseline} />
              </div>
              {unseen && (
                <p className="mt-3 text-xs text-slate-600">
                  On 2026 data it had never seen:{" "}
                  <span className="font-semibold font-mono text-slate-900">{formatShare(unseen.accuracy)}</span> against{" "}
                  <span className="font-mono">{formatShare(unseen.baseline)}</span>.
                </p>
              )}
            </>
          )}
        </Panel>

        <Panel
          title="Confidence means something"
          note="In testing, the more confident the model was, the more often it was right. Each card shows which group its call belongs to."
        >
          <div className="flex items-end justify-between gap-3 h-24">
            {card.confidence_bands.map((b) => (
              <div key={b.from} className="flex-1 flex flex-col items-center justify-end h-full">
                <span className="text-[11px] font-mono font-semibold text-slate-800 mb-1">
                  {formatShare(b.accuracy, 0)}
                </span>
                <div
                  className="w-full rounded-t bg-sky-600"
                  style={{ height: `${Math.round(((b.accuracy ?? 0.5) - 0.4) * 130)}%`, opacity: 0.45 + b.from * 0.6 - 0.2 }}
                />
              </div>
            ))}
          </div>
          <div className="mt-1.5 flex justify-between gap-3 text-[10px] text-slate-500">
            {card.confidence_bands.map((b) => (
              <span key={b.from} className="flex-1 text-center">
                {Math.round(b.from * 100)}&ndash;{Math.round(b.to * 100)}%
              </span>
            ))}
          </div>
          <div className="mt-0.5 text-center text-[10px] text-slate-400">Model&rsquo;s confidence in its call</div>
        </Panel>
      </div>

      {card.recent.length > 0 && <RecentResults card={card} />}
    </section>
  );
}

/** Last few mornings, call by call: one row per morning, one column per stock. */
function RecentResults({ card }: { card: MorningScorecardResponse }) {
  const tickers = Array.from(new Set(card.recent.flatMap((d) => d.results.map((r) => r.ticker)))).sort();
  return (
    <div className="mt-3 bg-white border border-slate-200/90 rounded-lg shadow-subtle overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100 flex items-baseline justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">Recent mornings</h3>
        <span className="text-[11px] text-slate-500 hidden sm:inline">Hover a mark to see the call</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-[10px] font-mono text-slate-500">
              <th className="text-left font-medium px-4 py-2 whitespace-nowrap">Morning</th>
              {tickers.map((t) => (
                <th key={t} className="font-medium px-1.5 py-2 text-center">
                  {t}
                </th>
              ))}
              <th className="text-right font-medium px-4 py-2">Right</th>
            </tr>
          </thead>
          <tbody>
            {card.recent.map((day) => {
              const by = new Map(day.results.map((r) => [r.ticker, r]));
              const scored = day.results.filter((r) => r.correct !== null);
              const right = scored.filter((r) => r.correct).length;
              return (
                <tr key={day.date} className="border-t border-slate-100">
                  <td className="px-4 py-2 text-slate-700 whitespace-nowrap">{formatDateShort(day.date)}</td>
                  {tickers.map((t) => {
                    const r = by.get(t);
                    const label = r
                      ? `${t}: called ${r.call.toLowerCase()} (${formatShare(r.prob_up, 0)} chance of higher), opened ${(r.actual || "").toLowerCase()}`
                      : `${t}: no forecast`;
                    return (
                      <td key={t} className="px-1.5 py-2 text-center" title={label}>
                        <span
                          className={cn(
                            "inline-flex w-5 h-5 rounded-full items-center justify-center",
                            r?.correct === true && "bg-emerald-100 text-emerald-700",
                            r?.correct === false && "bg-rose-100 text-rose-700",
                            (!r || r.correct === null) && "bg-slate-100 text-slate-400"
                          )}
                        >
                          {r?.correct === true ? (
                            <Check className="w-3 h-3" />
                          ) : r?.correct === false ? (
                            <X className="w-3 h-3" />
                          ) : (
                            <Minus className="w-3 h-3" />
                          )}
                        </span>
                      </td>
                    );
                  })}
                  <td className="px-4 py-2 text-right font-mono text-slate-700 whitespace-nowrap">
                    {right} of {scored.length}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
