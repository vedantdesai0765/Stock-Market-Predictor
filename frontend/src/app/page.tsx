"use client";

import React from "react";
import Link from "next/link";
import { ArrowRight, CalendarClock, Sunrise } from "lucide-react";
import { useMorningCall, useMorningScorecard } from "@/hooks/useMorning";
import { MarketPulse } from "@/components/morning/MarketPulse";
import { CallCard } from "@/components/morning/CallCard";
import { TrackRecord } from "@/components/morning/TrackRecord";
import { ErrorState } from "@/components/ui/ErrorState";
import { Skeleton } from "@/components/ui/LoadingSkeleton";
import { cn, formatDateLong, formatDateShort, formatTimeIST } from "@/lib/utils";
import { MorningCallResponse } from "@/types/api";

/** Works out what the top of the page should say for every situation the day can be in. */
function describe(d: MorningCallResponse) {
  const showCalls = d.status === "ok" && d.calls.length > 0;
  const next = `Next call: ${formatDateShort(d.next_call_date)}, about 8:00 am`;
  const scored = d.calls.filter((c) => c.result?.correct != null);
  const right = scored.filter((c) => c.result?.correct).length;
  const n = (d.summary?.higher ?? 0) + (d.summary?.lower ?? 0);

  if (d.status === "not_started") {
    return {
      showCalls, chip: "Starting soon", tone: "sky" as const,
      title: "The first Morning Call is on its way",
      text: `A forecast for all 10 stocks is published every trading day at about 8:00 am, before the market opens. ${next}.`,
    };
  }

  if (d.is_today && d.status === "ok") {
    const h = d.summary?.higher ?? 0;
    const l = d.summary?.lower ?? 0;
    const title =
      l === 0 ? `All ${h} stocks are expected to open higher today`
      : h === 0 ? `All ${l} stocks are expected to open lower today`
      : `${h} expected to open higher, ${l} lower`;
    const made = `Published at ${formatTimeIST(d.generated_at)}`;
    const text = d.on_time
      ? `${made}, before the market opened. It uses ${formatDateShort(d.based_on)}'s closing prices and everything that happened overnight.`
      : `${made}, after the market had already opened, so today is not counted in the track record.`;
    const chip =
      scored.length > 0 ? `${right} of ${scored.length} right so far`
      : d.market_phase === "pre_open" ? "Before the open"
      : d.market_phase === "open" ? "Market open" : "Market closed";
    const tone = scored.length > 0 ? ("emerald" as const) : d.market_phase === "pre_open" ? ("sky" as const) : ("slate" as const);
    return { showCalls, chip, tone, title, text, n };
  }

  if (d.market_phase === "holiday" || (d.is_today && d.status === "market_closed")) {
    return {
      showCalls, chip: "Market closed", tone: "slate" as const,
      title: `The market is closed today${d.reason && d.reason !== "Weekend" ? ` for ${d.reason}` : ""}`,
      text: `${next}.${showCalls ? " The most recent call and its results are below." : ""}`,
    };
  }

  if (!d.is_today && d.market_phase === "pre_open") {
    return {
      showCalls, chip: "Arriving soon", tone: "sky" as const,
      title: "Today's call arrives at about 8:00 am",
      text: `It is made once the Asian markets have opened.${showCalls ? " Until then, the previous call and its results are below." : ""}`,
    };
  }

  return {
    showCalls, chip: "No forecast today", tone: "amber" as const,
    title: "No forecast was published this morning",
    text: `${d.is_today && d.reason ? d.reason + " " : "The price data could not be collected in time. "}${next}.`,
  };
}

const CHIP_TONES = {
  sky: "bg-sky-50 text-sky-800 border-sky-200",
  emerald: "bg-emerald-50 text-emerald-800 border-emerald-200",
  slate: "bg-slate-100 text-slate-700 border-slate-200",
  amber: "bg-amber-50 text-amber-800 border-amber-200",
};

export default function MorningCallPage() {
  const { data, isLoading, isError, refetch } = useMorningCall();
  const { data: card } = useMorningScorecard(5);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-32 w-full" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-44" />)}
        </div>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <ErrorState
        title="The Morning Call could not be loaded"
        message="The website could not reach its data service. It may be starting up, which takes about a minute. Please try again."
        onRetry={() => refetch()}
      />
    );
  }

  const view = describe(data);
  const calls = [...data.calls].sort((a, b) => {
    const strength = (c: typeof a) => (c.available && c.prob_up != null ? Math.abs(c.prob_up - 0.5) : -1);
    return strength(b) - strength(a);
  });
  const awaiting = data.is_today && data.market_phase !== "pre_open";

  return (
    <div className="space-y-7">
      {/* What today looks like, in one sentence */}
      <section className="bg-white border border-slate-200/90 rounded-lg shadow-subtle p-5 sm:p-6">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <span className="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-sky-700">
            <Sunrise className="w-4 h-4" />
            Morning Call
          </span>
          <span className="text-xs text-slate-500">{formatDateLong(data.today)}</span>
          <span className={cn("ml-auto inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-semibold", CHIP_TONES[view.tone])}>
            {view.chip}
          </span>
        </div>
        <h1 className="mt-3 text-xl sm:text-2xl font-bold tracking-tight text-slate-900">{view.title}</h1>
        <p className="mt-1.5 text-sm text-slate-600 max-w-3xl">{view.text}</p>
        {data.is_today && data.status === "ok" && (
          <p className="mt-3 inline-flex items-center gap-1.5 text-xs text-slate-500">
            <CalendarClock className="w-3.5 h-3.5" />
            Next call: {formatDateShort(data.next_call_date)}, about 8:00 am
          </p>
        )}
        {data.notes.length > 0 && data.status === "ok" && (
          <ul className="mt-3 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-md px-3 py-2 space-y-0.5">
            {data.notes.map((n) => <li key={n}>{n}</li>)}
          </ul>
        )}
      </section>

      {view.showCalls && <MarketPulse items={data.market} />}

      {view.showCalls && (
        <section aria-label="Stock calls">
          <div className="flex items-baseline justify-between mb-2">
            <h2 className="text-sm font-semibold text-slate-900">
              {data.is_today ? "Today's calls" : `Previous call, ${formatDateShort(data.target_date)}`}
            </h2>
            <span className="hidden sm:inline text-[11px] text-slate-500">Will each stock open above or below its last close? Most confident first</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {calls.map((c) => (
              <CallCard key={c.ticker} call={c} awaitingResult={awaiting} baseRateUp={data.model?.base_rate_up} />
            ))}
          </div>
        </section>
      )}

      {card && <TrackRecord card={card} />}

      {/* What this is and what it is not */}
      <section className="bg-slate-50 border border-slate-200 rounded-lg p-5 text-xs text-slate-600 leading-relaxed">
        <h2 className="text-sm font-semibold text-slate-900 mb-1.5">How to read this page</h2>
        <p className="max-w-4xl">
          The Indian market is shut for almost 18 hours between one close and the next open, while markets in the US
          and Asia keep trading. Each morning this page uses what they did overnight to forecast one thing: whether
          each stock will <span className="font-semibold text-slate-800">open</span> above or below its previous
          closing price. It is a research result, not a trading tip. The opening jump happens the moment trading
          starts, so nobody can buy at yesterday&rsquo;s price after reading it.
        </p>
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5">
          <Link href="/forecast" className="inline-flex items-center gap-1 font-medium text-sky-700 hover:text-sky-900">
            See the full test results <ArrowRight className="w-3.5 h-3.5" />
          </Link>
          <Link href="/methodology" className="inline-flex items-center gap-1 font-medium text-sky-700 hover:text-sky-900">
            How the model works <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </section>
    </div>
  );
}
