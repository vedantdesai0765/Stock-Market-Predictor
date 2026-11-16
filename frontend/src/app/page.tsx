"use client";

import React, { useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  ArrowUpDown,
  CalendarClock,
  CheckCircle2,
  Filter,
  Layers,
  Search,
  SlidersHorizontal,
  Sunrise,
  AlertTriangle,
  X,
} from "lucide-react";
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
  const total = d.summary?.total ?? d.calls.length ?? 30;
  const covered = d.summary?.covered ?? d.calls.filter((c) => c.available).length;
  const h = d.summary?.higher ?? 0;
  const l = d.summary?.lower ?? 0;
  const unavail = d.summary?.unavailable ?? (total - covered);

  if (d.status === "not_started") {
    return {
      showCalls,
      chip: "Starting soon",
      tone: "sky" as const,
      title: "The first Morning Call is on its way",
      text: `A forecast for all ${total} stocks is published every trading day at about 8:00 am, before the market opens. ${next}.`,
      total,
      covered,
    };
  }

  if (d.is_today && d.status === "ok") {
    let title = "";
    if (covered === 0) {
      title = "No individual stock forecasts are available today";
    } else if (l === 0 && unavail === 0) {
      title = `All ${h} stocks are expected to open higher today`;
    } else if (h === 0 && unavail === 0) {
      title = `All ${l} stocks are expected to open lower today`;
    } else if (unavail === 0) {
      title = `${h} of ${total} stocks expected to open higher, ${l} lower`;
    } else {
      title = `${h} expected higher, ${l} lower (${covered} of ${total} stocks covered)`;
    }

    const made = `Published at ${formatTimeIST(d.generated_at)}`;
    const text = d.on_time
      ? `${made}, before the market opened. It uses ${formatDateShort(d.based_on)}'s closing prices and everything that happened overnight.`
      : `${made}, after the market had already opened, so today is not counted in the track record.`;
    const chip =
      scored.length > 0
        ? `${right} of ${scored.length} right so far`
        : d.market_phase === "pre_open"
        ? "Before the open"
        : d.market_phase === "open"
        ? "Market open"
        : "Market closed";
    const tone =
      scored.length > 0
        ? ("emerald" as const)
        : d.market_phase === "pre_open"
        ? ("sky" as const)
        : ("slate" as const);
    return { showCalls, chip, tone, title, text, total, covered, unavail };
  }

  if (d.market_phase === "holiday" || (d.is_today && d.status === "market_closed")) {
    return {
      showCalls,
      chip: "Market closed",
      tone: "slate" as const,
      title: `The market is closed today${d.reason && d.reason !== "Weekend" ? ` for ${d.reason}` : ""}`,
      text: `${next}.${showCalls ? " The most recent call and its results are below." : ""}`,
      total,
      covered,
    };
  }

  if (!d.is_today && d.market_phase === "pre_open") {
    return {
      showCalls,
      chip: "Arriving soon",
      tone: "sky" as const,
      title: "Today's call arrives at about 8:00 am",
      text: `It is made once the Asian markets have opened.${
        showCalls ? " Until then, the previous call and its results are below." : ""
      }`,
      total,
      covered,
    };
  }

  return {
    showCalls,
    chip: "No forecast today",
    tone: "amber" as const,
    title: "No forecast was published this morning",
    text: `${d.is_today && d.reason ? d.reason + " " : "The price data could not be collected in time. "}${next}.`,
    total,
    covered,
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

  const [search, setSearch] = useState("");
  const [selectedSector, setSelectedSector] = useState("ALL");
  const [selectedDirection, setSelectedDirection] = useState<"ALL" | "HIGHER" | "LOWER">("ALL");
  const [sortBy, setSortBy] = useState<"confidence" | "ticker" | "prob_up" | "prev_close">("confidence");

  // Derive unique sectors from calls
  const sectors = useMemo(() => {
    if (!data?.calls) return [];
    const set = new Set<string>();
    data.calls.forEach((c) => {
      if (c.sector) set.add(c.sector);
    });
    return Array.from(set).sort();
  }, [data?.calls]);

  // Filter and sort calls
  const filteredCalls = useMemo(() => {
    if (!data?.calls) return [];

    let list = [...data.calls];

    // Search query
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter(
        (c) => c.ticker.toLowerCase().includes(q) || c.name.toLowerCase().includes(q)
      );
    }

    // Sector filter
    if (selectedSector !== "ALL") {
      list = list.filter((c) => c.sector === selectedSector);
    }

    // Direction filter
    if (selectedDirection !== "ALL") {
      list = list.filter((c) => c.call === selectedDirection);
    }

    // Sorting
    list.sort((a, b) => {
      if (sortBy === "confidence") {
        const strA = a.available && a.prob_up != null ? Math.abs(a.prob_up - 0.5) : -1;
        const strB = b.available && b.prob_up != null ? Math.abs(b.prob_up - 0.5) : -1;
        return strB - strA;
      }
      if (sortBy === "ticker") {
        return a.ticker.localeCompare(b.ticker);
      }
      if (sortBy === "prob_up") {
        return (b.prob_up ?? 0) - (a.prob_up ?? 0);
      }
      if (sortBy === "prev_close") {
        return (b.prev_close ?? 0) - (a.prev_close ?? 0);
      }
      return 0;
    });

    return list;
  }, [data?.calls, search, selectedSector, selectedDirection, sortBy]);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-32 w-full" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-20" />
          ))}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-44" />
          ))}
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
  const awaiting = data.is_today && data.market_phase !== "pre_open";
  const totalStocks = data.calls.length || 30;
  const coveredStocks = data.calls.filter((c) => c.available).length;
  const coveragePct = totalStocks > 0 ? Math.round((coveredStocks / totalStocks) * 100) : 100;
  const isPartial = coveredStocks < totalStocks;
  const isStale = !data.is_today && data.status === "ok";

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

          {/* Coverage Badge */}
          <span
            className={cn(
              "inline-flex items-center gap-1 text-[11px] font-semibold px-2.5 py-0.5 rounded-full border",
              isPartial
                ? "bg-amber-50 text-amber-800 border-amber-200"
                : "bg-emerald-50 text-emerald-800 border-emerald-200"
            )}
            title={`${coveredStocks} of ${totalStocks} registered Indian equities forecast`}
          >
            {isPartial ? (
              <AlertTriangle className="w-3 h-3 text-amber-600" />
            ) : (
              <CheckCircle2 className="w-3 h-3 text-emerald-600" />
            )}
            {coveredStocks} of {totalStocks} stocks covered ({coveragePct}%)
          </span>

          {/* Model Source Badge */}
          <span className="hidden md:inline-flex items-center gap-1 text-[11px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
            <Layers className="w-3 h-3" />
            {data.model?.model_type || "Pooled 30-Stock Model"}
          </span>

          <span
            className={cn(
              "ml-auto inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-semibold",
              CHIP_TONES[view.tone]
            )}
          >
            {view.chip}
          </span>
        </div>

        <h1 className="mt-3 text-xl sm:text-2xl font-bold tracking-tight text-slate-900">{view.title}</h1>
        <p className="mt-1.5 text-sm text-slate-600 max-w-3xl">{view.text}</p>

        {isStale && (
          <div className="mt-3 flex items-center gap-2 p-3 bg-amber-50/70 border border-amber-200 rounded-md text-xs text-amber-900">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
            <span>
              Showing previous call from {formatDateShort(data.target_date)}. Today&rsquo;s forecast arrives at about 8:00 am IST before market open.
            </span>
          </div>
        )}

        {data.is_today && data.status === "ok" && (
          <p className="mt-3 inline-flex items-center gap-1.5 text-xs text-slate-500">
            <CalendarClock className="w-3.5 h-3.5" />
            Next call: {formatDateShort(data.next_call_date)}, about 8:00 am
          </p>
        )}

        {data.notes.length > 0 && data.status === "ok" && (
          <ul className="mt-3 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-md px-3 py-2 space-y-0.5">
            {data.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        )}
      </section>

      {view.showCalls && <MarketPulse items={data.market} />}

      {view.showCalls && (
        <section aria-label="Stock calls" className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold text-slate-900">
                  {data.is_today ? "Today's calls" : `Previous call, ${formatDateShort(data.target_date)}`}
                </h2>
                <span className="font-mono text-xs font-semibold text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                  {filteredCalls.length} of {totalStocks}
                </span>
              </div>
              <span className="text-[11px] text-slate-500">
                Will each stock open above or below its last close? Evaluated across 30 Indian equities
              </span>
            </div>

            {/* Filter and Search Bar Controls */}
            <div className="flex flex-wrap items-center gap-2">
              {/* Search Box */}
              <div className="relative">
                <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search ticker or name..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-8 pr-7 py-1.5 text-xs bg-white border border-slate-200 rounded-md focus:outline-none focus:ring-1 focus:ring-sky-500 w-36 sm:w-44"
                />
                {search && (
                  <button
                    onClick={() => setSearch("")}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>

              {/* Sector Filter Dropdown */}
              <select
                aria-label="Filter by sector"
                value={selectedSector}
                onChange={(e) => setSelectedSector(e.target.value)}
                className="py-1.5 px-2.5 text-xs bg-white border border-slate-200 rounded-md text-slate-700 focus:outline-none focus:ring-1 focus:ring-sky-500 font-sans"
              >
                <option value="ALL">All Sectors ({totalStocks})</option>
                {sectors.map((sec) => (
                  <option key={sec} value={sec}>
                    {sec}
                  </option>
                ))}
              </select>

              {/* Direction Filter Dropdown */}
              <select
                aria-label="Filter by direction"
                value={selectedDirection}
                onChange={(e) => setSelectedDirection(e.target.value as any)}
                className="py-1.5 px-2.5 text-xs bg-white border border-slate-200 rounded-md text-slate-700 focus:outline-none focus:ring-1 focus:ring-sky-500 font-sans"
              >
                <option value="ALL">All Calls</option>
                <option value="HIGHER">Expected HIGHER</option>
                <option value="LOWER">Expected LOWER</option>
              </select>

              {/* Sorting Dropdown */}
              <select
                aria-label="Sort calls"
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="py-1.5 px-2.5 text-xs bg-white border border-slate-200 rounded-md text-slate-700 focus:outline-none focus:ring-1 focus:ring-sky-500 font-sans"
              >
                <option value="confidence">Sort: Confidence</option>
                <option value="ticker">Sort: Ticker (A-Z)</option>
                <option value="prob_up">Sort: Bullish Probability</option>
                <option value="prev_close">Sort: Price</option>
              </select>
            </div>
          </div>

          {/* Calls Grid */}
          {filteredCalls.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {filteredCalls.map((c) => (
                <CallCard
                  key={c.ticker}
                  call={c}
                  awaitingResult={awaiting}
                  baseRateUp={data.model?.base_rate_up}
                />
              ))}
            </div>
          ) : (
            <div className="bg-white border border-slate-200 rounded-lg p-8 text-center space-y-2">
              <p className="text-sm font-medium text-slate-700">No stock forecasts match the selected filters.</p>
              <p className="text-xs text-slate-500">Try clearing the search query or changing the sector filter.</p>
              <button
                onClick={() => {
                  setSearch("");
                  setSelectedSector("ALL");
                  setSelectedDirection("ALL");
                }}
                className="mt-2 text-xs font-medium text-sky-700 hover:text-sky-900 underline"
              >
                Reset all filters
              </button>
            </div>
          )}
        </section>
      )}

      {card && <TrackRecord card={card} />}

      {/* What this is and what it is not */}
      <section className="bg-slate-50 border border-slate-200 rounded-lg p-5 text-xs text-slate-600 leading-relaxed">
        <h2 className="text-sm font-semibold text-slate-900 mb-1.5">How to read this page</h2>
        <p className="max-w-4xl">
          The Indian market is shut for almost 18 hours between one close and the next open, while markets in the US
          and Asia keep trading. Each morning this page uses what they did overnight to forecast one thing: whether
          each of the 30 registered stocks will <span className="font-semibold text-slate-800">open</span> above or
          below its previous closing price. It is an econometric research study, not a trading tip. The opening jump
          happens the moment trading starts (09:15 IST) during the pre-open auction, so nobody can buy at yesterday&rsquo;s
          price after reading it.
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
