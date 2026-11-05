"use client";

import React, { useState, useMemo } from "react";
import { useSelectedStock } from "@/context/StockContext";
import { useStock, useDataStatus, useNews } from "@/hooks/useStockData";
import { MetricCard } from "@/components/ui/MetricCard";
import { MetricCardSkeleton } from "@/components/ui/LoadingSkeleton";
import { ErrorState } from "@/components/ui/ErrorState";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import { FreshnessBadge } from "@/components/ui/Badge";
import {
  SentimentPriceTimelineChart,
  LeadLagChart,
} from "@/components/charts/ExplainCharts";
import { formatINR } from "@/lib/utils";
import {
  Compass,
  Newspaper,
  Search,
  Filter,
  Calendar,
  Layers,
  ChevronDown,
  Info,
  ExternalLink,
  Tag,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";

export default function ExplainPage() {
  const { selectedTicker } = useSelectedStock();

  const { data: stock } = useStock(selectedTicker);
  const { data: status } = useDataStatus(selectedTicker);
  const { data: news, isLoading, isError, refetch } = useNews(selectedTicker, 150);

  // Filter states for Section 3: News Explorer
  const [searchQuery, setSearchQuery] = useState("");
  const [sentimentFilter, setSentimentFilter] = useState<"all" | "positive" | "negative" | "neutral">("all");
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState<string>("all");
  const [showResearchNotes, setShowResearchNotes] = useState(false);

  if (isError) {
    return (
      <ErrorState
        title={`Failed to load news sentiment for ${selectedTicker}`}
        message="Could not load company news sentiment records from data/processed/news/ via FastAPI."
        onRetry={refetch}
      />
    );
  }

  // Derive available sources for filtering
  const sources = useMemo(() => {
    if (!news?.recent_articles) return [];
    const set = new Set<string>();
    news.recent_articles.forEach((a) => {
      if (a.source) set.add(a.source);
    });
    return Array.from(set);
  }, [news]);

  // Filtered articles list
  const filteredArticles = useMemo(() => {
    if (!news?.recent_articles) return [];
    return news.recent_articles.filter((item) => {
      // Date filter
      if (selectedDate && item.trade_date !== selectedDate) {
        return false;
      }
      // Sentiment filter
      const score = item.targeted_score ?? item.plain_score ?? 0;
      if (sentimentFilter === "positive" && score <= 0.05) return false;
      if (sentimentFilter === "negative" && score >= -0.05) return false;
      if (sentimentFilter === "neutral" && (score > 0.05 || score < -0.05)) return false;

      // Source filter
      if (sourceFilter !== "all" && item.source !== sourceFilter) return false;

      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesText = item.text.toLowerCase().includes(q);
        const matchesEntity = (item.entity || "").toLowerCase().includes(q);
        if (!matchesText && !matchesEntity) return false;
      }

      return true;
    });
  }, [news, selectedDate, sentimentFilter, sourceFilter, searchQuery]);

  return (
    <div className="space-y-6">
      {/* SECTION 1: STOCK & NEWS OVERVIEW */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-6 shadow-subtle flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="font-mono text-xs font-bold text-sky-700 bg-sky-50 px-2 py-0.5 rounded border border-sky-100">
              {selectedTicker} · NSE
            </span>
            <span className="text-xs text-slate-500 font-sans">
              {stock?.display_name || selectedTicker}
            </span>
            <FreshnessBadge isLive={false} date={status?.news_last_date} />
          </div>

          <h1 className="text-2xl font-bold text-slate-900 tracking-tight font-sans">
            Why Did It Move? (Explainability & Sentiment)
          </h1>
          <p className="mt-1 text-xs text-slate-500 max-w-2xl font-sans">
            Investigates how corporate announcements, regulatory filings, and financial news coincide with price changes, using company-targeted FinBERT natural language processing.
          </p>
        </div>

        <div className="flex flex-col md:items-end justify-center bg-slate-50 md:bg-transparent p-3 md:p-0 rounded border md:border-0 border-slate-200">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 font-mono">
            News Archive Horizon
          </span>
          <div className="text-sm font-bold text-slate-900 font-mono mt-0.5">
            Up to {status?.news_last_date || "2025-12-31"}
          </div>
          <span className="text-[11px] text-slate-400 font-sans">
            {news?.total_articles ? `${news.total_articles.toLocaleString()} headlines analyzed` : "Loading archive..."}
          </span>
        </div>
      </div>

      {/* KPI Overview Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {isLoading ? (
          <MetricCardSkeleton />
        ) : news ? (
          <>
            <MetricCard
              title="Analyzed Headlines"
              value={news.total_articles.toLocaleString()}
              subtitle="Entity-targeted dataset"
              infoTerm="Analyzed Headlines"
              infoExplanation="Total number of news articles and filings naming this specific company that have been parsed and classified by FinBERT."
              icon={<Newspaper className="w-4 h-4 text-sky-600" />}
            />
            <MetricCard
              title="Average Sentiment"
              value={news.average_sentiment >= 0 ? `+${news.average_sentiment.toFixed(3)}` : news.average_sentiment.toFixed(3)}
              subtitle="Polarity scale: -1.0 to +1.0"
              infoTerm="Average FinBERT Sentiment"
              infoExplanation="The average entity-targeted sentiment score across the entire news archive. Near zero indicates balanced reporting."
              technicalNote="Computed from FinBERT softmax output: P(positive) - P(negative)."
            />
            <MetricCard
              title="Positive News"
              value={news.positive_count.toLocaleString()}
              change={Math.round((news.positive_count / (news.total_articles || 1)) * 1000) / 10}
              changeSuffix="% of total"
              infoTerm="Positive News Headlines"
              infoExplanation="Headlines classified as net positive (> +0.05 polarity) such as earnings beats, contract wins, or strategic expansions."
            />
            <MetricCard
              title="Negative News"
              value={news.negative_count.toLocaleString()}
              change={Math.round((news.negative_count / (news.total_articles || 1)) * 1000) / 10}
              changeSuffix="% of total"
              infoTerm="Negative News Headlines"
              infoExplanation="Headlines classified as net negative (< -0.05 polarity) such as legal disputes, downgrades, or margin contractions."
            />
          </>
        ) : null}
      </div>

      {/* SECTION 2: PRICE AND NEWS INTERACTIVE TIMELINE */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
          <div>
            <div className="flex items-center gap-1.5">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                Interactive Stock Price & News Sentiment Timeline
              </h2>
              <InfoTooltip
                term="Interactive Price & News Timeline"
                explanation="Combines historical closing prices (blue line) with sentiment events. Green dots indicate positive news, red dots indicate negative news, grey dots indicate neutral news. Click any dot to filter headlines below."
              />
            </div>
            <p className="text-xs text-slate-500 font-sans mt-0.5">
              Hover over colored event markers to inspect headlines; click any date to filter the news explorer.
            </p>
          </div>

          {selectedDate && (
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-sky-700 bg-sky-50 px-2.5 py-1 rounded border border-sky-200 flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5" />
                Filtered: {selectedDate}
              </span>
              <button
                type="button"
                onClick={() => setSelectedDate(null)}
                className="text-xs text-slate-500 hover:text-slate-800 underline"
              >
                Clear
              </button>
            </div>
          )}
        </div>

        <div className="flex items-center gap-4 text-[11px] font-mono text-slate-500 my-2">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" /> Positive Sentiment Event
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" /> Negative Sentiment Event
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-400 inline-block" /> Neutral Announcement
          </span>
        </div>

        <SentimentPriceTimelineChart
          timeline={news?.timeline || []}
          onSelectDate={(date) => setSelectedDate(date)}
        />
      </div>

      {/* SECTION 5: LEAD-LAG ANALYSIS */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
              Statistical Lead-Lag Analysis (News vs. Price Movement)
            </h2>
            <InfoTooltip
              term="Lead-Lag Analysis"
              explanation="Measures whether news sentiment anticipates future returns (Lead, negative lags) or merely reports past price changes after they happened (Lag, positive lags)."
              technicalNote="Spearman rank correlation across ±5 trading days. Same-day (lag 0) is strong (ρ = 0.17); however, forward-looking predictive correlation at lag -1 drops to ρ = 0.05 and vanishes once retrospective price recaps are filtered out."
            />
          </div>
          <span className="text-[11px] font-mono text-slate-400">Spearman Rank Correlation (ρ)</span>
        </div>
        <p className="text-xs text-slate-500 font-sans mb-3">
          Notice the sharp drop from Day 0 (concurrent reporting) to Prior Days. Markets price information rapidly; news rarely provides a multiday leading edge.
        </p>

        <LeadLagChart leadLag={news?.lead_lag || []} />

        <div className="mt-3 p-3 rounded bg-slate-50 border border-slate-200 text-xs text-slate-600 font-sans flex items-start gap-2">
          <Info className="w-4 h-4 text-sky-600 shrink-0 mt-0.5" />
          <span>
            <strong className="text-slate-800">Critical Academic Finding: </strong>
            High same-day correlation ($Day\ 0$) does <em>not</em> mean sentiment can be traded profitably. Most news articles published during or after market hours are retrospective summaries explaining why the market already moved.
          </span>
        </div>
      </div>

      {/* SECTION 3: NEWS EXPLORER */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
              Company-Targeted News Explorer
            </h2>
            <p className="text-xs text-slate-500 font-sans">
              Filter by date, publication source, or sentiment tone.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500">
            Showing {filteredArticles.length} of {news?.recent_articles.length || 0} loaded headlines
          </span>
        </div>

        {/* Filters Toolbar */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* Search Input */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-slate-200 bg-slate-50">
            <Search className="w-3.5 h-3.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search headline keywords..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full text-xs bg-transparent outline-hidden text-slate-900 placeholder:text-slate-400 font-sans"
            />
          </div>

          {/* Sentiment Filter Tabs */}
          <div className="flex items-center rounded-md border border-slate-200 bg-slate-50 p-0.5 text-xs font-mono">
            {[
              { key: "all", label: "All" },
              { key: "positive", label: "Bullish (+)" },
              { key: "negative", label: "Bearish (-)" },
              { key: "neutral", label: "Neutral" },
            ].map((tab) => (
              <button
                key={tab.key}
                type="button"
                onClick={() => setSentimentFilter(tab.key as any)}
                className={`flex-1 py-1 rounded text-center transition-colors ${
                  sentimentFilter === tab.key
                    ? "bg-white text-slate-900 shadow-xs font-bold"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Source Filter Dropdown */}
          <select
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value)}
            className="px-3 py-1.5 rounded-md border border-slate-200 bg-slate-50 text-xs font-sans text-slate-800 outline-hidden"
          >
            <option value="all">All News Sources</option>
            {sources.map((src) => (
              <option key={src} value={src}>
                {src}
              </option>
            ))}
          </select>
        </div>

        {/* Articles List */}
        <div className="divide-y divide-slate-100 max-h-[500px] overflow-y-auto">
          {filteredArticles.length > 0 ? (
            filteredArticles.map((item, idx) => {
              const score = item.targeted_score ?? item.plain_score ?? 0;
              const isPos = score > 0.05;
              const isNeg = score < -0.05;

              return (
                <div key={idx} className="py-3 hover:bg-slate-50/60 transition-colors px-1 rounded flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                  <div className="space-y-1 max-w-3xl">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-mono text-[11px] text-slate-500 flex items-center gap-1">
                        <Calendar className="w-3 h-3 text-slate-400" />
                        {item.trade_date}
                      </span>
                      {item.source && (
                        <span className="text-[10px] bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded font-sans font-medium">
                          {item.source}
                        </span>
                      )}
                      {item.entity && (
                        <span className="text-[10px] bg-sky-50 text-sky-700 px-1.5 py-0.5 rounded font-mono font-semibold">
                          Target: {item.entity}
                        </span>
                      )}
                      {item.price_report && (
                        <span className="text-[10px] bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded font-sans italic">
                          Price Recap
                        </span>
                      )}
                    </div>
                    <p className="text-slate-900 font-sans font-medium text-xs leading-relaxed">
                      {item.text}
                    </p>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <span
                      className={`font-mono text-xs font-bold px-2.5 py-1 rounded border flex items-center gap-1 ${
                        isPos
                          ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                          : isNeg
                          ? "bg-rose-50 text-rose-800 border-rose-300"
                          : "bg-slate-100 text-slate-700 border-slate-300"
                      }`}
                    >
                      {isPos && <TrendingUp className="w-3 h-3 text-emerald-600" />}
                      {isNeg && <TrendingDown className="w-3 h-3 text-rose-600" />}
                      {!isPos && !isNeg && <Minus className="w-3 h-3 text-slate-500" />}
                      {score >= 0 ? `+${score.toFixed(3)}` : score.toFixed(3)}
                    </span>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="py-8 text-center text-xs text-slate-400 font-sans">
              No headlines match the selected date or search filter.
            </div>
          )}
        </div>
      </div>

      {/* SECTION 4 & 6: RESEARCH FINDINGS & METHODOLOGY ACCORDION */}
      <div className="bg-white border border-slate-200/90 rounded-lg shadow-subtle overflow-hidden">
        <button
          type="button"
          onClick={() => setShowResearchNotes(!showResearchNotes)}
          className="w-full px-5 py-3.5 flex items-center justify-between text-left hover:bg-slate-50 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Compass className="w-4 h-4 text-teal-600" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
              Research Findings: Why Sentiment Doesn&apos;t Easily Beat the Market
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500 font-sans">
            <span>{showResearchNotes ? "Hide research notes" : "Read empirical findings"}</span>
            <ChevronDown className={`w-4 h-4 transition-transform ${showResearchNotes ? "rotate-180" : ""}`} />
          </div>
        </button>

        {showResearchNotes && (
          <div className="p-5 border-t border-slate-100 bg-slate-50/40 space-y-4 text-xs font-sans text-slate-700 leading-relaxed">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-3.5 rounded bg-white border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-1 flex items-center gap-1.5">
                  <Tag className="w-3.5 h-3.5 text-sky-600" />
                  1. Company-Targeted vs. Generic FinBERT
                </h4>
                <p className="text-slate-600 text-[11px]">
                  Generic financial news sentiment tags market-wide mood (e.g. inflation, Fed rates). By fine-tuning FinBERT on entity-level named entities naming {selectedTicker}, the signal isolates true corporate announcements from general market noise.
                </p>
              </div>

              <div className="p-3.5 rounded bg-white border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-1 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                  2. Circular Lookahead in &ldquo;Price Recap&rdquo; News
                </h4>
                <p className="text-slate-600 text-[11px]">
                  Empirical audit revealed that up to 30% of financial headlines are retrospective price recaps (e.g. <em>&ldquo;TCS plunges 4% amid tech selloff&rdquo;</em>). Feeding these to a classifier creates false circular lookahead. Filtering them out confirms that true forward-looking news is rare.
                </p>
              </div>
            </div>

            <div className="p-3 rounded bg-amber-50 border border-amber-200 text-amber-900 text-xs">
              <strong className="block mb-0.5">Summary of Ablation Study:</strong>
              In our walk-forward tests, adding raw news sentiment to technical indicators provided only a +0.1% to +0.3% incremental gain. Natural language sentiment is valuable for explainability (understanding why prices moved), but should be confirmed with technical trend filters before executing trades.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
