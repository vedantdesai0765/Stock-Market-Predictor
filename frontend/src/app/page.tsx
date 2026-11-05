"use client";

import React from "react";
import { useSelectedStock } from "@/context/StockContext";
import {
  useStock,
  useDataStatus,
  usePrediction,
  useForecast,
  useLatestSignal,
} from "@/hooks/useStockData";
import { MetricCard } from "@/components/ui/MetricCard";
import { SignalBadge, FreshnessBadge, RegimeBadge } from "@/components/ui/Badge";
import { ModuleCard } from "@/components/ui/ModuleCard";
import { MetricCardSkeleton } from "@/components/ui/LoadingSkeleton";
import { ErrorState } from "@/components/ui/ErrorState";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import { formatINR, formatPercent } from "@/lib/utils";
import {
  Activity,
  SunMedium,
  Compass,
  Zap,
  Info,
  Calendar,
  Database,
  TrendingUp,
  TrendingDown,
  Pause,
  MinusCircle,
  HelpCircle,
} from "lucide-react";

export default function OverviewPage() {
  const { selectedTicker } = useSelectedStock();

  // Queries
  const { data: stock, isLoading: isStockLoading, isError: isStockError, refetch: refetchStock } = useStock(selectedTicker);
  const { data: status, isLoading: isStatusLoading, isError: isStatusError, refetch: refetchStatus } = useDataStatus(selectedTicker);
  const { data: prediction, isLoading: isPredLoading } = usePrediction(selectedTicker);
  const { data: forecast, isLoading: isForecastLoading } = useForecast(selectedTicker);
  const { data: signal, isLoading: isSignalLoading } = useLatestSignal(selectedTicker, "A", 15.0);

  const isLoadingAny = isStockLoading || isStatusLoading;

  if (isStockError || isStatusError) {
    return (
      <ErrorState
        title={`Failed to load data for ${selectedTicker}`}
        message="Unable to fetch stock metadata or data status from the FastAPI backend. Ensure the backend server is running on http://127.0.0.1:8000."
        onRetry={() => {
          refetchStock();
          refetchStatus();
        }}
      />
    );
  }

  // Derive latest price and date from available real API data
  const latestPrice = signal?.close ?? prediction?.close ?? null;
  const latestDate = signal?.date ?? prediction?.date ?? status?.raw_data_last_date ?? "N/A";

  // Derive signal tones
  const signalTone =
    signal?.signal_label === "BUY"
      ? "buy"
      : signal?.signal_label === "SELL"
      ? "sell"
      : signal?.signal_label === "HOLD"
      ? "hold"
      : "neutral";

  const predTone =
    prediction?.signal === "UP"
      ? "buy"
      : prediction?.signal === "DOWN"
      ? "sell"
      : "neutral";

  return (
    <div className="space-y-6">
      {/* Stock Hero Section */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-6 shadow-subtle flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="font-mono text-xs font-bold text-sky-700 bg-sky-50 px-2 py-0.5 rounded border border-sky-100">
              {selectedTicker} · NSE
            </span>
            <span className="text-xs text-slate-500 font-sans">
              {stock?.symbol || `${selectedTicker}.NS`}
            </span>
            <FreshnessBadge isLive={status?.is_live_market} date={status?.raw_data_last_date} />
          </div>

          <h1 className="text-2xl font-bold text-slate-900 tracking-tight font-sans">
            {stock?.display_name || selectedTicker}
          </h1>

          <p className="mt-1 text-xs text-slate-500 max-w-2xl font-sans">
            {stock?.notes ||
              "Institutional predictive modeling and quantitative strategy evaluation on Indian NSE equity."}
          </p>
        </div>

        {/* Price & As-of Box */}
        <div className="flex flex-col md:items-end justify-center bg-slate-50 md:bg-transparent p-3 md:p-0 rounded border md:border-0 border-slate-200">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-600 font-mono">
            Latest Close ({latestDate})
          </span>
          <div className="text-2xl md:text-3xl font-bold text-slate-900 font-mono tracking-tight my-0.5">
            {latestPrice !== null ? formatINR(latestPrice) : "Loading..."}
          </div>
          <span className="text-[11px] text-slate-600 font-sans flex items-center gap-1">
            <Calendar className="w-3 h-3 text-slate-600" />
            As of last available market session
          </span>
        </div>
      </div>

      {/* KPI Grid with Strict Color System & Intuitive Tooltips */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* KPI 1: Actionable Trading Signal */}
        {isSignalLoading ? (
          <MetricCardSkeleton />
        ) : signal ? (
          <MetricCard
            title="Actionable Signal"
            value={signal.signal_label}
            signalTone={signalTone}
            badge={<SignalBadge signal={signal.signal_label} size="md" />}
            subtitle={`Net Expectancy: ${signal.net_expectancy >= 0 ? "+" : ""}${formatPercent(signal.net_expectancy)}`}
            infoTerm="Actionable Trading Signal"
            infoExplanation="Evidence-based trading signal after deducting 15 bps round-trip statutory costs. BUY (enter long), SELL (exit long), or HOLD (stay in cash because returns do not overcome costs)."
            technicalNote="Uses isotonic calibrated probabilities and realistic Indian transaction friction (STT, brokerage, stamp duty, slippage)."
            icon={<Zap className="w-4 h-4 text-amber-500" />}
          />
        ) : (
          <MetricCard title="Actionable Signal" value="Unavailable" subtitle="No signal computed" />
        )}

        {/* KPI 2: Walk-Forward Prediction */}
        {isPredLoading ? (
          <MetricCardSkeleton />
        ) : prediction ? (
          <MetricCard
            title="3-Day Direction"
            value={prediction.signal}
            signalTone={predTone}
            badge={<SignalBadge signal={prediction.signal} size="md" />}
            subtitle={`Conviction: ${formatPercent(prediction.confidence)} (Gate: ${formatPercent(prediction.confidence_threshold)})`}
            infoTerm="3-Day Directional Forecast"
            infoExplanation="Predicts whether price will move UP or DOWN over the next 3 trading days. If the model is not confident enough, it outputs NO SIGNAL to protect you from noise."
            technicalNote="Regularized logistic regression with abstention operating point selected on out-of-sample data."
            icon={<Activity className="w-4 h-4 text-sky-500" />}
          />
        ) : (
          <MetricCard title="3-Day Direction" value="Unavailable" subtitle="Model not loaded" />
        )}

        {/* KPI 3: Next-Morning Forecast */}
        {isForecastLoading ? (
          <MetricCardSkeleton />
        ) : forecast ? (
          <MetricCard
            title="Morning Open Drift"
            value={forecast.best_accuracy ? formatPercent(forecast.best_accuracy) : "N/A"}
            subtitle={`Model: ${forecast.best_model || "LightGBM"} (${forecast.best_edge !== undefined ? (forecast.best_edge >= 0 ? "+" : "") + formatPercent(forecast.best_edge) : "0%"} edge)`}
            infoTerm="Next-Morning Forecast"
            infoExplanation="Predicts tomorrow morning's opening gap direction at 09:15 AM IST using overnight international cues and corporate announcements."
            technicalNote="Evaluated across 12 years of out-of-sample historical morning sessions (2013-2025)."
            icon={<SunMedium className="w-4 h-4 text-amber-500" />}
          />
        ) : (
          <MetricCard title="Morning Open Drift" value="Unavailable" subtitle="Evaluation missing" />
        )}

        {/* KPI 4: Market Regime */}
        {isSignalLoading ? (
          <MetricCardSkeleton />
        ) : signal ? (
          <MetricCard
            title="Market Regime"
            value={signal.market_regime.replace(/_/g, " ")}
            badge={<RegimeBadge regime={signal.market_regime} />}
            subtitle={`RSI: ${signal.rsi.toFixed(1)} · MACD: ${signal.macd_hist.toFixed(3)}`}
            infoTerm="Market Regime"
            infoExplanation="Identifies the prevailing trend and volatility environment: Trending Bull, Trending Bear, High Volatility, or Sideways Choppy."
            technicalNote="Based on 20/50-day moving averages alignment and expanding historical volatility quantiles without lookahead bias."
            icon={<Compass className="w-4 h-4 text-teal-500" />}
          />
        ) : (
          <MetricCard title="Market Regime" value="Unavailable" subtitle="Regime filter pending" />
        )}
      </div>

      {/* Signal System Explanatory Guide Banner */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-4 shadow-subtle">
        <div className="flex items-center gap-1.5 mb-2.5">
          <HelpCircle className="w-4 h-4 text-sky-600" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
            How to Interpret Platform Signals (Accessibility Guide)
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs font-sans">
          <div className="p-3 rounded-md bg-emerald-50/70 border border-emerald-200">
            <div className="flex items-center gap-1.5 font-bold text-emerald-800 mb-1">
              <TrendingUp className="w-4 h-4 text-emerald-600" />
              <span>BUY / UP (Green)</span>
            </div>
            <p className="text-emerald-950 text-[11px] leading-relaxed">
              Model probability and estimated return exceed transaction costs. A position entry is statistically justified.
            </p>
          </div>

          <div className="p-3 rounded-md bg-rose-50/70 border border-rose-200">
            <div className="flex items-center gap-1.5 font-bold text-rose-800 mb-1">
              <TrendingDown className="w-4 h-4 text-rose-600" />
              <span>SELL / DOWN (Red)</span>
            </div>
            <p className="text-rose-950 text-[11px] leading-relaxed">
              Downward price probability indicates exiting existing positions to preserve capital.
            </p>
          </div>

          <div className="p-3 rounded-md bg-amber-50/70 border border-amber-200">
            <div className="flex items-center gap-1.5 font-bold text-amber-800 mb-1">
              <Pause className="w-4 h-4 text-amber-600" />
              <span>HOLD (Amber)</span>
            </div>
            <p className="text-amber-950 text-[11px] leading-relaxed">
              Deliberate pause. Expected gross return is smaller than 15 bps friction or indicators conflict. Stay in cash.
            </p>
          </div>

          <div className="p-3 rounded-md bg-slate-100/80 border border-slate-300">
            <div className="flex items-center gap-1.5 font-bold text-slate-800 mb-1">
              <MinusCircle className="w-4 h-4 text-slate-500" />
              <span>NO SIGNAL (Grey)</span>
            </div>
            <p className="text-slate-900 text-[11px] leading-relaxed">
              Model conviction is below the operating threshold. The model abstains rather than making a random guess.
            </p>
          </div>
        </div>
      </div>

      {/* Research Modules Navigation Section */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-base font-bold text-slate-900 font-sans tracking-tight">
              Research & Analytics Modules
            </h2>
            <p className="text-xs text-slate-600 font-sans">
              Access the specialized engines powering directional inference, overnight forecasting, and backtested signals.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Module 1: Prediction Terminal */}
          <ModuleCard
            title="Directional Prediction Terminal"
            tag="Phase 4.1 Model Engine"
            href="/prediction"
            icon={<Activity className="w-4 h-4" />}
            description="3-day non-overlapping directional predictions, confidence-based abstention curve, walk-forward fold evaluations, and feature weight rankings."
            statusText={
              prediction
                ? `Edge: ${prediction.walk_forward_edge ? (prediction.walk_forward_edge >= 0 ? "+" : "") + formatPercent(prediction.walk_forward_edge) : "N/A"} | ${prediction.features_count || 52} features`
                : "Checking status..."
            }
            badge={<SignalBadge signal={prediction?.signal || "NO SIGNAL"} size="sm" />}
          />

          {/* Module 2: Next-Morning Forecast */}
          <ModuleCard
            title="Next-Morning Forecast"
            tag="Overnight Gap Engine"
            href="/forecast"
            icon={<SunMedium className="w-4 h-4" />}
            description="Predicts overnight market gap direction at market opening using international cues, technical drift, and overnight company news items."
            statusText={
              forecast
                ? `Best: ${forecast.best_model} (${formatPercent(forecast.best_accuracy || 0)} acc)`
                : "Checking status..."
            }
            badge={
              <span className="text-[10px] font-mono font-medium text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                12-Year History
              </span>
            }
          />

          {/* Module 3: Why Did It Move? */}
          <ModuleCard
            title="Why Did It Move? (Explainability)"
            tag="FinBERT Sentiment Engine"
            href="/explain"
            icon={<Compass className="w-4 h-4" />}
            description="Entity-targeted FinBERT sentiment analysis isolating company-specific headlines from market noise, with statistical lead-lag cross-correlations."
            statusText={
              status?.news_count
                ? `${status.news_count} news articles recorded`
                : "News feed ready"
            }
            badge={
              <span className="text-[10px] font-mono font-medium text-sky-700 bg-sky-50 px-2 py-0.5 rounded border border-sky-200">
                Entity Targeted
              </span>
            }
          />

          {/* Module 4: Actionable Signals */}
          <ModuleCard
            title="Actionable Trading Signals & Backtesting"
            tag="Quantitative Execution"
            href="/signals"
            icon={<Zap className="w-4 h-4" />}
            description="Evidence-based BUY/SELL/HOLD decision engine combining calibrated probabilities, technical trend filters, and Indian statutory transaction friction."
            statusText={
              signal
                ? `Current: ${signal.signal_label} (Net: ${formatPercent(signal.net_expectancy)})`
                : "Ready"
            }
            badge={<SignalBadge signal={signal?.signal_label || "HOLD"} size="sm" />}
          />
        </div>
      </div>

      {/* Data Lineage & Freshness Transparency Card */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-4 h-4 text-slate-500" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono">
            Data Lineage & Freshness Specifications
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs mb-4">
          <div className="p-3 rounded bg-slate-50 border border-slate-200">
            <span className="text-slate-600 block text-[11px] font-medium">Raw Market Dataset</span>
            <span className="font-mono font-bold text-slate-800 text-sm block mt-0.5">
              {status?.raw_data_last_date || "N/A"}
            </span>
            <span className="text-[11px] text-slate-600">
              {status?.raw_data_rows ? `${status.raw_data_rows.toLocaleString()} daily records` : "Missing file"}
            </span>
          </div>

          <div className="p-3 rounded bg-slate-50 border border-slate-200">
            <span className="text-slate-600 block text-[11px] font-medium">Out-of-Sample Boundary</span>
            <span className="font-mono font-bold text-slate-800 text-sm block mt-0.5">
              {status?.oof_eval_last_date || "N/A"}
            </span>
            <span className="text-[11px] text-slate-600">Walk-forward evaluation cutoff</span>
          </div>

          <div className="p-3 rounded bg-slate-50 border border-slate-200">
            <span className="text-slate-600 block text-[11px] font-medium">Model Training Timestamp</span>
            <span className="font-mono font-bold text-slate-800 text-sm block mt-0.5 truncate">
              {status?.model_trained_at ? status.model_trained_at.replace("T", " ") : "Not trained"}
            </span>
            <span className="text-[11px] text-slate-600">Phase 4.1 regularized logistic</span>
          </div>

          <div className="p-3 rounded bg-slate-50 border border-slate-200">
            <span className="text-slate-600 block text-[11px] font-medium">News Sentiment Horizon</span>
            <span className="font-mono font-bold text-slate-800 text-sm block mt-0.5">
              {status?.news_last_date || "N/A"}
            </span>
            <span className="text-[11px] text-slate-600">
              {status?.news_count ? `${status.news_count.toLocaleString()} articles recorded` : "No articles"}
            </span>
          </div>
        </div>

        <div className="rounded bg-amber-50/70 border border-amber-200/80 p-3 flex items-start gap-2.5 text-xs text-amber-900 font-sans">
          <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold">Temporal Separation Integrity: </span>
            {status?.status_summary ||
              "All signals and predictions shown above are evaluated on rigorous out-of-sample historical test sets with embargo periods. Real-time trading broker API execution is intentionally disabled in this research release."}
          </div>
        </div>
      </div>
    </div>
  );
}
