"use client";

import React, { useState } from "react";
import { useSelectedStock } from "@/context/StockContext";
import { usePrediction } from "@/hooks/useStockData";
import { SignalBadge } from "@/components/ui/Badge";
import { MetricCard } from "@/components/ui/MetricCard";
import { MetricCardSkeleton } from "@/components/ui/LoadingSkeleton";
import { ErrorState } from "@/components/ui/ErrorState";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import {
  CoverageCurveChart,
  FeatureImportanceChart,
} from "@/components/charts/PredictionCharts";
import { formatPercent, formatINR } from "@/lib/utils";
import {
  Activity,
  ShieldCheck,
  ChevronDown,
  Layers,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Info,
} from "lucide-react";

export default function PredictionTerminalPage() {
  const { selectedTicker } = useSelectedStock();
  const { data: pred, isLoading, isError, refetch } = usePrediction(selectedTicker);
  const [showAdvanced, setShowAdvanced] = useState(false);

  if (isError) {
    return (
      <ErrorState
        title={`Failed to load prediction for ${selectedTicker}`}
        message="Could not retrieve model artifacts from FastAPI backend. Verify model files exist in models/."
        onRetry={refetch}
      />
    );
  }

  // Derive signal tone for UI tinting
  const signalTone =
    pred?.signal === "UP"
      ? "buy"
      : pred?.signal === "DOWN"
      ? "sell"
      : "neutral";

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white p-5 rounded-lg border border-slate-200/90 shadow-subtle">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-xl font-bold text-slate-900 font-sans tracking-tight">
              Directional Prediction Terminal
            </h1>
            <span className="font-mono text-xs font-bold text-sky-700 bg-sky-50 px-2 py-0.5 rounded border border-sky-100">
              {selectedTicker} · NSE
            </span>
          </div>
          <p className="text-xs text-slate-500 font-sans max-w-2xl">
            Forecasts whether the stock price will move higher or lower over the next 3 trading days, using walk-forward machine learning with confidence-based abstention gating.
          </p>
        </div>

        {pred && (
          <div className="flex flex-col items-end">
            <SignalBadge signal={pred.signal} size="lg" showDescription />
          </div>
        )}
      </div>

      {/* KPI Row with Plain-English Explanations */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {isLoading ? (
          <MetricCardSkeleton />
        ) : pred ? (
          <>
            {/* KPI 1: Directional Signal */}
            <MetricCard
              title="Directional Signal"
              value={pred.signal}
              signalTone={signalTone}
              badge={<SignalBadge signal={pred.signal} size="sm" />}
              subtitle={`3-day forecast horizon`}
              infoTerm="Directional Signal"
              infoExplanation="The model's recommended outlook for the next 3 trading days: UP (bullish), DOWN (bearish), or NO SIGNAL (abstaining because confidence is too low)."
              technicalNote="Binary label target based on 3-day non-overlapping returns. Evaluated with 17-day embargo to eliminate lookahead bias."
            />

            {/* KPI 2: Model Probability */}
            <MetricCard
              title="Probability Upward"
              value={formatPercent(pred.probability_up)}
              subtitle={`Downward: ${formatPercent(pred.probability_down)}`}
              infoTerm="Upward Probability"
              infoExplanation="The estimated likelihood that the stock price closes higher over the next 3 trading days. A score of 50% is completely neutral."
              technicalNote="P(Y=1 | X) estimated by regularized Logistic Regression using 52 stationary financial and momentum features."
            />

            {/* KPI 3: Conviction vs Threshold */}
            <MetricCard
              title="Model Conviction"
              value={formatPercent(pred.confidence)}
              subtitle={`Required Gate: ${formatPercent(pred.confidence_threshold)}`}
              infoTerm="Model Conviction & Gate"
              infoExplanation="How strongly the model leans away from 50/50 neutral. If conviction is smaller than the required threshold, the model abstains to protect you from noisy guesses."
              technicalNote="Conviction = |p - 0.5|. The operating threshold is chosen during walk-forward validation to ensure abstention genuinely improves test set accuracy."
            />

            {/* KPI 4: Historical Edge */}
            <MetricCard
              title="Historical Edge"
              value={
                pred.walk_forward_edge !== undefined
                  ? `${pred.walk_forward_edge >= 0 ? "+" : ""}${formatPercent(pred.walk_forward_edge)}`
                  : "N/A"
              }
              subtitle={`Walk-forward acc: ${formatPercent(pred.walk_forward_accuracy || 0)}`}
              infoTerm="Historical Edge"
              infoExplanation="How much better the model performed over simple guessing across 5 separate historical walk-forward market periods."
              technicalNote="Edge = Model Accuracy - Majority Baseline across chronological walk-forward splits."
            />
          </>
        ) : null}
      </div>

      {/* Conviction Gauge Banner */}
      {pred && (
        <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono">
                Model Conviction Spectrum
              </span>
              <InfoTooltip
                term="Conviction Spectrum"
                explanation="Visual gauge comparing downward probability (left, red) against upward probability (right, green). The neutral point is 50%."
              />
            </div>
            <span className="text-xs font-mono text-slate-500">
              Operating Gate: ±{formatPercent(pred.confidence_threshold)}
            </span>
          </div>

          {/* Progress bar */}
          <div className="relative w-full h-5 bg-slate-100 rounded-full overflow-hidden border border-slate-200 flex">
            {/* Bearish region */}
            <div
              className="h-full bg-rose-500 transition-all duration-500"
              style={{ width: `${pred.probability_down * 100}%` }}
              title={`Downward Probability: ${formatPercent(pred.probability_down)}`}
            />
            {/* Bullish region */}
            <div
              className="h-full bg-emerald-500 transition-all duration-500"
              style={{ width: `${pred.probability_up * 100}%` }}
              title={`Upward Probability: ${formatPercent(pred.probability_up)}`}
            />
            {/* 50% Center Marker */}
            <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-slate-900 z-10 -translate-x-1/2" />
          </div>

          <div className="flex justify-between text-[11px] font-mono text-slate-500 mt-1.5">
            <span className="text-rose-600 font-bold">Bearish ({formatPercent(pred.probability_down)})</span>
            <span className="text-slate-400">50% Neutral Baseline</span>
            <span className="text-emerald-600 font-bold">Bullish ({formatPercent(pred.probability_up)})</span>
          </div>
        </div>
      )}

      {/* Two Analytical Charts */}
      {pred && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Chart 1: Feature Influences */}
          <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-1.5">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                    Key Drivers (Feature Weights)
                  </h3>
                  <InfoTooltip
                    term="Feature Weights"
                    explanation="Shows which market indicators are currently pushing the model toward a Bullish (green, positive) or Bearish (red, negative) prediction."
                    technicalNote="Standardized model regression coefficients. Features are engineered as stationary ratios to prevent non-stationary drift."
                  />
                </div>
                <span className="text-[11px] text-slate-400 font-mono">{pred.features_count || 52} total features</span>
              </div>
              <p className="text-xs text-slate-500 font-sans mb-3">
                Green bars push toward an upward forecast; red bars push downward.
              </p>
              <FeatureImportanceChart features={pred.top_features || []} />
            </div>
          </div>

          {/* Chart 2: Coverage vs Accuracy Operating Curve */}
          <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-1.5">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                    Coverage vs. Accuracy Tradeoff
                  </h3>
                  <InfoTooltip
                    term="Coverage vs. Accuracy Tradeoff"
                    explanation="Demonstrates that when the model is allowed to abstain on days with weak evidence, the accuracy on the remaining days increases."
                    technicalNote="Out-of-sample coverage-accuracy curve. Operating point selected with a strict floor (min 20% coverage) to prevent overfitting."
                  />
                </div>
                <span className="text-[11px] text-slate-400 font-mono">Walk-Forward Curve</span>
              </div>
              <p className="text-xs text-slate-500 font-sans mb-3">
                Higher confidence gates yield higher prediction accuracy by filtering noisy market days.
              </p>
              <CoverageCurveChart
                data={pred.coverage_curve || []}
                operatingThreshold={pred.confidence_threshold}
              />
            </div>
          </div>
        </div>
      )}

      {/* Progressive Disclosure: Advanced Statistical Validation */}
      <div className="bg-white border border-slate-200/90 rounded-lg shadow-subtle overflow-hidden">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-5 py-3.5 flex items-center justify-between text-left hover:bg-slate-50 transition-colors"
        >
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-sky-600" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
              Advanced Research & Statistical Validation Details
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500 font-sans">
            <span>{showAdvanced ? "Hide technical parameters" : "Show technical parameters"}</span>
            <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
          </div>
        </button>

        {showAdvanced && pred && (
          <div className="p-5 border-t border-slate-100 bg-slate-50/50 space-y-4 text-xs font-sans">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 font-mono">
              <div className="p-3 rounded bg-white border border-slate-200">
                <span className="text-slate-500 block text-[11px]">Model Architecture</span>
                <span className="font-bold text-slate-900 text-sm mt-0.5 block">{pred.model_name || "Regularized Logistic"}</span>
                <span className="text-[10px] text-slate-400">Linear estimator robust across sampling offsets</span>
              </div>

              <div className="p-3 rounded bg-white border border-slate-200">
                <span className="text-slate-500 block text-[11px]">Embargo Window</span>
                <span className="font-bold text-slate-900 text-sm mt-0.5 block">17 Trading Days</span>
                <span className="text-[10px] text-slate-400">Guarantees zero overlapping label leakage</span>
              </div>

              <div className="p-3 rounded bg-white border border-slate-200">
                <span className="text-slate-500 block text-[11px]">Validation Protocol</span>
                <span className="font-bold text-slate-900 text-sm mt-0.5 block">5-Fold Walk-Forward</span>
                <span className="text-[10px] text-slate-400">Chronological training without future peeking</span>
              </div>
            </div>

            <div className="rounded bg-sky-50 border border-sky-200 p-3.5 text-sky-900 text-xs leading-relaxed">
              <span className="font-bold block mb-1">Academic Integrity Note on Machine Learning in Finance:</span>
              Unlike tree ensembles which flipped negative on different sampling offsets, regularized linear models maintained consistent out-of-sample edge. Every prediction reported by this terminal comes from genuine out-of-sample walk-forward validation.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
