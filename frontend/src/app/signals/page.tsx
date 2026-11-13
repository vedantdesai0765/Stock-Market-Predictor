"use client";

import React, { useState, useMemo } from "react";
import { useSelectedStock } from "@/context/StockContext";
import { useLatestSignal, useBacktest, useStock } from "@/hooks/useStockData";
import { SignalBadge, RegimeBadge, FreshnessBadge } from "@/components/ui/Badge";
import { MetricCard } from "@/components/ui/MetricCard";
import { MetricCardSkeleton } from "@/components/ui/LoadingSkeleton";
import { ErrorState } from "@/components/ui/ErrorState";
import { SimulatorNotice } from "@/components/ui/SimulatorNotice";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import {
  BacktestEquityChart,
  UnderwaterDrawdownChart,
} from "@/components/charts/BacktestCharts";
import { formatPercent, formatINR } from "@/lib/utils";
import {
  Zap,
  Sliders,
  ShieldCheck,
  TrendingUp,
  TrendingDown,
  Pause,
  MinusCircle,
  Clock,
  Layers,
  ChevronDown,
  HelpCircle,
  AlertCircle,
  Calendar,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Briefcase,
} from "lucide-react";
import { OrderModal } from "@/components/portfolio/OrderModal";

const STRATEGY_VARIANTS = [
  {
    key: "A",
    name: "Variant A: Baseline Long-Only",
    tag: "Baseline",
    summary: "Enters LONG whenever calibrated probability exceeds 50%; moves to cash on downward signals.",
    technical: "Unconstrained direction: BUY when P(Up) > 0.5, SELL when P(Up) <= 0.5. Evaluates raw directional edge before cost filters.",
  },
  {
    key: "B",
    name: "Variant B: Cost-Aware Hurdle",
    tag: "Cost-Filtered",
    summary: "Requires expected gross return to exceed the 15 bps transaction cost hurdle before buying.",
    technical: "Trade gate: BUY only if P(Up)*exp_gain > friction_hurdle. Prevents turnover churn where gross gains are consumed by fees.",
  },
  {
    key: "C",
    name: "Variant C: Technical Confirmation",
    tag: "Technical Filter",
    summary: "Model signals must be corroborated by RSI and Moving Average trend alignment.",
    technical: "BUY requires Close > SMA20 > SMA50 and RSI between 35 and 65 (non-overbought momentum).",
  },
  {
    key: "D",
    name: "Variant D: Multi-Factor Consensus",
    tag: "Consensus",
    summary: "Requires triple agreement: ML Model + Technical Indicators + FinBERT News Sentiment.",
    technical: "Enters LONG only when Model = BUY, Tech = Bullish, and News Sentiment >= -0.05 (no bearish news veto).",
  },
  {
    key: "E",
    name: "Variant E: Regime Filtered",
    tag: "Regime-Adaptive",
    summary: "Adapts to market environment: vetoes trades during high volatility and demands high conviction in sideways markets.",
    technical: "Vetoes trades if 20-day realized volatility > 80th percentile. In choppy regimes, requires conviction > 8% (P(Up) > 58%).",
  },
];

export default function SignalsPage() {
  const { selectedTicker } = useSelectedStock();

  // Execution Settings States
  const [variant, setVariant] = useState<string>("A");
  const [longOnly, setLongOnly] = useState<boolean>(true);
  const [frictionBps, setFrictionBps] = useState<number>(15.0);

  // Queries
  const { data: stock } = useStock(selectedTicker);
  const {
    data: signal,
    isLoading: isSignalLoading,
    isError: isSignalError,
    refetch: refetchSignal,
  } = useLatestSignal(selectedTicker, variant, frictionBps);

  const {
    data: bt,
    isLoading: isBtLoading,
    isError: isBtError,
    refetch: refetchBt,
  } = useBacktest(selectedTicker, variant, longOnly, frictionBps, 1000000.0);

  const [activeTab, setActiveTab] = useState<"equity" | "drawdown">("equity");
  const [showMethodology, setShowMethodology] = useState<boolean>(false);
  const [tradeSearch, setTradeSearch] = useState<string>("");
  const [isTradeModalOpen, setIsTradeModalOpen] = useState<boolean>(false);

  if (isSignalError || isBtError) {
    return (
      <ErrorState
        title={`Failed to load trading signals for ${selectedTicker}`}
        message="Could not generate signals or execute backtest simulation via FastAPI backend. The data service may be starting up. Please try again in a minute."
        onRetry={() => {
          refetchSignal();
          refetchBt();
        }}
      />
    );
  }

  const selectedVariantDef =
    STRATEGY_VARIANTS.find((v) => v.key === variant) || STRATEGY_VARIANTS[0];

  // Derive signal tone for UI tinting
  const signalTone =
    signal?.signal_label === "BUY"
      ? "buy"
      : signal?.signal_label === "SELL"
      ? "sell"
      : signal?.signal_label === "HOLD"
      ? "hold"
      : "neutral";

  // Filter trade logs
  const filteredTrades = useMemo(() => {
    if (!bt?.trades) return [];
    if (!tradeSearch.trim()) return bt.trades;
    const q = tradeSearch.toLowerCase();
    return bt.trades.filter(
      (t) =>
        t.entry_date.includes(q) ||
        (t.exit_date && t.exit_date.includes(q)) ||
        t.position.toLowerCase().includes(q)
    );
  }, [bt, tradeSearch]);

  return (
    <div className="space-y-6">
      <SimulatorNotice />

      {/* SECTION 1: HEADER & CURRENT SIGNAL BANNER */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-6 shadow-subtle flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="font-mono text-xs font-bold text-sky-700 bg-sky-50 px-2 py-0.5 rounded border border-sky-100">
              {selectedTicker} · NSE
            </span>
            <span className="text-xs text-slate-500 font-sans">
              {stock?.display_name || selectedTicker}
            </span>
            <FreshnessBadge isLive={false} date={signal?.date} />
          </div>

          <h1 className="text-2xl font-bold text-slate-900 tracking-tight font-sans">
            Actionable Trading Signals & Backtesting
          </h1>
          <p className="mt-1 text-xs text-slate-500 max-w-2xl font-sans">
            Evidence-based decision engine applying probability calibration, technical filters, and Indian statutory transaction friction (15 bps round-trip).
          </p>
        </div>

        {signal && (
          <div className="flex flex-col md:items-end gap-1.5">
            <span className="text-[10px] uppercase font-mono font-bold text-slate-400">
              Active Strategy Signal
            </span>
            <SignalBadge signal={signal.signal_label} size="lg" showDescription />

            {/* Quick Action Button for BUY or SELL */}
            {signal.signal_label === "BUY" || signal.signal_label === "SELL" ? (
              <button
                type="button"
                onClick={() => setIsTradeModalOpen(true)}
                className="mt-1 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-700 text-white font-mono text-xs font-bold transition-colors flex items-center gap-1.5 shadow-xs"
              >
                <Briefcase className="w-3.5 h-3.5" /> Simulate Trade with Signal →
              </button>
            ) : (
              <div className="mt-1 px-2.5 py-1 rounded bg-slate-100 text-slate-400 text-[10px] font-mono text-right border border-slate-200">
                Simulated trade disabled for {signal.signal_label}
              </div>
            )}
          </div>
        )}
      </div>

      {/* SECTION 4: EXECUTION SETTINGS & STRATEGY SELECTOR */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-sky-600" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
              Strategy Variant & Execution Parameters
            </h2>
            <InfoTooltip
              term="Strategy Execution Settings"
              explanation="Switch between the 5 academically evaluated strategy variants (A to E) and test the impact of statutory Indian transaction costs on net backtested performance."
            />
          </div>

          {/* Execution Mode Toggle */}
          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="text-slate-500">Execution:</span>
            <div className="inline-flex rounded-md border border-slate-200 p-0.5 bg-slate-50">
              <button
                type="button"
                onClick={() => setLongOnly(true)}
                className={`px-2.5 py-1 rounded text-xs transition-colors ${
                  longOnly ? "bg-white font-bold text-slate-900 shadow-xs" : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Long-Only (Cash on SELL)
              </button>
              <button
                type="button"
                onClick={() => setLongOnly(false)}
                className={`px-2.5 py-1 rounded text-xs transition-colors ${
                  !longOnly ? "bg-white font-bold text-slate-900 shadow-xs" : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Long-Short
              </button>
            </div>
          </div>
        </div>

        {/* 5 Variant Buttons */}
        <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-5 gap-2.5">
          {STRATEGY_VARIANTS.map((v) => {
            const isSelected = variant === v.key;
            return (
              <button
                key={v.key}
                type="button"
                onClick={() => setVariant(v.key)}
                className={`p-3 rounded-lg border text-left transition-all ${
                  isSelected
                    ? "bg-[#0B132B] text-white border-[#0B132B] shadow-md ring-2 ring-sky-500/20"
                    : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50 hover:border-slate-300"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span
                    className={`font-mono text-xs font-bold ${
                      isSelected ? "text-sky-400" : "text-slate-900"
                    }`}
                  >
                    Variant {v.key}
                  </span>
                  <span
                    className={`text-[9px] font-mono px-1.5 py-0.2 rounded uppercase ${
                      isSelected ? "bg-white/10 text-slate-300" : "bg-slate-100 text-slate-500"
                    }`}
                  >
                    {v.tag}
                  </span>
                </div>
                <p
                  className={`text-[11px] leading-snug font-sans ${
                    isSelected ? "text-slate-300" : "text-slate-500"
                  }`}
                >
                  {v.summary}
                </p>
              </button>
            );
          })}
        </div>

        {/* Friction Slider & Definition */}
        <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-4 text-xs font-sans bg-slate-50/70 p-3 rounded-md border border-slate-100">
          <div className="flex items-center gap-3">
            <span className="font-mono font-semibold text-slate-700">Transaction Friction:</span>
            <div className="flex items-center gap-1.5">
              {[10.0, 15.0, 20.0, 25.0, 30.0].map((bps) => (
                <button
                  key={bps}
                  type="button"
                  onClick={() => setFrictionBps(bps)}
                  className={`px-2 py-1 rounded font-mono text-[11px] border transition-colors ${
                    frictionBps === bps
                      ? "bg-sky-600 text-white border-sky-600 font-bold"
                      : "bg-white text-slate-600 border-slate-200 hover:bg-slate-100"
                  }`}
                >
                  {bps.toFixed(1)} bps
                </button>
              ))}
            </div>
            <InfoTooltip
              term="Indian Statutory Friction"
              explanation="Includes 0.1% Securities Transaction Tax (STT) on delivery, exchange turn-over fees, SEBI charges, stamp duty, brokerage (approx. 0.03%), and bid-ask slippage. 15.0 bps (0.15% round-trip) is the standard conservative institutional benchmark."
            />
          </div>

          <span className="text-[11px] text-slate-500 font-mono">
            One-way fee: {(frictionBps / 2).toFixed(1)} bps · Hurdle rate: {(frictionBps / 10000).toFixed(4)}
          </span>
        </div>
      </div>

      {/* SECTION 1: SIGNAL KPI METRIC CARDS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {isSignalLoading ? (
          <MetricCardSkeleton />
        ) : signal ? (
          <>
            <MetricCard
              title="Current Signal"
              value={signal.signal_label}
              signalTone={signalTone}
              badge={<SignalBadge signal={signal.signal_label} size="md" />}
              subtitle={`Variant ${variant} · Date: ${signal.date}`}
              infoTerm="Actionable Signal"
              infoExplanation="BUY (enter long), SELL (exit long), or HOLD (stay in cash because estimated return does not exceed friction hurdle)."
              icon={<Zap className="w-4 h-4 text-amber-500" />}
            />

            <MetricCard
              title="Calibrated Confidence"
              value={formatPercent(signal.calibrated_proba_up)}
              subtitle={`Conviction: ${formatPercent(signal.confidence)}`}
              infoTerm="Calibrated Upward Probability"
              infoExplanation="Isotonic calibrated probability of upward price movement over the 3-day holding period."
              technicalNote="P(Up) calibrated using out-of-sample isotonic regression to map raw model scores to true empirical frequencies."
            />

            <MetricCard
              title="Net Expectancy"
              value={`${signal.net_expectancy >= 0 ? "+" : ""}${formatPercent(signal.net_expectancy)}`}
              change={Math.round(signal.net_expectancy * 10000) / 10}
              changeSuffix=" bps net"
              subtitle={`Gross: ${formatPercent(signal.expected_gross_return)} - ${formatPercent(signal.friction_rate)}`}
              infoTerm="Net Expected Return"
              infoExplanation="Gross expected return minus round-trip transaction costs. A trade is only executed when Net Expectancy is positive."
            />

            <MetricCard
              title="Market Regime"
              value={signal.market_regime.replace(/_/g, " ")}
              badge={<RegimeBadge regime={signal.market_regime} />}
              subtitle={`RSI: ${signal.rsi.toFixed(1)} · MACD: ${signal.macd_hist.toFixed(3)}`}
              infoTerm="Market Regime Classification"
              infoExplanation="Classifies the market into Trending Bull, Trending Bear, High Volatility, or Sideways Choppy."
            />
          </>
        ) : null}
      </div>

      {/* Decision Rationale Banner */}
      {signal && (
        <div className="bg-white border border-slate-200/90 rounded-lg p-4 shadow-subtle flex items-start gap-3 text-xs font-sans">
          <div className="p-2 rounded bg-sky-50 text-sky-700 mt-0.5 shrink-0">
            <Zap className="w-4 h-4" />
          </div>
          <div>
            <span className="font-bold text-slate-800 font-mono uppercase tracking-wider block mb-0.5">
              Decision Engine Execution Rationale ({selectedVariantDef.name})
            </span>
            <p className="text-slate-700 leading-relaxed">{signal.rationale}</p>
          </div>
        </div>
      )}

      {/* SECTION 3: MULTI-FACTOR CONFIRMATION MATRIX */}
      {signal && (
        <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle space-y-3">
          <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
            <div className="flex items-center gap-1.5">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                Multi-Factor Confirmation Matrix
              </h3>
              <InfoTooltip
                term="Multi-Factor Confirmation"
                explanation="Inspects whether technical trends, machine learning conviction, and FinBERT news sentiment all agree with the signal direction."
              />
            </div>
            <span className="text-[11px] font-mono text-slate-400">Factor Agreement Status</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs font-sans">
            {/* Factor 1: Model Probability */}
            <div className="p-3 rounded-lg border border-slate-200 bg-slate-50 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono font-bold text-slate-700">1. ML Model</span>
                  {signal.flag_model ? (
                    <span className="inline-flex items-center gap-1 text-[10px] text-emerald-700 font-mono font-bold">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> Confirmed
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-[10px] text-amber-700 font-mono font-bold">
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-600" /> Neutral Gate
                    </span>
                  )}
                </div>
                <p className="text-slate-500 text-[11px] leading-tight">
                  Calibrated P(Up) is {formatPercent(signal.calibrated_proba_up)} (conviction: {formatPercent(signal.confidence)}).
                </p>
              </div>
            </div>

            {/* Factor 2: Technical Confirmation */}
            <div className="p-3 rounded-lg border border-slate-200 bg-slate-50 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono font-bold text-slate-700">2. Technicals</span>
                  {signal.flag_tech ? (
                    <span className="inline-flex items-center gap-1 text-[10px] text-emerald-700 font-mono font-bold">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> Aligned
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-[10px] text-slate-500 font-mono">
                      <MinusCircle className="w-3.5 h-3.5" /> Disagree
                    </span>
                  )}
                </div>
                <p className="text-slate-500 text-[11px] leading-tight">
                  RSI: {signal.rsi.toFixed(1)} · SMA20: {formatINR(signal.sma20)} · SMA50: {formatINR(signal.sma50)}
                </p>
              </div>
            </div>

            {/* Factor 3: News Sentiment */}
            <div className="p-3 rounded-lg border border-slate-200 bg-slate-50 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono font-bold text-slate-700">3. FinBERT News</span>
                  {signal.flag_sentiment ? (
                    <span className="inline-flex items-center gap-1 text-[10px] text-emerald-700 font-mono font-bold">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> Agreed
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-[10px] text-slate-500 font-mono">
                      <MinusCircle className="w-3.5 h-3.5" /> Neutral
                    </span>
                  )}
                </div>
                <p className="text-slate-500 text-[11px] leading-tight">
                  Rolling 5-day sentiment: {signal.sentiment_score >= 0 ? `+${signal.sentiment_score.toFixed(3)}` : signal.sentiment_score.toFixed(3)}
                </p>
              </div>
            </div>

            {/* Factor 4: Market Regime */}
            <div className="p-3 rounded-lg border border-slate-200 bg-slate-50 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono font-bold text-slate-700">4. Market Regime</span>
                  <RegimeBadge regime={signal.market_regime} />
                </div>
                <p className="text-slate-500 text-[11px] leading-tight">
                  {signal.market_regime.includes("VOLATILITY")
                    ? "High volatility triggers safety veto."
                    : "Regime supports active trend trading."}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 5 & 6: BACKTEST PERFORMANCE & CHARTS */}
      {bt && (
        <div className="space-y-4">
          {/* Backtest KPI Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard
              title="Strategy CAGR"
              value={`+${formatPercent(bt.metrics.strategy.cagr)}`}
              change={Math.round((bt.metrics.strategy.cagr - bt.metrics.buy_and_hold.cagr) * 1000) / 10}
              changeSuffix="% vs B&H"
              subtitle={`Buy & Hold: +${formatPercent(bt.metrics.buy_and_hold.cagr)}`}
              infoTerm="Compound Annual Growth Rate (CAGR)"
              infoExplanation="Annualized portfolio return over the multi-year out-of-sample backtest after deducting all transaction friction."
            />

            <MetricCard
              title="Sharpe Ratio"
              value={bt.metrics.strategy.sharpe_ratio.toFixed(2)}
              change={Math.round((bt.metrics.strategy.sharpe_ratio - bt.metrics.buy_and_hold.sharpe_ratio) * 100) / 100}
              changeSuffix=" vs B&H"
              subtitle={`Buy & Hold: ${bt.metrics.buy_and_hold.sharpe_ratio.toFixed(2)} (Cash: 5.0%)`}
              infoTerm="Sharpe Ratio (Risk-Adjusted Return)"
              infoExplanation="Return generated per unit of volatility above the 5% Indian risk-free rate. Higher is better (> 1.0 is good; > 1.5 is excellent)."
            />

            <MetricCard
              title="Maximum Drawdown"
              value={`-${formatPercent(bt.metrics.strategy.max_drawdown)}`}
              subtitle={`Buy & Hold: -${formatPercent(bt.metrics.buy_and_hold.max_drawdown)}`}
              infoTerm="Maximum Portfolio Drawdown"
              infoExplanation="The largest peak-to-trough decline experienced by the strategy. A lower drawdown indicates superior capital protection."
            />

            <MetricCard
              title="Win Rate & Trades"
              value={`${formatPercent(bt.metrics.strategy.win_rate)}`}
              subtitle={`${bt.metrics.strategy.total_trades} trades · PF: ${bt.metrics.strategy.profit_factor.toFixed(2)}`}
              infoTerm="Win Rate & Profit Factor"
              infoExplanation="Percentage of closed trades with positive net returns after friction. Profit Factor is the ratio of gross gains to gross losses."
            />
          </div>

          {/* Interactive Backtest Charts Card */}
          <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                  Walk-Forward Backtest Simulation ({bt.metrics.strategy.start_date} to {bt.metrics.strategy.end_date})
                </h3>
                <p className="text-xs text-slate-500 font-sans mt-0.5">
                  Initial capital: ₹10,00,000 · Round-trip friction: {frictionBps.toFixed(1)} bps · Duration: {bt.metrics.strategy.years} years
                </p>
              </div>

              {/* Chart Tab Switcher */}
              <div className="flex items-center rounded-md border border-slate-200 bg-slate-50 p-0.5 text-xs font-mono">
                <button
                  type="button"
                  onClick={() => setActiveTab("equity")}
                  className={`px-3 py-1 rounded transition-colors ${
                    activeTab === "equity"
                      ? "bg-white font-bold text-slate-900 shadow-xs"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  Equity Curve
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab("drawdown")}
                  className={`px-3 py-1 rounded transition-colors ${
                    activeTab === "drawdown"
                      ? "bg-white font-bold text-slate-900 shadow-xs"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  Underwater Drawdown
                </button>
              </div>
            </div>

            {activeTab === "equity" ? (
              <BacktestEquityChart equityCurve={bt.equity_curve || []} />
            ) : (
              <UnderwaterDrawdownChart equityCurve={bt.equity_curve || []} />
            )}
          </div>
        </div>
      )}

      {/* SECTION 7: HISTORICAL TRADES LOG */}
      {bt && bt.trades && (
        <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-2.5">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                Historical Out-of-Sample Trade Execution Log
              </h3>
              <p className="text-xs text-slate-500 font-sans">
                Real trades recorded across walk-forward holding windows. Zero synthetic trades.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="text"
                placeholder="Search date or position..."
                value={tradeSearch}
                onChange={(e) => setTradeSearch(e.target.value)}
                className="px-2.5 py-1 rounded border border-slate-200 text-xs bg-slate-50 font-sans outline-hidden"
              />
              <span className="text-xs font-mono text-slate-400">
                {filteredTrades.length} trades
              </span>
            </div>
          </div>

          <div className="overflow-x-auto max-h-80">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-mono sticky top-0">
                <tr>
                  <th className="py-2 px-3">Entry Date</th>
                  <th className="py-2 px-3">Exit Date</th>
                  <th className="py-2 px-3">Direction</th>
                  <th className="py-2 px-3">Entry Price</th>
                  <th className="py-2 px-3">Exit Price</th>
                  <th className="py-2 px-3">Net Return</th>
                  <th className="py-2 px-3">Holding Period</th>
                  <th className="py-2 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {filteredTrades.length > 0 ? (
                  filteredTrades.map((tr, idx) => {
                    const isWin = (tr.net_return || 0) > 0;
                    return (
                      <tr key={idx} className="hover:bg-slate-50/70">
                        <td className="py-1.5 px-3 font-semibold text-slate-900">{tr.entry_date}</td>
                        <td className="py-1.5 px-3 text-slate-600">{tr.exit_date || "Open"}</td>
                        <td className="py-1.5 px-3">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                              tr.position === "LONG"
                                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                                : "bg-rose-50 text-rose-700 border border-rose-200"
                            }`}
                          >
                            {tr.position}
                          </span>
                        </td>
                        <td className="py-1.5 px-3 text-slate-800">{formatINR(tr.entry_price)}</td>
                        <td className="py-1.5 px-3 text-slate-800">
                          {tr.exit_price ? formatINR(tr.exit_price) : "-"}
                        </td>
                        <td className={`py-1.5 px-3 font-bold ${isWin ? "text-emerald-600" : "text-rose-600"}`}>
                          {tr.net_return !== null && tr.net_return !== undefined
                            ? `${tr.net_return >= 0 ? "+" : ""}${formatPercent(tr.net_return)}`
                            : "-"}
                        </td>
                        <td className="py-1.5 px-3 text-slate-600">
                          {tr.holding_steps ? `${tr.holding_steps * 3} days (${tr.holding_steps} step)` : "3 days"}
                        </td>
                        <td className="py-1.5 px-3 text-slate-400 text-[10px]">{tr.status}</td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={8} className="py-6 text-center text-slate-400 font-sans">
                      No trade records match the filter query.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* SECTION 8: RISK, METHODOLOGY & DISCLAIMER ACCORDION */}
      <div className="bg-white border border-slate-200/90 rounded-lg shadow-subtle overflow-hidden">
        <button
          type="button"
          onClick={() => setShowMethodology(!showMethodology)}
          className="w-full px-5 py-3.5 flex items-center justify-between text-left hover:bg-slate-50 transition-colors"
        >
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
              Quantitative Methodology & Institutional Risk Disclaimer
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500 font-sans">
            <span>{showMethodology ? "Hide methodology" : "Read quantitative specifications"}</span>
            <ChevronDown className={`w-4 h-4 transition-transform ${showMethodology ? "rotate-180" : ""}`} />
          </div>
        </button>

        {showMethodology && (
          <div className="p-5 border-t border-slate-100 bg-slate-50/50 space-y-4 text-xs font-sans text-slate-700 leading-relaxed">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3 rounded bg-white border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-1">1. Non-Overlapping 3-Day Steps</h4>
                Position decisions are evaluated at trading step $i$ and held until step $i+1$. Because holding windows are strictly non-overlapping, returns do not suffer from artificial serial correlation.
              </div>

              <div className="p-3 rounded bg-white border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-1">2. Statutory Friction Accounting</h4>
                Turnover-based fee deduction: fees are paid only on rebalancing. Holding cash earns the standard 5% annual risk-free cash rate without incurring churn costs.
              </div>

              <div className="p-3 rounded bg-white border border-slate-200">
                <h4 className="font-bold text-slate-900 mb-1">3. Research vs. Live Trading</h4>
                Historical walk-forward backtests simulate past out-of-sample data. They do not constitute investment advice or a guarantee of future market performance.
              </div>
            </div>

            <div className="p-3 rounded bg-amber-50 border border-amber-200 text-amber-900 text-xs">
              <strong className="block mb-0.5">Accessibility Note on Decision Making:</strong>
              Unlike naive black-box models that force a BUY or SELL every day, this quantitative engine abstains whenever market conditions are unfavorable. A high proportion of &ldquo;HOLD (Cash)&rdquo; days is a feature of risk management, preserving capital during volatile market regimes.
            </div>
          </div>
        )}
      </div>

      {/* SIMULATED ORDER MODAL */}
      <OrderModal
        isOpen={isTradeModalOpen}
        onClose={() => setIsTradeModalOpen(false)}
        defaultTicker={selectedTicker}
        defaultSide={signal?.signal_label === "SELL" ? "SELL" : "BUY"}
        defaultQuantity={10}
        signalSource={`ACTIONABLE_SIGNAL_${variant}`}
      />
    </div>
  );
}
