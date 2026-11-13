"use client";

import React, { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { usePortfolio, usePortfolioAnalytics, useStocks } from "@/hooks/useStockData";
import { api } from "@/lib/api";
import { MetricCard } from "@/components/ui/MetricCard";
import { MetricCardSkeleton } from "@/components/ui/LoadingSkeleton";
import { ErrorState } from "@/components/ui/ErrorState";
import { SimulatorNotice } from "@/components/ui/SimulatorNotice";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import { SignalBadge } from "@/components/ui/Badge";
import { OrderModal } from "@/components/portfolio/OrderModal";
import { ResetModal } from "@/components/portfolio/ResetModal";
import {
  PortfolioEquityCurveChart,
  AssetAllocationDonutChart,
} from "@/components/charts/PortfolioCharts";
import { formatINR, formatPercent } from "@/lib/utils";
import {
  Briefcase,
  TrendingUp,
  TrendingDown,
  RotateCcw,
  PlusCircle,
  Clock,
  ShieldAlert,
  ArrowRight,
  PieChart as PieIcon,
  Layers,
  History,
  CheckCircle2,
} from "lucide-react";

export default function PortfolioPage() {
  const queryClient = useQueryClient();
  const { data: portfolio, isLoading, isError, error, refetch } = usePortfolio();
  const { data: analytics, isLoading: isAnalyticsLoading } = usePortfolioAnalytics();
  const { data: stocksData } = useStocks();

  // Order Modal State
  const [isOrderModalOpen, setIsOrderModalOpen] = useState(false);
  const [modalTicker, setModalTicker] = useState("TCS");
  const [modalSide, setModalSide] = useState<"BUY" | "SELL">("BUY");
  const [modalQuantity, setModalQuantity] = useState(10);
  const [modalSource, setModalSource] = useState("MANUAL");

  const [isResetModalOpen, setIsResetModalOpen] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [resetMessage, setResetMessage] = useState<string | null>(null);

  const openBuyModal = (ticker: string = "TCS", qty: number = 10, source: string = "MANUAL") => {
    setModalTicker(ticker);
    setModalSide("BUY");
    setModalQuantity(qty);
    setModalSource(source);
    setIsOrderModalOpen(true);
  };

  const openSellModal = (ticker: string, ownedQty: number) => {
    setModalTicker(ticker);
    setModalSide("SELL");
    setModalQuantity(Math.min(ownedQty, 10));
    setModalSource("MANUAL");
    setIsOrderModalOpen(true);
  };

  const handleConfirmReset = async () => {
    setIsResetting(true);
    try {
      const res = await api.resetPortfolio();
      setResetMessage(res.message);
      queryClient.invalidateQueries({ queryKey: ["portfolio"] });
      queryClient.invalidateQueries({ queryKey: ["portfolio-analytics"] });
      setTimeout(() => setResetMessage(null), 4000);
    } catch (err: any) {
      alert(err.message || "Reset failed");
    } finally {
      setIsResetting(false);
    }
  };

  if (isError) {
    return (
      <div className="space-y-6">
        <ErrorState
          title="Virtual Portfolio Unavailable"
          message={error?.message || "Failed to load wallet data from the backend."}
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  const pnlTone =
    !portfolio || portfolio.total_pnl === 0
      ? "neutral"
      : portfolio.total_pnl > 0
      ? "buy"
      : "sell";

  return (
    <div className="space-y-6">
      <SimulatorNotice>
        This wallet uses virtual money only. Its BUY and SELL suggestions come from the 3-day model, which did not
        beat a simple guess for any of the 10 stocks in testing, so treat it as practice with orders, costs and
        record-keeping.
      </SimulatorNotice>

      {/* SECTION 1: HEADER & SIMULATED TRADING NOTICE */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-6 shadow-subtle flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="font-mono text-xs font-bold text-sky-700 bg-sky-50 px-2 py-0.5 rounded border border-sky-100 flex items-center gap-1">
              <Briefcase className="w-3.5 h-3.5" /> Paper Trading Wallet
            </span>
            <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-300">
              <ShieldAlert className="w-3.5 h-3.5 text-amber-600" />
              SIMULATED TRADING (VIRTUAL CAPITAL ₹1,00,000)
            </span>
          </div>

          <h1 className="text-2xl font-bold text-slate-900 tracking-tight font-sans">
            Virtual Portfolio & Trade Execution
          </h1>
          <p className="mt-1 text-xs text-slate-500 max-w-2xl font-sans">
            Simulated paper trading environment with realistic Indian statutory delivery charges (7.5 bps one-way / 15 bps round-trip friction).
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            disabled={isResetting}
            onClick={() => setIsResetModalOpen(true)}
            className="px-3 py-2 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 text-xs font-mono font-medium transition-colors flex items-center gap-1.5 shadow-xs"
            title="Safely reset wallet capital back to ₹1,00,000"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${isResetting ? "animate-spin" : ""}`} />
            Reset Wallet
          </button>
          <button
            type="button"
            onClick={() => openBuyModal()}
            className="px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-700 text-white text-xs font-bold font-sans transition-colors flex items-center gap-1.5 shadow-xs"
          >
            <PlusCircle className="w-4 h-4" /> Place Order
          </button>
        </div>
      </div>

      {resetMessage && (
        <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-800 font-sans flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          <span>{resetMessage}</span>
        </div>
      )}

      {/* SECTION 2: PORTFOLIO KPI SUMMARY CARDS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {isLoading ? (
          <MetricCardSkeleton />
        ) : portfolio ? (
          <>
            <MetricCard
              title="Portfolio Valuation"
              value={formatINR(portfolio.portfolio_value)}
              subtitle={`Initial: ${formatINR(portfolio.initial_capital)}`}
              infoTerm="Portfolio Valuation"
              infoExplanation="Total combined value of your available liquid cash and current market value of all held stocks."
            />
            <MetricCard
              title="Available Cash"
              value={formatINR(portfolio.available_cash)}
              subtitle={`Invested: ${formatINR(portfolio.invested_capital)}`}
              infoTerm="Available Cash"
              infoExplanation="Unallocated liquid capital ready for new stock purchases. Deducts statutory brokerage and STT upon trade execution."
            />
            <MetricCard
              title="Total P&L"
              value={formatINR(portfolio.total_pnl)}
              signalTone={pnlTone}
              subtitle={`Overall Return: ${portfolio.total_return_pct > 0 ? "+" : ""}${portfolio.total_return_pct.toFixed(2)}%`}
              infoTerm="Total P&L"
              infoExplanation="Net cumulative profit or loss since wallet inception (combines closed trade realised gains and open position unrealised gains)."
            />
            <MetricCard
              title="Unrealised P&L"
              value={formatINR(portfolio.unrealised_pnl)}
              signalTone={portfolio.unrealised_pnl >= 0 ? (portfolio.unrealised_pnl > 0 ? "buy" : "neutral") : "sell"}
              subtitle={`Realised P&L: ${formatINR(portfolio.realised_pnl)}`}
              infoTerm="Unrealised P&L"
              infoExplanation="Paper profit/loss on currently active stock holdings compared to purchase price. Becomes realised only upon selling."
            />
          </>
        ) : null}
      </div>

      {/* SECTION 3: ACTIONABLE MODEL SIGNALS QUICK-TRADE PANEL */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
          <div>
            <div className="flex items-center gap-1.5">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                Model-Generated Trading Opportunities
              </h2>
              <InfoTooltip
                term="Signal-Driven Paper Trading"
                explanation="Allows initiating simulated trades directly from research engine recommendations. Trades are NEVER executed automatically and always require your explicit confirmation."
                technicalNote="Only active BUY recommendations enable trade creation. Neutral signals (HOLD) and abstentions (NO SIGNAL) are disabled to protect capital."
              />
            </div>
            <p className="text-xs text-slate-500 font-sans mt-0.5">
              Current actionable directional signals across supported NSE equities.
            </p>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            Friction Hurdle: 15 bps · Research Models
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {[
            { ticker: "TCS", name: "Tata Consultancy Services", signal: "HOLD", price: 2296.2, reason: "Model abstains: Gross edge below 15 bps friction" },
            { ticker: "INFY", name: "Infosys", signal: "BUY", price: 1144.0, reason: "P(Up) = 57.1% · Net expectancy +1.61%" },
            { ticker: "SBI", name: "State Bank of India", signal: "BUY", price: 1048.0, reason: "P(Up) = 52.3% · Regime: Trending Bull" },
          ].map((item) => (
            <div
              key={item.ticker}
              className="p-3.5 rounded-lg border border-slate-200 bg-slate-50/70 hover:bg-slate-50 transition-colors flex flex-col justify-between gap-3"
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono font-bold text-xs text-slate-900">{item.ticker}</span>
                    <span className="text-[11px] text-slate-500 truncate max-w-[120px]">{item.name}</span>
                  </div>
                  <SignalBadge signal={item.signal} size="sm" />
                </div>
                <div className="text-xs font-mono font-bold text-slate-900 mb-1">
                  {formatINR(item.price)}
                </div>
                <p className="text-[11px] text-slate-500 font-sans leading-snug">
                  {item.reason}
                </p>
              </div>

              {item.signal === "BUY" ? (
                <button
                  type="button"
                  onClick={() => openBuyModal(item.ticker, 10, "ACTIONABLE_SIGNAL_ENGINE")}
                  className="w-full py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-700 text-white font-mono text-xs font-bold transition-colors flex items-center justify-center gap-1 shadow-xs"
                >
                  Simulate Trade with Signal →
                </button>
              ) : (
                <div className="py-1 px-2 rounded bg-slate-100 text-center text-[10px] font-mono text-slate-400">
                  Execution Inactive for {item.signal}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* SECTION 4: HOLDINGS DASHBOARD TABLE */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
              Current Stock Holdings ({portfolio?.holdings_count || 0})
            </h2>
            <p className="text-xs text-slate-500 font-sans">
              Active equity positions marked to market against latest genuine historical closing prices.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500">
            Total Position Value: {formatINR(portfolio?.invested_capital || 0)}
          </span>
        </div>

        {portfolio?.holdings && portfolio.holdings.length > 0 ? (
          <div className="overflow-x-auto border border-slate-200 rounded-lg">
            <table className="w-full text-left border-collapse text-xs font-sans">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 font-mono text-[11px] uppercase tracking-wider text-slate-500">
                  <th className="py-2.5 px-3">Stock Symbol</th>
                  <th className="py-2.5 px-3 text-right">Shares Held</th>
                  <th className="py-2.5 px-3 text-right">Avg Buy Price</th>
                  <th className="py-2.5 px-3 text-right">Current Price</th>
                  <th className="py-2.5 px-3 text-right">Invested (₹)</th>
                  <th className="py-2.5 px-3 text-right">Market Value (₹)</th>
                  <th className="py-2.5 px-3 text-right">Unrealised P&L</th>
                  <th className="py-2.5 px-3 text-center">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-xs text-slate-800">
                {portfolio.holdings.map((h) => {
                  const isPositive = h.unrealised_pnl >= 0;
                  return (
                    <tr key={h.ticker} className="hover:bg-slate-50/80 transition-colors">
                      <td className="py-3 px-3">
                        <div className="font-bold text-slate-900">{h.ticker}</div>
                        <div className="text-[10px] text-slate-400 font-sans">{h.display_name}</div>
                      </td>
                      <td className="py-3 px-3 text-right font-bold">{h.quantity}</td>
                      <td className="py-3 px-3 text-right text-slate-600">{formatINR(h.avg_buy_price)}</td>
                      <td className="py-3 px-3 text-right font-bold text-slate-900">
                        {formatINR(h.current_price)}
                        <span className="block text-[10px] text-slate-400 font-normal">as-of {h.price_date}</span>
                      </td>
                      <td className="py-3 px-3 text-right text-slate-600">{formatINR(h.invested_amount)}</td>
                      <td className="py-3 px-3 text-right font-bold text-slate-900">{formatINR(h.current_value)}</td>
                      <td className="py-3 px-3 text-right">
                        <span
                          className={`inline-block font-bold ${
                            isPositive ? "text-emerald-700" : "text-rose-700"
                          }`}
                        >
                          {isPositive ? "+" : ""}
                          {formatINR(h.unrealised_pnl)}
                        </span>
                        <span
                          className={`block text-[10px] ${
                            isPositive ? "text-emerald-600" : "text-rose-600"
                          }`}
                        >
                          ({isPositive ? "+" : ""}
                          {h.return_pct.toFixed(2)}%)
                        </span>
                      </td>
                      <td className="py-3 px-3 text-center">
                        <div className="flex items-center justify-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => openBuyModal(h.ticker, 5)}
                            className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 text-[11px] font-bold"
                          >
                            Buy More
                          </button>
                          <button
                            type="button"
                            onClick={() => openSellModal(h.ticker, h.quantity)}
                            className="px-2 py-1 rounded bg-rose-50 hover:bg-rose-100 text-rose-700 text-[11px] font-bold border border-rose-200"
                          >
                            Sell
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center bg-slate-50 rounded-lg border border-dashed border-slate-200">
            <Briefcase className="w-8 h-8 text-slate-400 mx-auto mb-2" />
            <h3 className="text-xs font-bold text-slate-700 uppercase font-mono">No Active Holdings</h3>
            <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1 font-sans">
              Your wallet has ₹1,00,000 in available cash. Use "Place Order" or simulate a trade from the model recommendations above.
            </p>
            <button
              type="button"
              onClick={() => openBuyModal("TCS", 10)}
              className="mt-3 px-4 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-700 text-white font-bold text-xs"
            >
              Start First Simulated Trade
            </button>
          </div>
        )}
      </div>

      {/* SECTION 5: INTERACTIVE PORTFOLIO ANALYTICS CHARTS */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                Portfolio Equity Progression (Cumulative Value)
              </h2>
              <p className="text-xs text-slate-500 font-sans">
                Growth trajectory of simulated trading account starting at ₹1,00,000 capital.
              </p>
            </div>
            <span className="text-[11px] font-mono text-slate-400">Total Valuation (₹)</span>
          </div>
          <PortfolioEquityCurveChart data={analytics?.equity_curve || []} />
        </div>

        <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                Asset Allocation
              </h2>
              <p className="text-xs text-slate-500 font-sans">
                Liquid cash vs. equity distribution.
              </p>
            </div>
            <PieIcon className="w-4 h-4 text-slate-400" />
          </div>
          <AssetAllocationDonutChart data={analytics?.asset_allocation || []} />
        </div>
      </div>

      {/* SECTION 6: AUDITABLE TRANSACTION HISTORY TABLE */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
              Auditable Transaction Ledger
            </h2>
            <p className="text-xs text-slate-500 font-sans">
              Complete chronological audit trail with statutory Indian delivery fees and execution timestamps.
            </p>
          </div>
          <History className="w-4 h-4 text-slate-400" />
        </div>

        {portfolio?.recent_transactions && portfolio.recent_transactions.length > 0 ? (
          <div className="overflow-x-auto border border-slate-200 rounded-lg max-h-80">
            <table className="w-full text-left border-collapse text-xs font-sans">
              <thead className="sticky top-0 bg-slate-50 border-b border-slate-200 font-mono text-[11px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Stock</th>
                  <th className="py-2.5 px-3 text-center">Type</th>
                  <th className="py-2.5 px-3 text-right">Shares</th>
                  <th className="py-2.5 px-3 text-right">Execution Price</th>
                  <th className="py-2.5 px-3 text-right">Friction Paid</th>
                  <th className="py-2.5 px-3 text-right">Net Cash Impact</th>
                  <th className="py-2.5 px-3 text-left">Signal Source</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-xs text-slate-800">
                {portfolio.recent_transactions.map((tx) => (
                  <tr key={tx.id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-2.5 px-3 text-slate-500">
                      {tx.executed_at ? tx.executed_at.substring(0, 19).replace("T", " ") : "-"}
                    </td>
                    <td className="py-2.5 px-3 font-bold text-slate-900">{tx.ticker}</td>
                    <td className="py-2.5 px-3 text-center">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                          tx.side === "BUY"
                            ? "bg-emerald-50 text-emerald-800 border border-emerald-300"
                            : "bg-rose-50 text-rose-800 border border-rose-300"
                        }`}
                      >
                        {tx.side}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold">{tx.quantity}</td>
                    <td className="py-2.5 px-3 text-right">{formatINR(tx.execution_price)}</td>
                    <td className="py-2.5 px-3 text-right text-amber-700">+{formatINR(tx.friction_amount)}</td>
                    <td className="py-2.5 px-3 text-right font-bold">
                      <span className={tx.side === "BUY" ? "text-rose-700" : "text-emerald-700"}>
                        {tx.side === "BUY" ? "-" : "+"}
                        {formatINR(tx.net_amount)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px]">{tx.signal_source || "MANUAL"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-6 text-center text-xs font-mono text-slate-400 bg-slate-50 rounded-lg">
            No transactions executed yet.
          </div>
        )}
      </div>

      {/* ORDER PLACEMENT MODAL */}
      <OrderModal
        isOpen={isOrderModalOpen}
        onClose={() => setIsOrderModalOpen(false)}
        defaultTicker={modalTicker}
        defaultSide={modalSide}
        defaultQuantity={modalQuantity}
        signalSource={modalSource}
        availableCash={portfolio?.available_cash || 100000.0}
      />

      {/* EXPLICIT WALLET RESET MODAL */}
      <ResetModal
        isOpen={isResetModalOpen}
        onClose={() => setIsResetModalOpen(false)}
        onConfirmReset={handleConfirmReset}
        isResetting={isResetting}
      />
    </div>
  );
}
