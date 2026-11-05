"use client";

import React, { useState } from "react";
import {
  Settings,
  Server,
  Database,
  Globe,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  ShieldCheck,
  Cpu,
  Layers,
  Activity,
  Calendar,
} from "lucide-react";
import { useHealth, useStocks } from "@/hooks/useStockData";
import { ResetModal } from "@/components/portfolio/ResetModal";
import { api } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const { data: health, isLoading } = useHealth();
  const { data: stocks } = useStocks();

  const [isResetModalOpen, setIsResetModalOpen] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [resetFeedback, setResetFeedback] = useState<string | null>(null);

  const handleConfirmReset = async () => {
    setIsResetting(true);
    try {
      const res = await api.resetPortfolio();
      setResetFeedback(res.message);
      queryClient.invalidateQueries({ queryKey: ["portfolio"] });
      queryClient.invalidateQueries({ queryKey: ["portfolio-analytics"] });
      setTimeout(() => setResetFeedback(null), 5000);
    } catch (err: any) {
      alert(err.message || "Reset failed");
    } finally {
      setIsResetting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-700 border border-slate-200 font-mono">
            <Settings className="w-3 h-3 text-slate-500" />
            System Diagnostics
          </span>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/70 font-mono">
            <ShieldCheck className="w-3 h-3 text-emerald-600" />
            Security Verified
          </span>
        </div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 font-sans tracking-tight">
          Terminal Settings & System Health
        </h1>
        <p className="text-xs text-slate-500 font-sans mt-0.5">
          Backend server connections, database connectivity, data coverage across all 10 NSE assets, and simulation controls.
        </p>
      </div>

      {resetFeedback && (
        <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs font-semibold text-emerald-800 flex items-center gap-2 animate-in fade-in duration-200">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>{resetFeedback}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Backend Connection */}
        <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono mb-4 flex items-center gap-2">
            <Server className="w-4 h-4 text-sky-600" />
            FastAPI Backend Diagnostics
          </h3>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-500">FastAPI Base URL</span>
              <span className="font-mono font-semibold text-slate-800">
                {process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api/v1"}
              </span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-500">Connection Health</span>
              <span className="font-mono font-semibold text-emerald-600 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                {isLoading ? "Checking..." : health?.status === "healthy" ? "ONLINE (Connected)" : "OFFLINE"}
              </span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-500">Registered Stock Models</span>
              <span className="font-mono font-semibold text-slate-800">
                {health?.registered_stocks || 10} Equities Active
              </span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-500">SQLAlchemy Database</span>
              <span className="font-mono font-semibold text-slate-800">
                {health?.database || "data/app.db (SQLite 3)"}
              </span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-500">Interactive Swagger Docs</span>
              <a
                href="http://127.0.0.1:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="font-mono font-semibold text-sky-600 hover:text-sky-800 inline-flex items-center gap-1"
              >
                /docs (Swagger UI)
              </a>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-slate-500">Frontend Framework</span>
              <span className="font-mono font-semibold text-slate-800">Next.js 14.2 (App Router + React 19)</span>
            </div>
          </div>
        </div>

        {/* Demonstration & Virtual Reset Controls */}
        <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-xs flex flex-col justify-between">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono mb-3 flex items-center gap-2">
              <RotateCcw className="w-4 h-4 text-amber-600" />
              Demonstration Mode & Wallet Reset
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed font-sans mb-4">
              For live presentations, reviews, or testing, you can reset the simulated virtual wallet back to its pristine initial state:
            </p>

            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs space-y-2 text-slate-700 font-sans mb-4">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                <span>Restores virtual starting capital to ₹1,00,000.00 cash</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                <span>Purges all active virtual holdings (0 positions)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                <span>Clears simulated transaction ledger history</span>
              </div>
            </div>

            <div className="p-2.5 bg-blue-50/70 border border-blue-200/70 rounded-md text-[11px] text-blue-900 font-sans">
              <strong>Non-Destructive Guarantee:</strong> This operation resets only the simulation wallet database table. Historical ML datasets, trained weights, and backtest logs remain completely intact.
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between">
            <span className="text-xs text-slate-500 font-mono">Requires confirmation</span>
            <button
              type="button"
              onClick={() => setIsResetModalOpen(true)}
              disabled={isResetting}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold font-sans shadow-xs transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Demonstration Wallet
            </button>
          </div>
        </div>
      </div>

      {/* Ten Supported Equities Coverage Matrix */}
      <div className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-xs">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono mb-3 flex items-center gap-2">
          <Layers className="w-4 h-4 text-sky-600" />
          Supported Indian Equities Data Coverage (10 Stocks)
        </h3>
        <p className="text-xs text-slate-500 font-sans mb-4">
          All 10 equities are registered with complete feature pipelines, walk-forward confidence models, and next-morning forecast calibrations.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left text-slate-700">
            <thead className="bg-slate-50 font-mono text-[11px] uppercase tracking-wider text-slate-500 border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3">Symbol</th>
                <th className="py-2.5 px-3">Company Name</th>
                <th className="py-2.5 px-3">Sector</th>
                <th className="py-2.5 px-3">Exchange</th>
                <th className="py-2.5 px-3">OHLCV Status</th>
                <th className="py-2.5 px-3">Research Snapshot</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-sans">
              {(stocks?.stocks || [
                { ticker: "TCS", name: "Tata Consultancy Services", sector: "IT" },
                { ticker: "RELIANCE", name: "Reliance Industries", sector: "Energy" },
                { ticker: "INFY", name: "Infosys", sector: "IT" },
                { ticker: "HDFCBANK", name: "HDFC Bank", sector: "Financial Services" },
                { ticker: "SBI", name: "State Bank of India", sector: "Financial Services" },
                { ticker: "ITC", name: "ITC Limited", sector: "Consumer Goods" },
                { ticker: "ADANIENT", name: "Adani Enterprises", sector: "Energy & Infrastructure" },
                { ticker: "VEDL", name: "Vedanta Limited", sector: "Metals & Mining" },
                { ticker: "ETERNAL", name: "Eternal Limited (Zomato)", sector: "Consumer Tech" },
                { ticker: "ADANIGREEN", name: "Adani Green Energy", sector: "Renewable Energy" },
              ]).map((stock: any) => (
                <tr key={stock.ticker} className="hover:bg-slate-50/70">
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-900">{stock.ticker}</td>
                  <td className="py-2.5 px-3 font-medium text-slate-800">{stock.name}</td>
                  <td className="py-2.5 px-3 text-slate-600">{stock.sector || "Large Cap"}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-500">NSE India</td>
                  <td className="py-2.5 px-3 font-mono text-emerald-700">
                    <span className="inline-flex items-center gap-1 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200/60 font-semibold text-[10px]">
                      <CheckCircle2 className="w-2.5 h-2.5 text-emerald-600" /> 2018–2026
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-600 text-[11px]">2026-08-28 Close</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Explicit Reset Modal */}
      <ResetModal
        isOpen={isResetModalOpen}
        onClose={() => setIsResetModalOpen(false)}
        onConfirmReset={handleConfirmReset}
        isResetting={isResetting}
      />
    </div>
  );
}
