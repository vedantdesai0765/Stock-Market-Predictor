"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { BacktestEquityPoint } from "@/types/api";
import { formatINR, formatPercent } from "@/lib/utils";

interface BacktestEquityChartProps {
  equityCurve: BacktestEquityPoint[];
}

export function BacktestEquityChart({ equityCurve }: BacktestEquityChartProps) {
  if (!equityCurve || equityCurve.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center text-xs text-slate-400 font-mono">
        No equity curve data available
      </div>
    );
  }

  // Format equity curve in Lakhs / Thousands for clean display
  const chartData = equityCurve.map((pt) => ({
    date: pt.date,
    strategy: pt.strategy_equity,
    buyHold: pt.buy_hold_equity,
    cash: pt.cash_equity,
    position: pt.position,
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: "#64748B" }}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fontSize: 10, fill: "#64748B" }}
            tickFormatter={(val) => `₹${(val / 100000).toFixed(1)}L`}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const item = payload[0].payload;
                return (
                  <div className="bg-white p-3 rounded shadow-xl border border-slate-200 text-xs font-mono">
                    <p className="font-bold text-slate-900 border-b border-slate-100 pb-1 mb-1.5">{label}</p>
                    <p className="text-emerald-700 font-semibold">Strategy: {formatINR(item.strategy)}</p>
                    <p className="text-sky-600 font-semibold">Buy & Hold: {formatINR(item.buyHold)}</p>
                    <p className="text-slate-500">Cash: {formatINR(item.cash)}</p>
                    <div className="mt-1 pt-1 border-t border-slate-100 text-[10px] text-slate-600 font-sans">
                      Position: {item.position > 0 ? "LONG (Invested)" : "CASH (Risk-Off)"}
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
          <Line
            type="monotone"
            dataKey="strategy"
            name="Strategy Equity"
            stroke="#10B981"
            strokeWidth={2.5}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="buyHold"
            name="Buy & Hold Equity"
            stroke="#0284C7"
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="cash"
            name="Risk-Free Cash (5% p.a.)"
            stroke="#94A3B8"
            strokeWidth={1}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

interface UnderwaterDrawdownChartProps {
  equityCurve: BacktestEquityPoint[];
}

export function UnderwaterDrawdownChart({ equityCurve }: UnderwaterDrawdownChartProps) {
  if (!equityCurve || equityCurve.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-xs text-slate-400 font-mono">
        No drawdown data available
      </div>
    );
  }

  const chartData = equityCurve.map((pt) => ({
    date: pt.date,
    drawdown: Math.round(pt.drawdown * 1000) / 10,
  }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: "#64748B" }}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            domain={["dataMin - 2", 0]}
            tick={{ fontSize: 10, fill: "#64748B" }}
            unit="%"
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const item = payload[0].payload;
                return (
                  <div className="bg-white p-2.5 rounded shadow-lg border border-slate-200 text-xs font-mono">
                    <p className="font-bold text-slate-900">{label}</p>
                    <p className="text-rose-600 font-semibold">Drawdown: {item.drawdown}%</p>
                    <p className="text-[10px] text-slate-500 font-sans mt-0.5">Decline from all-time portfolio peak</p>
                  </div>
                );
              }
              return null;
            }}
          />
          <ReferenceLine y={0} stroke="#94A3B8" />
          <Area
            type="monotone"
            dataKey="drawdown"
            name="Strategy Underwater Drawdown (%)"
            stroke="#EF4444"
            fill="#FEE2E2"
            fillOpacity={0.7}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
