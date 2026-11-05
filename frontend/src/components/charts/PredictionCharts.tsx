"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  ReferenceLine,
  Cell,
} from "recharts";
import { CoverageCurvePoint, FeatureImportanceItem } from "@/types/api";

interface CoverageCurveChartProps {
  data: CoverageCurvePoint[];
  operatingThreshold: number;
}

export function CoverageCurveChart({ data, operatingThreshold }: CoverageCurveChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-xs text-slate-400 font-mono">
        No operating curve data available
      </div>
    );
  }

  // Format data for Recharts
  const chartData = data.map((pt) => ({
    threshold: pt.threshold,
    coverage: Math.round(pt.coverage * 100),
    accuracy: Math.round(pt.accuracy * 100),
    edge: Math.round(pt.edge * 1000) / 10,
    n: pt.n_committed,
  }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
          <XAxis
            dataKey="coverage"
            tick={{ fontSize: 11, fill: "#64748B" }}
            unit="%"
            label={{ value: "Coverage (% of Trading Days)", position: "insideBottom", offset: -10, fontSize: 11, fill: "#475569" }}
          />
          <YAxis
            domain={["dataMin - 2", "dataMax + 2"]}
            tick={{ fontSize: 11, fill: "#64748B" }}
            unit="%"
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const item = payload[0].payload;
                return (
                  <div className="bg-white p-2.5 rounded shadow-lg border border-slate-200 text-xs font-mono">
                    <p className="font-bold text-slate-900 mb-1">Threshold: {item.threshold}</p>
                    <p className="text-sky-600">Coverage: {item.coverage}% ({item.n} days)</p>
                    <p className="text-emerald-600">Accuracy: {item.accuracy}%</p>
                    <p className="text-slate-500">Edge: {item.edge > 0 ? `+${item.edge}` : item.edge}%</p>
                  </div>
                );
              }
              return null;
            }}
          />
          <ReferenceLine y={50} stroke="#94A3B8" strokeDasharray="3 3" label={{ value: "50% Neutral", fill: "#94A3B8", fontSize: 10 }} />
          <Line
            type="monotone"
            dataKey="accuracy"
            name="Accuracy (%)"
            stroke="#0284C7"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "#0284C7" }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

interface FeatureImportanceChartProps {
  features: FeatureImportanceItem[];
}

export function FeatureImportanceChart({ features }: FeatureImportanceChartProps) {
  if (!features || features.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-xs text-slate-400 font-mono">
        No feature weights available
      </div>
    );
  }

  // Friendly name map for technical indicator columns
  const friendlyName = (name: string): string => {
    const map: Record<string, string> = {
      close_vs_sma20: "Price vs 20-Day SMA",
      close_vs_sma50: "Price vs 50-Day SMA",
      close_vs_ema20: "Price vs 20-Day EMA",
      sma20_vs_sma50: "20 vs 50 SMA Trend",
      macd_norm: "MACD Normalized",
      macd_hist_norm: "MACD Histogram Momentum",
      rsi_centered: "RSI Momentum (14D)",
      ret_lag_1: "Yesterday Return",
      ret_lag_2: "2-Day Prior Return",
      ret_lag_3: "3-Day Prior Return",
      realized_vol_20: "20-Day Volatility",
      realized_vol_10: "10-Day Volatility",
      momentum_5: "5-Day Price Velocity",
      momentum_10: "10-Day Price Velocity",
      momentum_20: "20-Day Price Velocity",
      volume_ratio_20: "20-Day Volume Ratio",
      bb_bandwidth: "Bollinger Squeeze",
      price_above_cloud: "Ichimoku Cloud Breakout",
    };
    return map[name] || name.replace(/_/g, " ");
  };

  const chartData = features.slice(0, 8).map((f) => ({
    name: friendlyName(f.feature),
    rawName: f.feature,
    weight: f.importance,
    absWeight: Math.abs(f.importance),
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 10, fill: "#64748B" }} />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 10, fill: "#475569" }}
            width={120}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const item = payload[0].payload;
                const isBull = item.weight > 0;
                return (
                  <div className="bg-white p-2.5 rounded shadow-lg border border-slate-200 text-xs">
                    <p className="font-bold text-slate-900">{item.name}</p>
                    <p className="font-mono text-[10px] text-slate-400 mb-1">{item.rawName}</p>
                    <p className={isBull ? "text-emerald-600 font-mono" : "text-rose-600 font-mono"}>
                      Weight: {item.weight > 0 ? `+${item.weight.toFixed(4)}` : item.weight.toFixed(4)}
                    </p>
                    <p className="text-[10px] text-slate-500 mt-1">
                      {isBull ? "Pushes prediction UPWARD" : "Pushes prediction DOWNWARD"}
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <ReferenceLine x={0} stroke="#94A3B8" />
          <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.weight >= 0 ? "#10B981" : "#F43F5E"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
