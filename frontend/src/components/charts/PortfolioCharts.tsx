"use client";

import React from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import { AssetAllocationItem, PortfolioEquityPoint } from "@/types/api";
import { formatINR } from "@/lib/utils";

interface PortfolioEquityChartProps {
  data: PortfolioEquityPoint[];
}

export function PortfolioEquityCurveChart({ data }: PortfolioEquityChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-xs font-mono text-slate-400 bg-slate-50/50 rounded-lg border border-slate-100">
        No portfolio valuation snapshots recorded yet.
      </div>
    );
  }

  const chartData = data.map((d) => ({
    timestamp: d.timestamp.length > 10 ? d.timestamp.substring(5, 16) : d.timestamp,
    portfolio_value: d.portfolio_value,
    cash: d.cash,
    invested: d.invested,
  }));

  const minVal = Math.min(...data.map((d) => d.portfolio_value)) * 0.98;
  const maxVal = Math.max(...data.map((d) => d.portfolio_value)) * 1.02;

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
          <defs>
            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#2563EB" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#2563EB" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
          <XAxis
            dataKey="timestamp"
            tick={{ fontSize: 11, fill: "#64748B", fontFamily: "var(--font-mono)" }}
            tickLine={false}
            axisLine={{ stroke: "#CBD5E1" }}
          />
          <YAxis
            domain={[Math.floor(minVal), Math.ceil(maxVal)]}
            tick={{ fontSize: 11, fill: "#64748B", fontFamily: "var(--font-mono)" }}
            tickLine={false}
            axisLine={{ stroke: "#CBD5E1" }}
            tickFormatter={(val) => `₹${(val / 1000).toFixed(0)}k`}
            width={65}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || !payload.length) return null;
              const p = payload[0].payload;
              return (
                <div className="bg-slate-900 text-white p-3 rounded shadow-lg border border-slate-700 text-xs font-mono space-y-1">
                  <div className="text-slate-400 border-b border-slate-700 pb-1 mb-1">
                    {p.timestamp}
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-sky-400">Total Valuation:</span>
                    <span className="font-bold">{formatINR(p.portfolio_value)}</span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-slate-300">Available Cash:</span>
                    <span>{formatINR(p.cash)}</span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-slate-300">Invested Capital:</span>
                    <span>{formatINR(p.invested)}</span>
                  </div>
                </div>
              );
            }}
          />
          <Area
            type="monotone"
            dataKey="portfolio_value"
            stroke="#2563EB"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#equityGrad)"
            name="Portfolio Value"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

interface AssetAllocationChartProps {
  data: AssetAllocationItem[];
}

export function AssetAllocationDonutChart({ data }: AssetAllocationChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-xs font-mono text-slate-400 bg-slate-50/50 rounded-lg border border-slate-100">
        No capital allocated yet.
      </div>
    );
  }

  return (
    <div className="w-full h-72 flex flex-col items-center justify-center">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius={55}
            outerRadius={85}
            paddingAngle={2}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} stroke="#FFFFFF" strokeWidth={1.5} />
            ))}
          </Pie>
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || !payload.length) return null;
              const p = payload[0].payload as AssetAllocationItem;
              return (
                <div className="bg-slate-900 text-white p-2.5 rounded shadow-lg border border-slate-700 text-xs font-mono space-y-1">
                  <div className="font-bold flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: p.color }} />
                    {p.name} ({p.ticker})
                  </div>
                  <div className="text-sky-300">{formatINR(p.value)}</div>
                  <div className="text-slate-400">{p.percentage.toFixed(1)}% of portfolio</div>
                </div>
              );
            }}
          />
          <Legend
            verticalAlign="bottom"
            height={36}
            formatter={(value, entry: any) => (
              <span className="text-xs font-mono text-slate-700">
                {value} ({entry.payload.percentage.toFixed(1)}%)
              </span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
