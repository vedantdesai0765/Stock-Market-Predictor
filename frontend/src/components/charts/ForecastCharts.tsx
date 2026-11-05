"use client";

import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { ForecastSummaryItem } from "@/types/api";

interface ForecastComparisonChartProps {
  summaries: ForecastSummaryItem[];
}

export function ForecastComparisonChart({ summaries }: ForecastComparisonChartProps) {
  if (!summaries || summaries.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-xs text-slate-400 font-mono">
        No forecast summary data available
      </div>
    );
  }

  const chartData = summaries.map((s) => ({
    model: s.model,
    guess: Math.round(s.acc_guess * 1000) / 10,
    modelAcc: Math.round(s.acc_model * 1000) / 10,
    newsAcc: Math.round(s.acc_news * 1000) / 10,
    edge: Math.round(s.edge * 1000) / 10,
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
          <XAxis dataKey="model" tick={{ fontSize: 11, fill: "#475569" }} />
          <YAxis
            domain={[40, 85]}
            tick={{ fontSize: 11, fill: "#64748B" }}
            unit="%"
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const item = payload[0].payload;
                return (
                  <div className="bg-white p-3 rounded shadow-lg border border-slate-200 text-xs font-mono">
                    <p className="font-bold text-slate-900 mb-1">{item.model}</p>
                    <p className="text-slate-500">Baseline Guess: {item.guess}%</p>
                    <p className="text-sky-600 font-semibold">Model Technicals: {item.modelAcc}%</p>
                    <p className="text-teal-600 font-semibold">With Overnight News: {item.newsAcc}%</p>
                    <p className="text-emerald-600 font-bold mt-1">
                      Edge over Guess: +{item.edge}%
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
          <ReferenceLine y={50} stroke="#CBD5E1" strokeDasharray="2 2" />
          <Bar dataKey="guess" name="Prior Guess Baseline (%)" fill="#94A3B8" radius={[3, 3, 0, 0]} />
          <Bar dataKey="modelAcc" name="Technicals Only (%)" fill="#0284C7" radius={[3, 3, 0, 0]} />
          <Bar dataKey="newsAcc" name="With Overnight News (%)" fill="#0D9488" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
