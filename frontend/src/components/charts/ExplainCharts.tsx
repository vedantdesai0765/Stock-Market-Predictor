"use client";

import React, { useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  Cell,
} from "recharts";
import { SentimentTimelinePoint, LeadLagPoint } from "@/types/api";
import { formatINR } from "@/lib/utils";

interface SentimentPriceTimelineProps {
  timeline: SentimentTimelinePoint[];
  onSelectDate?: (date: string) => void;
}

export function SentimentPriceTimelineChart({
  timeline,
  onSelectDate,
}: SentimentPriceTimelineProps) {
  if (!timeline || timeline.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center text-xs text-slate-400 font-mono">
        No price & sentiment timeline data available
      </div>
    );
  }

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={timeline}
          onClick={(state) => {
            if (state && state.activeLabel && onSelectDate) {
              onSelectDate(state.activeLabel);
            }
          }}
          margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: "#64748B" }}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            yAxisId="price"
            domain={["auto", "auto"]}
            tick={{ fontSize: 10, fill: "#64748B" }}
            tickFormatter={(val) => `₹${Math.round(val)}`}
          />
          <YAxis
            yAxisId="sent"
            orientation="right"
            domain={[-0.8, 0.8]}
            tick={{ fontSize: 10, fill: "#94A3B8" }}
            tickFormatter={(val) => (val > 0 ? `+${val}` : `${val}`)}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const pt = payload[0].payload as SentimentTimelinePoint;
                const hasNews = pt.article_count > 0;
                const isPos = (pt.sentiment_score || 0) > 0.05;
                const isNeg = (pt.sentiment_score || 0) < -0.05;

                return (
                  <div className="bg-white p-3 rounded-lg shadow-xl border border-slate-200 text-xs max-w-sm">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-1 mb-1.5 font-mono">
                      <span className="font-bold text-slate-900">{label}</span>
                      {pt.close && <span className="text-slate-600 font-semibold">{formatINR(pt.close)}</span>}
                    </div>

                    {hasNews ? (
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-bold font-mono uppercase ${
                              isPos
                                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                                : isNeg
                                ? "bg-rose-50 text-rose-700 border border-rose-200"
                                : "bg-slate-50 text-slate-700 border border-slate-200"
                            }`}
                          >
                            Sentiment: {pt.sentiment_score ? (pt.sentiment_score > 0 ? `+${pt.sentiment_score.toFixed(3)}` : pt.sentiment_score.toFixed(3)) : "0.00"}
                          </span>
                          <span className="text-slate-400 text-[10px]">
                            {pt.article_count} headline{pt.article_count > 1 ? "s" : ""}
                          </span>
                        </div>
                        {pt.top_headline && (
                          <p className="text-[11px] text-slate-800 font-sans line-clamp-2 mt-1">
                            &ldquo;{pt.top_headline}&rdquo;
                          </p>
                        )}
                        <p className="text-[10px] text-sky-600 font-mono mt-1">Click to filter news below ↓</p>
                      </div>
                    ) : (
                      <p className="text-slate-400 text-[11px] italic">No news published on this trading day</p>
                    )}
                  </div>
                );
              }
              return null;
            }}
          />
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="close"
            name="Stock Price"
            stroke="#0284C7"
            strokeWidth={2}
            dot={(props) => {
              const { cx, cy, payload } = props;
              const pt = payload as SentimentTimelinePoint;
              if (!pt || pt.article_count === 0) return <circle key={`dot-${pt?.date}`} r={0} />;

              const isPos = (pt.sentiment_score || 0) > 0.05;
              const isNeg = (pt.sentiment_score || 0) < -0.05;
              const fill = isPos ? "#10B981" : isNeg ? "#F43F5E" : "#94A3B8";

              return (
                <circle
                  key={`dot-${pt.date}`}
                  cx={cx}
                  cy={cy}
                  r={5}
                  fill={fill}
                  stroke="#FFFFFF"
                  strokeWidth={1.5}
                  className="cursor-pointer hover:scale-125 transition-transform"
                />
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

interface LeadLagChartProps {
  leadLag: LeadLagPoint[];
}

export function LeadLagChart({ leadLag }: LeadLagChartProps) {
  if (!leadLag || leadLag.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-xs text-slate-400 font-mono">
        No lead-lag correlation data available
      </div>
    );
  }

  // Filter for 'all headlines' or first available variant
  const filtered = leadLag.filter((p) => p.variant.toLowerCase().includes("all") || p.variant.toLowerCase().includes("excluding"));
  const displayData = (filtered.length > 0 ? filtered : leadLag).slice(0, 11).map((p) => ({
    lag: p.lag === 0 ? "Day 0 (Same-day)" : p.lag < 0 ? `T${p.lag} (News Leads)` : `T+${p.lag} (Price Leads)`,
    rawLag: p.lag,
    rho: p.rho,
    p: p.p,
    n: p.n,
    significant: p.p < 0.05,
  }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={displayData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
          <XAxis dataKey="lag" tick={{ fontSize: 10, fill: "#64748B" }} />
          <YAxis
            domain={[-0.05, 0.25]}
            tick={{ fontSize: 10, fill: "#64748B" }}
            tickFormatter={(val) => val.toFixed(2)}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const item = payload[0].payload;
                return (
                  <div className="bg-white p-2.5 rounded shadow-lg border border-slate-200 text-xs font-mono">
                    <p className="font-bold text-slate-900">{item.lag}</p>
                    <p className="text-sky-600 font-semibold">
                      Spearman Rank Correlation (ρ): {item.rho >= 0 ? `+${item.rho.toFixed(4)}` : item.rho.toFixed(4)}
                    </p>
                    <p className="text-slate-500 text-[10px]">Statistical p-value: {item.p.toFixed(5)}</p>
                    <p className="text-[10px] text-slate-400">Sample size: {item.n.toLocaleString()} sessions</p>
                    <div className="mt-1 pt-1 border-t border-slate-100 text-[10px] text-slate-600 font-sans">
                      {item.rawLag === 0 && "Concurrent news and market reaction."}
                      {item.rawLag < 0 && "Prior news anticipating future movement (genuine predictive lead)."}
                      {item.rawLag > 0 && "News published after the price moved (retrospective reporting)."}
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <ReferenceLine y={0} stroke="#94A3B8" />
          <Bar dataKey="rho" radius={[3, 3, 0, 0]}>
            {displayData.map((entry, index) => {
              const isLead = entry.rawLag < 0;
              const isSame = entry.rawLag === 0;
              const fill = isSame ? "#0284C7" : isLead ? "#10B981" : "#F59E0B";
              return <Cell key={`cell-${index}`} fill={fill} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
