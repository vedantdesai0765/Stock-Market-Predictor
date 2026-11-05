"use client";

import React, { useState, useRef, useEffect } from "react";
import { useSelectedStock } from "@/context/StockContext";
import { useStocks } from "@/hooks/useStockData";
import { ChevronDown, Check, Search, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";

export function StockSelector() {
  const { selectedTicker, setSelectedTicker } = useSelectedStock();
  const { data: stocksData, isLoading } = useStocks();
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);

  const stocks = stocksData?.stocks || [
    { ticker: "ADANIENT", symbol: "ADANIENT.NS", display_name: "Adani Enterprises", has_model: true },
    { ticker: "ADANIGREEN", symbol: "ADANIGREEN.NS", display_name: "Adani Green Energy", has_model: true },
    { ticker: "ETERNAL", symbol: "ETERNAL.NS", display_name: "Eternal Limited", has_model: true },
    { ticker: "HDFCBANK", symbol: "HDFCBANK.NS", display_name: "HDFC Bank", has_model: true },
    { ticker: "INFY", symbol: "INFY.NS", display_name: "Infosys", has_model: true },
    { ticker: "ITC", symbol: "ITC.NS", display_name: "ITC", has_model: true },
    { ticker: "RELIANCE", symbol: "RELIANCE.NS", display_name: "Reliance Industries", has_model: true },
    { ticker: "SBI", symbol: "SBIN.NS", display_name: "State Bank of India", has_model: true },
    { ticker: "TCS", symbol: "TCS.NS", display_name: "Tata Consultancy Services", has_model: true },
    { ticker: "VEDL", symbol: "VEDL.NS", display_name: "Vedanta", has_model: true },
  ];

  const currentStock = stocks.find((s) => s.ticker === selectedTicker) || stocks[0];

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filtered = stocks.filter(
    (s) =>
      s.ticker.toLowerCase().includes(query.toLowerCase()) ||
      s.display_name.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="relative inline-block text-left" ref={dropdownRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 transition-colors shadow-xs text-left focus:outline-hidden focus:ring-2 focus:ring-sky-500/20"
      >
        <div className="flex items-center justify-center w-6 h-6 rounded bg-sky-50 text-sky-700 font-mono font-bold text-xs border border-sky-100">
          ₹
        </div>
        <div>
          <div className="flex items-center gap-1.5">
            <span className="font-mono font-bold text-sm text-slate-900 leading-none">
              {currentStock.ticker}
            </span>
            <span className="text-[11px] font-medium text-slate-600 bg-slate-100 px-1 rounded">
              NSE
            </span>
          </div>
          <span className="text-xs text-slate-600 truncate block max-w-[130px] sm:max-w-[180px] leading-tight">
            {currentStock.display_name}
          </span>
        </div>
        <ChevronDown className={cn("w-4 h-4 text-slate-400 transition-transform ml-1", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <div className="absolute left-0 mt-1.5 w-72 rounded-lg bg-white border border-slate-200 shadow-xl z-50 overflow-hidden animate-in fade-in-50 zoom-in-95 duration-100">
          <div className="p-2 border-b border-slate-100 bg-slate-50/70">
            <div className="flex items-center gap-2 px-2.5 py-1 rounded bg-white border border-slate-200">
              <Search className="w-3.5 h-3.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search symbol or name..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full text-xs outline-hidden text-slate-900 placeholder:text-slate-400 bg-transparent"
                autoFocus
              />
            </div>
          </div>

          <div className="max-h-64 overflow-y-auto p-1 divide-y divide-slate-50">
            {filtered.map((stock) => {
              const isSelected = stock.ticker === selectedTicker;
              return (
                <button
                  key={stock.ticker}
                  type="button"
                  onClick={() => {
                    setSelectedTicker(stock.ticker);
                    setIsOpen(false);
                    setQuery("");
                  }}
                  className={cn(
                    "w-full flex items-center justify-between px-3 py-2 text-left rounded-md transition-colors text-xs",
                    isSelected
                      ? "bg-sky-50/80 text-sky-900 font-medium"
                      : "hover:bg-slate-50 text-slate-700"
                  )}
                >
                  <div className="flex flex-col">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono font-bold text-slate-900">{stock.ticker}</span>
                      <span className="text-[10px] text-slate-600 bg-slate-100 px-1 rounded">NSE</span>
                    </div>
                    <span className="text-slate-600 truncate max-w-[190px]">{stock.display_name}</span>
                  </div>
                  {isSelected && <Check className="w-4 h-4 text-sky-600 ml-2" />}
                </button>
              );
            })}
            {filtered.length === 0 && (
              <div className="py-4 text-center text-xs text-slate-600">No stocks match your query</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
