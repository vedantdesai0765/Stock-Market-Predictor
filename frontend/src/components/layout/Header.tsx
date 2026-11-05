"use client";

import React, { useState, useEffect } from "react";
import { StockSelector } from "@/components/ui/StockSelector";
import { FreshnessBadge } from "@/components/ui/Badge";
import { useSelectedStock } from "@/context/StockContext";
import { useDataStatus } from "@/hooks/useStockData";
import { ExternalLink, Database, Clock, Menu } from "lucide-react";

interface HeaderProps {
  onToggleMobileMenu?: () => void;
}

export function Header({ onToggleMobileMenu }: HeaderProps) {
  const { selectedTicker } = useSelectedStock();
  const { data: status } = useDataStatus(selectedTicker);
  const [timeStr, setTimeStr] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(
        now.toLocaleTimeString("en-IN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
          timeZone: "Asia/Kolkata",
        }) + " IST"
      );
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="h-16 bg-white border-b border-slate-200 px-4 sm:px-6 flex items-center justify-between shrink-0 shadow-xs z-10">
      <div className="flex items-center gap-3 sm:gap-4">
        {/* Mobile Menu Toggle */}
        <button
          type="button"
          onClick={onToggleMobileMenu}
          className="md:hidden p-2 rounded-md text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
          aria-label="Toggle navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Global Stock Selector */}
        <StockSelector />

        {/* Data Freshness Indicator */}
        <div className="hidden sm:block">
          <FreshnessBadge
            isLive={status?.is_live_market}
            date={status?.raw_data_last_date}
          />
        </div>
      </div>

      <div className="flex items-center gap-5 text-xs text-slate-500 font-sans">
        {/* Live IST Clock */}
        <div className="hidden md:flex items-center gap-1.5 font-mono text-slate-600 bg-slate-50 px-2.5 py-1 rounded border border-slate-200">
          <Clock className="w-3.5 h-3.5 text-slate-400" />
          <span>{timeStr || "Loading clock..."}</span>
        </div>

        {/* API Docs Link */}
        <a
          href="http://127.0.0.1:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-slate-600 hover:text-sky-600 font-medium transition-colors"
          title="Open FastAPI Swagger Interactive Documentation"
        >
          <Database className="w-3.5 h-3.5 text-sky-600" />
          <span>FastAPI Docs</span>
          <ExternalLink className="w-3 h-3 text-slate-400" />
        </a>
      </div>
    </header>
  );
}
