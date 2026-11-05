"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

interface StockContextType {
  selectedTicker: string;
  setSelectedTicker: (ticker: string) => void;
}

const StockContext = createContext<StockContextType>({
  selectedTicker: "TCS",
  setSelectedTicker: () => {},
});

export function StockProvider({ children }: { children: React.ReactNode }) {
  const [selectedTicker, setSelectedTickerState] = useState<string>("TCS");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("selected_stock_ticker");
      if (saved) {
        setSelectedTickerState(saved);
      }
    } catch {
      // Ignore localStorage failure in SSR
    }
  }, []);

  const setSelectedTicker = (ticker: string) => {
    setSelectedTickerState(ticker);
    try {
      localStorage.setItem("selected_stock_ticker", ticker);
    } catch {
      // Ignore
    }
  };

  return (
    <StockContext.Provider value={{ selectedTicker, setSelectedTicker }}>
      {children}
    </StockContext.Provider>
  );
}

export function useSelectedStock() {
  return useContext(StockContext);
}
