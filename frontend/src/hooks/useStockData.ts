import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => api.getHealth(),
    refetchInterval: 30000, // Heartbeat every 30s
  });
}

export function useStocks() {
  return useQuery({
    queryKey: ["stocks"],
    queryFn: () => api.getStocks(),
    staleTime: 5 * 60 * 1000,
  });
}

export function useStock(ticker: string) {
  return useQuery({
    queryKey: ["stock", ticker],
    queryFn: () => api.getStock(ticker),
    enabled: !!ticker,
  });
}

export function useDataStatus(ticker: string) {
  return useQuery({
    queryKey: ["data-status", ticker],
    queryFn: () => api.getDataStatus(ticker),
    enabled: !!ticker,
  });
}

export function usePrediction(ticker: string) {
  return useQuery({
    queryKey: ["prediction", ticker],
    queryFn: () => api.getPrediction(ticker),
    enabled: !!ticker,
  });
}

export function useForecast(ticker: string) {
  return useQuery({
    queryKey: ["forecast", ticker],
    queryFn: () => api.getForecast(ticker),
    enabled: !!ticker,
  });
}

export function useLatestSignal(
  ticker: string,
  variant: string = "A",
  frictionBps: number = 15.0
) {
  return useQuery({
    queryKey: ["latest-signal", ticker, variant, frictionBps],
    queryFn: () => api.getLatestSignal(ticker, variant, frictionBps),
    enabled: !!ticker,
  });
}

export function useBacktest(
  ticker: string,
  variant: string = "A",
  longOnly: boolean = true,
  frictionBps: number = 15.0,
  capital: number = 1000000.0
) {
  return useQuery({
    queryKey: ["backtest", ticker, variant, longOnly, frictionBps, capital],
    queryFn: () => api.getBacktest(ticker, variant, longOnly, frictionBps, capital),
    enabled: !!ticker,
  });
}

export function useNews(ticker: string, limit: number = 50) {
  return useQuery({
    queryKey: ["news", ticker, limit],
    queryFn: () => api.getNews(ticker, limit),
    enabled: !!ticker,
  });
}

export function usePortfolio() {
  return useQuery({
    queryKey: ["portfolio"],
    queryFn: () => api.getPortfolio(),
    refetchInterval: 10000, // Background poll every 10s
  });
}

export function usePortfolioAnalytics() {
  return useQuery({
    queryKey: ["portfolio-analytics"],
    queryFn: () => api.getPortfolioAnalytics(),
    refetchInterval: 15000,
  });
}
