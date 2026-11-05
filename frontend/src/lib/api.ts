import {
  SystemHealth,
  StockListResponse,
  StockItem,
  DataStatusResponse,
  PredictionResponse,
  ForecastResponse,
  LatestSignalResponse,
  BacktestResponse,
  NewsResponse,
  WalletSummaryResponse,
  OrderPreviewRequest,
  OrderPreviewResponse,
  ExecuteOrderRequest,
  ExecuteOrderResponse,
  PortfolioAnalyticsResponse,
  TransactionItem,
} from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api/v1";

export class ApiError extends Error {
  statusCode: number;
  errorCode?: string;

  constructor(message: string, statusCode: number, errorCode?: string) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.errorCode = errorCode;
  }
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers || {}),
      },
      next: { revalidate: 0 }, // Disable static caching for live API state
    });

    if (!res.ok) {
      let errorMsg = `API request failed with status ${res.status}`;
      let errorCode: string | undefined;
      try {
        const errorData = await res.json();
        if (errorData.message) errorMsg = errorData.message;
        if (errorData.error_code) errorCode = errorData.error_code;
      } catch {
        // Fallback to text status
      }
      throw new ApiError(errorMsg, res.status, errorCode);
    }

    return (await res.json()) as T;
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      err?.message || "Failed to connect to backend server at " + API_BASE,
      503,
      "CONNECTION_REFUSED"
    );
  }
}

export const api = {
  getHealth: () => request<SystemHealth>("/health"),
  getStocks: () => request<StockListResponse>("/stocks"),
  getStock: (ticker: string) => request<StockItem>(`/stocks/${ticker}`),
  getDataStatus: (ticker: string) => request<DataStatusResponse>(`/stocks/${ticker}/data-status`),
  getPrediction: (ticker: string) => request<PredictionResponse>(`/stocks/${ticker}/prediction`),
  getForecast: (ticker: string) => request<ForecastResponse>(`/stocks/${ticker}/forecast`),
  getLatestSignal: (ticker: string, variant: string = "A", frictionBps: number = 15.0) =>
    request<LatestSignalResponse>(`/stocks/${ticker}/signals/latest?variant=${variant}&friction_bps=${frictionBps}`),
  getBacktest: (
    ticker: string,
    variant: string = "A",
    longOnly: boolean = true,
    frictionBps: number = 15.0,
    capital: number = 1000000.0
  ) =>
    request<BacktestResponse>(
      `/stocks/${ticker}/backtest?variant=${variant}&long_only=${longOnly}&friction_bps=${frictionBps}&starting_capital=${capital}`
    ),
  getNews: (ticker: string, limit: number = 50) =>
    request<NewsResponse>(`/stocks/${ticker}/news?limit=${limit}`),
  getPortfolio: () => request<WalletSummaryResponse>("/portfolio"),
  previewOrder: (data: OrderPreviewRequest) =>
    request<OrderPreviewResponse>("/portfolio/preview", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  executeOrder: (data: ExecuteOrderRequest) =>
    request<ExecuteOrderResponse>("/portfolio/orders", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  getTransactions: (limit: number = 50, offset: number = 0) =>
    request<TransactionItem[]>(`/portfolio/transactions?limit=${limit}&offset=${offset}`),
  getPortfolioAnalytics: () => request<PortfolioAnalyticsResponse>("/portfolio/analytics"),
  resetPortfolio: () =>
    request<{ status: string; message: string }>("/portfolio/reset", {
      method: "POST",
    }),
};
