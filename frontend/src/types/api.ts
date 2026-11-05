/**
 * TypeScript types matching FastAPI Pydantic schemas.
 */

export interface SystemHealth {
  status: string;
  project: string;
  version: string;
  timestamp: string;
  python_version: string;
  registered_stocks: number;
  database: string;
  api_prefix: string;
}

export interface StockItem {
  ticker: string;
  symbol: string;
  display_name: string;
  currency: string;
  start_date?: string;
  end_date?: string;
  has_model: boolean;
  has_raw_data: boolean;
  has_sentiment: boolean;
  notes?: string;
}

export interface StockListResponse {
  stocks: StockItem[];
  total: number;
}

export interface DataStatusResponse {
  ticker: string;
  display_name: string;
  raw_data_file: string;
  raw_data_exists: boolean;
  raw_data_last_date?: string;
  raw_data_rows?: number;
  model_exists: boolean;
  model_trained_at?: string;
  oof_eval_last_date?: string;
  news_file_exists: boolean;
  news_last_date?: string;
  news_count?: number;
  status_summary: string;
  is_live_market: boolean;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface CoverageCurvePoint {
  threshold: number;
  coverage: number;
  accuracy: number;
  edge: number;
  n_committed: number;
}

export interface PredictionResponse {
  ticker: string;
  display_name: string;
  date: string;
  close: number;
  target_horizon_days: number;
  probability_up: number;
  probability_down: number;
  confidence: number;
  confidence_threshold: number;
  signal: "UP" | "DOWN" | "NO SIGNAL";
  abstention_active: boolean;
  expected_accuracy?: number;
  expected_coverage?: number;
  walk_forward_accuracy?: number;
  walk_forward_baseline?: number;
  walk_forward_edge?: number;
  model_name?: string;
  trained_at?: string;
  features_count?: number;
  top_features?: FeatureImportanceItem[];
  coverage_curve?: CoverageCurvePoint[];
}

export interface ForecastSummaryItem {
  period: string;
  target: string;
  model: string;
  ticker: string;
  n: number;
  first_year: number;
  last_year: number;
  acc_guess: number;
  acc_model: number;
  acc_news: number;
  edge: number;
  p_vs_guess: number;
  news_effect: number;
  p_news: number;
}

export interface AnnualForecastItem {
  year: number;
  model: string;
  acc_guess: number;
  acc_model: number;
  acc_news: number;
  edge: number;
  sample_count: number;
}

export interface ForecastResponse {
  ticker: string;
  display_name: string;
  summaries: ForecastSummaryItem[];
  best_model?: string;
  best_accuracy?: number;
  best_edge?: number;
  annual_breakdown?: AnnualForecastItem[];
}

export interface LatestSignalResponse {
  ticker: string;
  date: string;
  signal: number; // 1, -1, 0
  signal_label: "BUY" | "SELL" | "HOLD";
  calibrated_proba_up: number;
  confidence: number;
  expected_gross_return: number;
  friction_rate: number;
  net_expectancy: number;
  rationale: string;
  close: number;
  rsi: number;
  macd_hist: number;
  sma20: number;
  sma50: number;
  sentiment_score: number;
  market_regime: string;
  flag_model: boolean;
  flag_tech: boolean;
  flag_sentiment: boolean;
}

export interface BacktestEquityPoint {
  date: string;
  close: number;
  strategy_equity: number;
  buy_hold_equity: number;
  cash_equity: number;
  strategy_return: number;
  drawdown: number;
  position: number;
}

export interface TradeRecordItem {
  entry_date: string;
  entry_price: number;
  position: string;
  exit_date?: string;
  exit_price?: number;
  net_return?: number;
  status: string;
  holding_steps?: number;
}

export interface SeriesMetrics {
  cumulative_return: number;
  cagr: number;
  annualized_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
}

export interface StrategyMetrics extends SeriesMetrics {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  avg_trade_return: number;
  avg_holding_days: number;
  total_friction_bps: number;
  time_in_market: number;
  start_date: string;
  end_date: string;
  years: number;
}

export interface BacktestMetricsResponse {
  strategy: StrategyMetrics;
  buy_and_hold: SeriesMetrics;
  cash: SeriesMetrics;
}

export interface BacktestResponse {
  ticker: string;
  variant: string;
  long_only: boolean;
  friction_bps: number;
  starting_capital: number;
  metrics: BacktestMetricsResponse;
  equity_curve: BacktestEquityPoint[];
  trades: TradeRecordItem[];
}

export interface NewsArticleItem {
  trade_date: string;
  published?: string;
  source?: string;
  entity?: string;
  text: string;
  targeted_score?: number;
  plain_score?: number;
  label?: string;
  price_report?: boolean;
}

export interface LeadLagPoint {
  scope: string;
  variant: string;
  lag: number;
  rho: number;
  p: number;
  n: number;
}

export interface SentimentTimelinePoint {
  date: string;
  close?: number;
  sentiment_score?: number;
  article_count: number;
  sentiment_label?: string;
  top_headline?: string;
}

export interface NewsResponse {
  ticker: string;
  total_articles: number;
  average_sentiment: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  recent_articles: NewsArticleItem[];
  lead_lag?: LeadLagPoint[];
  timeline?: SentimentTimelinePoint[];
}

// ----------------------------------------------------
// Virtual Portfolio & Paper Trading Interfaces
// ----------------------------------------------------

export interface HoldingItem {
  ticker: string;
  display_name: string;
  quantity: number;
  avg_buy_price: number;
  current_price: number;
  price_date: string;
  invested_amount: number;
  current_value: number;
  unrealised_pnl: number;
  return_pct: number;
}

export interface TransactionItem {
  id: string;
  wallet_id: string;
  ticker: string;
  side: "BUY" | "SELL";
  quantity: number;
  execution_price: number;
  gross_amount: number;
  friction_bps: number;
  friction_amount: number;
  net_amount: number;
  signal_source?: string;
  idempotency_key?: string;
  executed_at: string;
}

export interface WalletSummaryResponse {
  id: string;
  name: string;
  currency: string;
  initial_capital: number;
  available_cash: number;
  invested_capital: number;
  portfolio_value: number;
  total_pnl: number;
  realised_pnl: number;
  unrealised_pnl: number;
  total_return_pct: number;
  holdings: HoldingItem[];
  holdings_count: number;
  recent_transactions: TransactionItem[];
}

export interface OrderPreviewRequest {
  ticker: string;
  side: "BUY" | "SELL";
  quantity: number;
}

export interface OrderPreviewResponse {
  ticker: string;
  display_name: string;
  side: "BUY" | "SELL";
  quantity: number;
  execution_price: number;
  price_date: string;
  gross_amount: number;
  friction_bps: number;
  friction_amount: number;
  net_amount: number;
  available_cash: number;
  owned_quantity: number;
  is_valid: boolean;
  validation_error?: string;
  max_shares_buyable: number;
}

export interface ExecuteOrderRequest {
  ticker: string;
  side: "BUY" | "SELL";
  quantity: number;
  signal_source?: string;
  idempotency_key?: string;
}

export interface ExecuteOrderResponse {
  success: boolean;
  order_id: string;
  ticker: string;
  display_name: string;
  side: "BUY" | "SELL";
  quantity: number;
  execution_price: number;
  price_date: string;
  gross_amount: number;
  friction_amount: number;
  net_amount: number;
  cash_balance_after: number;
  message: string;
}

export interface PortfolioEquityPoint {
  timestamp: string;
  portfolio_value: number;
  cash: number;
  invested: number;
}

export interface AssetAllocationItem {
  name: string;
  ticker: string;
  value: number;
  percentage: number;
  color: string;
}

export interface PortfolioAnalyticsResponse {
  equity_curve: PortfolioEquityPoint[];
  asset_allocation: AssetAllocationItem[];
  cash: number;
  invested: number;
  portfolio_value: number;
  realised_pnl: number;
  unrealised_pnl: number;
  total_pnl: number;
  total_trades: number;
  winning_trades: number;
  win_rate: number;
}
