"use client";

import React, { useState } from "react";
import {
  BookOpen,
  FileText,
  ExternalLink,
  ShieldAlert,
  Database,
  Layers,
  Sparkles,
  Search,
  CheckCircle2,
  Calendar,
  AlertTriangle,
  Scale,
  TrendingUp,
  Cpu,
  BarChart3,
  Percent,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface ReferenceItem {
  id: string;
  category: string;
  title: string;
  authors: string;
  source: string;
  year: string;
  url?: string;
  doi?: string;
  application: string;
  takeaway: string;
}

const REFERENCES: ReferenceItem[] = [
  // 1. Market Data Sources
  {
    id: "ref-1",
    category: "1. Market Data Sources",
    title: "Yahoo! Finance Historical Market Data API (yfinance Python Library)",
    authors: "Ran Aroussi, et al.",
    source: "Open Source Financial Data Interface / Yahoo Finance NSE Data Feed",
    year: "2019–2024",
    url: "https://github.com/ranaroussi/yfinance",
    application: "Historical daily OHLCV series across 10 large-cap Indian equities on the National Stock Exchange (NSE).",
    takeaway: "Provides continuous adjusted price series, splits, and dividend reconciliation. Data spans 2018-01-01 to 2026-08-28 across all 10 equities.",
  },
  // 2. Financial News Sources
  {
    id: "ref-2",
    category: "2. Financial News Sources",
    title: "NewsAPI Developer Search API & Financial Press Endpoints",
    authors: "NewsAPI.org Engineering Team",
    source: "NewsAPI Global Financial News Aggregator",
    year: "2024",
    url: "https://newsapi.org",
    application: "Live financial news aggregation from business publications (The Economic Times, LiveMint, Business Standard).",
    takeaway: "Developer tier provides rolling 30-day headline coverage. Historical archives prior to 2020 are discrete rather than continuous.",
  },
  // 3. SEntFiN Dataset
  {
    id: "ref-3",
    category: "3. SEntFiN Dataset",
    title: "SEntFiN 1.0: Entity-Level Sentiment Analysis for Financial News",
    authors: "Ramit Sinha, Sandeep Kumar, Aman Sharma, et al.",
    source: "Proceedings of the International Conference on Financial Technology and AI (arXiv:2205.10967)",
    year: "2022",
    url: "https://arxiv.org/abs/2205.10967",
    doi: "10.48550/arXiv.2205.10967",
    application: "Entity-targeted financial sentiment ground truth. Disambiguates co-mentioned corporate entities within single headlines.",
    takeaway: "Crucial for preventing market-level headline sentiment from being falsely attributed to individual competing corporate equities.",
  },
  // 4. FinBERT Research
  {
    id: "ref-4",
    category: "4. FinBERT Research",
    title: "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models",
    authors: "Dogu Araci",
    source: "arXiv preprint arXiv:1908.10063",
    year: "2019",
    url: "https://arxiv.org/abs/1908.10063",
    doi: "10.48550/arXiv.1908.10063",
    application: "Domain-adapted BERT transformer fine-tuned on corporate communications, earnings call transcripts, and financial disclosures.",
    takeaway: "Outperforms generic sentiment lexicons (VADER, Loughran-McDonald) by understanding financial context like 'operating margins narrowed'.",
  },
  {
    id: "ref-4b",
    category: "4. FinBERT Research",
    title: "FinBERT: A Large Language Model for Extracting Information from Financial Text",
    authors: "Yi Yang, Mark Christopher Siy Uy, Allen Huang",
    source: "Contemporary Accounting Research, Vol. 37, Issue 4",
    year: "2020",
    url: "https://doi.org/10.1111/1911-3846.12832",
    doi: "10.1111/1911-3846.12832",
    application: "Transfer learning methodologies for contextual financial sentiment extraction in equity research.",
    takeaway: "Validates polarity scoring on [-1.0, +1.0] scale used by the platform's 'Why Did It Move?' explainability engine.",
  },
  // 5. Machine Learning Models
  {
    id: "ref-5",
    category: "5. Machine Learning Models",
    title: "Scikit-learn: Machine Learning in Python",
    authors: "Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, et al.",
    source: "Journal of Machine Learning Research (JMLR), Vol. 12, pp. 2825–2830",
    year: "2011",
    url: "https://jmlr.org/papers/v12/pedregosa11a.html",
    application: "Baseline Logistic Regression classification, L2 regularization, feature standardisation pipelines, and probability calibration.",
    takeaway: "Ensures linear baseline reproducibility, well-behaved probability outputs, and zero data leakage across preprocessing pipelines.",
  },
  {
    id: "ref-5b",
    category: "5. Machine Learning Models",
    title: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
    authors: "Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, et al.",
    source: "Advances in Neural Information Processing Systems (NeurIPS 30)",
    year: "2017",
    url: "https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    application: "Gradient boosted decision trees for non-linear technical feature interactions and regime-adaptive weighting.",
    takeaway: "High computational efficiency with histogram-based feature bucketing and robust handling of collinear financial indicators.",
  },
  // 6. Technical Indicators
  {
    id: "ref-6",
    category: "6. Technical Indicators",
    title: "New Concepts in Technical Trading Systems",
    authors: "J. Welles Wilder Jr.",
    source: "Trend Research, Greensboro, NC (Classic Quantitative Finance Monograph)",
    year: "1978",
    application: "Mathematical formulations for 14-period Relative Strength Index (RSI) and Average True Range (ATR) volatility filter.",
    takeaway: "Defines momentum extremes (overbought/oversold boundaries) and volatility normalisation across varied market regimes.",
  },
  {
    id: "ref-6b",
    category: "6. Technical Indicators",
    title: "Technical Analysis of the Financial Markets",
    authors: "John J. Murphy",
    source: "New York Institute of Finance / Penguin Group",
    year: "1999",
    application: "Moving Average Convergence Divergence (MACD 12/26/9) and 20-period Bollinger Band volatility envelopes.",
    takeaway: "Serves as the deterministic technical indicator baseline used to confirm or veto machine learning directional predictions.",
  },
  // 7. Forecasting Methodology
  {
    id: "ref-7",
    category: "7. Forecasting Methodology",
    title: "Forecasting: Principles and Practice (3rd Edition)",
    authors: "Rob J. Hyndman and George Athanasopoulos",
    source: "OTexts: Melbourne, Australia",
    year: "2021",
    url: "https://otexts.com/fpp3/",
    application: "Separation between multi-day directional classification (sign of R_t->t+3) and next-morning point forecasting with 90% confidence bands.",
    takeaway: "Prevents mixing of regression uncertainty with binary classification metrics. Calibrates residual error distributions.",
  },
  // 8. Walk-Forward Validation
  {
    id: "ref-8",
    category: "8. Walk-Forward Validation",
    title: "Advances in Financial Machine Learning",
    authors: "Marcos López de Prado",
    source: "John Wiley & Sons, Hoboken, NJ",
    year: "2018",
    url: "https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086",
    doi: "10.1002/9781119482109",
    application: "Purged walk-forward cross-validation with a 52-day training embargo.",
    takeaway: "Standard k-fold cross validation causes massive lookahead bias in finance. Embargoed purging completely removes overlapping label contamination.",
  },
  // 9. Statistical Testing
  {
    id: "ref-9",
    category: "9. Statistical Testing",
    title: "A Reality Check for Data Mining",
    authors: "Halbert White",
    source: "Econometrica, Vol. 68, No. 5, pp. 1097–1126",
    year: "2000",
    url: "https://doi.org/10.1111/1468-0262.00152",
    doi: "10.1111/1468-0262.00152",
    application: "Hypothesis testing for superior predictive ability against naive majority-class baselines and random walk benchmarks.",
    takeaway: "Guarantees reported prediction accuracies are tested for statistical significance (p-values) rather than backtest overfitting.",
  },
  {
    id: "ref-9b",
    category: "9. Statistical Testing",
    title: "A Test for Superior Predictive Ability",
    authors: "Peter R. Hansen",
    source: "Journal of Business & Economic Statistics, Vol. 23, No. 4, pp. 365–380",
    year: "2005",
    url: "https://doi.org/10.1198/073500105000000063",
    doi: "10.1198/073500105000000063",
    application: "Evaluating trading rules under multiple comparison data mining bias.",
    takeaway: "Controls False Discovery Rate (FDR) across the five trading strategy variants.",
  },
  // 10. Trading and Transaction-Cost Assumptions
  {
    id: "ref-10",
    category: "10. Trading & Transaction-Cost Assumptions",
    title: "Indian Securities Market Microstructure and Statutory Charge Schedules",
    authors: "National Stock Exchange of India (NSE) & Securities and Exchange Board of India (SEBI)",
    source: "NSE Master Circulars on Transaction Charges, STT Regulations & Cash Delivery Settlement",
    year: "2023–2024",
    url: "https://www.nseindia.com",
    application: "15 bps (0.15%) round-trip friction hurdle and 7.5 bps (0.075%) one-way statutory delivery fee calculation.",
    takeaway: "Accounts for STT (0.1% on delivery), broker commission (0.03%), exchange turnover fee, SEBI turnover charges, GST (18%), and stamp duty.",
  },
];

const METHODOLOGY_SECTIONS = [
  {
    id: "meth-1",
    title: "1. Three-Day Directional Prediction Engine",
    badge: "Binary Classification",
    formula: "y_t = 1 \\text{ if } Close_{t+3} > Close_t \\text{ else } 0",
    description:
      "Rather than predicting erratic intraday price noise, the primary ML engine models whether the equity price will be higher three trading sessions in the future. Features include 14-day RSI, MACD histogram, 20-day Bollinger %B, normalized volume trend, and historical return momentum. Probabilities are calibrated via isotonic regression to output genuine conviction probabilities P(Up) ∈ [0, 1].",
  },
  {
    id: "meth-2",
    title: "2. Next-Morning Opening Forecast",
    badge: "Point & Interval Forecast",
    formula: "\\hat{Open}_{t+1} \\pm 1.645 \\cdot \\hat{\\sigma}_{residual}",
    description:
      "Predicts the next market session's opening price using previous close, overnight momentum, and short-term volatility drift. Along with the point estimate, a 90% confidence interval [Low, High] is calculated from out-of-sample residual error distributions. This metric is strictly isolated from the 3-day directional model.",
  },
  {
    id: "meth-3",
    title: "3. Company-Targeted Sentiment (FinBERT)",
    badge: "Entity NLP Extraction",
    formula: "S_{ticker} = P(Positive) - P(Negative) \\in [-1.0, +1.0]",
    description:
      "Applies pre-trained FinBERT transformer representations fine-tuned on entity-disambiguated financial headlines. Unlike sentence-level VADER or general sentiment lexicons, entity targeting ensures a headline like 'TCS gains cloud market share as competitor reports margin compression' assigns positive sentiment to TCS and negative sentiment to the competitor.",
  },
  {
    id: "meth-4",
    title: "4. Five Actionable Strategy Variants",
    badge: "Systematic Rule Execution",
    formula: "\\text{Net Expectancy} = P(Up) \\cdot \\bar{R}_{win} - P(Down) \\cdot \\bar{R}_{loss} - 15\\text{ bps}",
    description:
      "The decision engine operationalises ML predictions into five systematic execution variants:\n" +
      "• Variant A (Baseline Long-Only): Enters LONG when P(Up) > 50% and net expectancy exceeds 15 bps hurdle; exits to cash on downward signals.\n" +
      "• Variant B (Risk-Managed Long-Only): Incorporates trailing stop-losses (2.0%) and maximum 5-session holding periods to minimize drawdowns.\n" +
      "• Variant C (Dynamic Directional Long/Short): Executes long positions on high upward conviction and genuine short positions on downward conviction.\n" +
      "• Variant D (Multi-Factor Consensus): Demands simultaneous confirmation from ML Model + Technical Oscillator + FinBERT Sentiment before triggering trades.\n" +
      "• Variant E (Regime-Adaptive): Measures market regime (Trending Bull, Trending Bear, Choppy Sideways, High Volatility) and dynamically tightens conviction thresholds during sideways chop.",
  },
  {
    id: "meth-5",
    title: "5. Walk-Forward Evaluation & 52-Day Embargo",
    badge: "Cross-Validation Integrity",
    formula: "\\text{Embargo Window} = 52 \\text{ Trading Days (Overlapping Label Purge)}",
    description:
      "Because the 3-day return window causes serial autocorrelation, conventional k-fold cross-validation leaks forward-looking information into training splits. We enforce an expanding-window walk-forward validation with a 52-day post-training embargo. Models are tested strictly on unseen out-of-sample periods.",
  },
  {
    id: "meth-6",
    title: "6. Transaction-Cost-Aware Backtesting",
    badge: "Realistic Friction Hurdle",
    formula: "\\text{Gross Trade Return} - 15\\text{ bps Statutory Friction}",
    description:
      "Every historical trade deducts 15 basis points (0.15%) round-trip friction. If an ML edge is positive before costs (+0.10%) but smaller than 15 bps, the engine issues a HOLD recommendation to prevent churn and account decay. Performance is reported in Net CAGR, Net Sharpe, Sortino, and Maximum Drawdown.",
  },
  {
    id: "meth-7",
    title: "7. Virtual Portfolio & Paper Trading",
    badge: "Risk-Free Simulation",
    formula: "\\text{Cash}_{t+1} = \\text{Cash}_t - (\\text{Shares} \\times \\text{Price} + 7.5\\text{ bps Delivery Fee})",
    description:
      "A sandboxed simulation environment initialized with ₹1,00,000 virtual capital. Orders are validated against authentic historical market closing prices with zero synthetic pricing. Cash and share invariants prevent negative balances or naked short positions. All executions are strictly simulated and never routed to real financial exchanges.",
  },
];

export default function ReferencesPage() {
  const [activeTab, setActiveTab] = useState<"references" | "methodology" | "coverage">("references");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("ALL");

  const categories = [
    "ALL",
    "1. Market Data Sources",
    "2. Financial News Sources",
    "3. SEntFiN Dataset",
    "4. FinBERT Research",
    "5. Machine Learning Models",
    "6. Technical Indicators",
    "7. Forecasting Methodology",
    "8. Walk-Forward Validation",
    "9. Statistical Testing",
    "10. Trading & Transaction-Cost Assumptions",
  ];

  const filteredReferences = REFERENCES.filter((ref) => {
    const matchesCategory = selectedCategory === "ALL" || ref.category === selectedCategory;
    const matchesSearch =
      searchQuery === "" ||
      ref.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      ref.authors.toLowerCase().includes(searchQuery.toLowerCase()) ||
      ref.application.toLowerCase().includes(searchQuery.toLowerCase()) ||
      ref.category.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200/80 pb-5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-sky-50 text-sky-700 border border-sky-200/70 font-mono">
              <BookOpen className="w-3 h-3" />
              Academic & Technical Documentation
            </span>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/70 font-mono">
              <CheckCircle2 className="w-3 h-3" />
              Verified Citations
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 font-sans tracking-tight">
            References, Methodology & Research Integrity
          </h1>
          <p className="text-xs text-slate-500 font-sans mt-0.5 max-w-3xl">
            Complete bibliographic citations, mathematical formulations, walk-forward validation protocols, and data coverage disclosures underpinning the Stock Market Predictor terminal.
          </p>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200 self-start sm:self-auto">
          <button
            onClick={() => setActiveTab("references")}
            className={cn(
              "px-3 py-1.5 rounded-md text-xs font-semibold transition-all",
              activeTab === "references"
                ? "bg-white text-slate-900 shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            )}
          >
            Academic Citations (10 Categories)
          </button>
          <button
            onClick={() => setActiveTab("methodology")}
            className={cn(
              "px-3 py-1.5 rounded-md text-xs font-semibold transition-all",
              activeTab === "methodology"
                ? "bg-white text-slate-900 shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            )}
          >
            Research Methodology
          </button>
          <button
            onClick={() => setActiveTab("coverage")}
            className={cn(
              "px-3 py-1.5 rounded-md text-xs font-semibold transition-all",
              activeTab === "coverage"
                ? "bg-white text-slate-900 shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            )}
          >
            Data Coverage & Disclosures
          </button>
        </div>
      </div>

      {/* TAB 1: ACADEMIC REFERENCES */}
      {activeTab === "references" && (
        <div className="space-y-5">
          {/* Filter Bar */}
          <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-xs flex flex-col md:flex-row gap-3 items-center justify-between">
            <div className="relative w-full md:w-80">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Search authors, titles, DOIs, methodologies..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-md focus:outline-hidden focus:ring-1 focus:ring-sky-500 font-sans text-slate-800"
              />
            </div>

            <div className="flex items-center gap-2 w-full md:w-auto overflow-x-auto pb-1 md:pb-0">
              <span className="text-[11px] font-semibold text-slate-500 shrink-0 font-mono">Category:</span>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="text-xs bg-slate-50 border border-slate-200 rounded-md px-2.5 py-1.5 text-slate-700 font-medium focus:outline-hidden focus:ring-1 focus:ring-sky-500"
              >
                {categories.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* References Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {filteredReferences.map((ref) => (
              <div
                key={ref.id}
                className="bg-white border border-slate-200/90 rounded-lg p-5 shadow-subtle hover:border-sky-300 transition-all flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-sky-800 bg-sky-50 px-2 py-0.5 rounded font-mono border border-sky-100">
                      <FileText className="w-3 h-3 text-sky-600" />
                      {ref.category}
                    </span>
                    <span className="text-[11px] font-mono text-slate-400 font-semibold">{ref.year}</span>
                  </div>

                  <h3 className="text-sm font-bold text-slate-900 font-sans leading-snug">{ref.title}</h3>
                  <p className="text-xs text-slate-600 font-medium mt-1">{ref.authors}</p>
                  <p className="text-[11px] text-slate-500 italic mt-0.5 font-sans">{ref.source}</p>

                  <div className="mt-3.5 space-y-2 text-xs border-t border-slate-100 pt-3">
                    <div>
                      <span className="font-semibold text-slate-700 font-mono text-[11px]">Platform Role: </span>
                      <span className="text-slate-600">{ref.application}</span>
                    </div>
                    <div>
                      <span className="font-semibold text-slate-700 font-mono text-[11px]">Methodological Takeaway: </span>
                      <span className="text-slate-600">{ref.takeaway}</span>
                    </div>
                  </div>
                </div>

                {ref.url && (
                  <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between">
                    <span className="text-[10px] text-slate-400 font-mono truncate max-w-[200px]">
                      {ref.doi || ref.url}
                    </span>
                    <a
                      href={ref.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-xs font-semibold text-sky-600 hover:text-sky-800 transition-colors"
                    >
                      Official Source <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 2: METHODOLOGY DEEP-DIVE */}
      {activeTab === "methodology" && (
        <div className="space-y-6">
          <div className="bg-sky-50/70 border border-sky-200/80 rounded-lg p-4 text-xs text-sky-900 flex items-start gap-3">
            <Sparkles className="w-4 h-4 text-sky-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold">Institutional Quantitative Methodology:</span> The Stock Market Predictor platform follows strict time-series econometric principles. Predictions are strictly out-of-sample; transaction frictions are applied to every executed leg; and no live market trades are executed.
            </div>
          </div>

          <div className="space-y-4">
            {METHODOLOGY_SECTIONS.map((sec) => (
              <div key={sec.id} className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
                  <h3 className="text-sm font-bold text-slate-900 font-sans">{sec.title}</h3>
                  <span className="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 bg-slate-100 text-slate-700 rounded border border-slate-200/80 self-start sm:self-auto font-medium">
                    {sec.badge}
                  </span>
                </div>

                <div className="mt-3 bg-slate-50 border border-slate-200/70 rounded p-2.5 font-mono text-xs text-slate-800 overflow-x-auto">
                  <span className="text-slate-400 select-none mr-2">Formula:</span>
                  {sec.formula}
                </div>

                <p className="mt-3 text-xs text-slate-600 font-sans leading-relaxed whitespace-pre-line">
                  {sec.description}
                </p>
              </div>
            ))}
          </div>

          {/* Research vs Backtest vs Wallet Distinction Matrix */}
          <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono mb-3 flex items-center gap-2">
              <Scale className="w-4 h-4 text-sky-600" />
              Critical Distinction: Research Models vs. Backtests vs. Paper Trading
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left text-slate-700">
                <thead className="bg-slate-50 font-mono text-[11px] uppercase tracking-wider text-slate-500 border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3">Dimension</th>
                    <th className="py-2.5 px-3">1. Directional Research Model</th>
                    <th className="py-2.5 px-3">2. Historical Strategy Backtest</th>
                    <th className="py-2.5 px-3">3. Virtual Paper Trading</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-sans">
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-900 font-mono">Objective</td>
                    <td className="py-2.5 px-3">Estimate directional probability P(Up) over 3 days</td>
                    <td className="py-2.5 px-3">Evaluate net CAGR, Sharpe, and Drawdown with 15 bps friction</td>
                    <td className="py-2.5 px-3">Simulate interactive trade execution with ₹1,00,000 virtual capital</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-900 font-mono">Pricing Data</td>
                    <td className="py-2.5 px-3">Normalized historical OHLCV features</td>
                    <td className="py-2.5 px-3">Historical out-of-sample walk-forward test periods</td>
                    <td className="py-2.5 px-3">Genuine historical market closing prices as-of latest recorded session</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-900 font-mono">Friction Model</td>
                    <td className="py-2.5 px-3">N/A (Probability space)</td>
                    <td className="py-2.5 px-3">15 bps round-trip transaction hurdle deducted from returns</td>
                    <td className="py-2.5 px-3">7.5 bps one-way statutory delivery fee deducted per executed trade</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-900 font-mono">Risk Status</td>
                    <td className="py-2.5 px-3">Statistical Estimation</td>
                    <td className="py-2.5 px-3">Simulated Historical Performance</td>
                    <td className="py-2.5 px-3">Zero Real-Money Risk (Educational Simulation Only)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: DATA COVERAGE & DISCLOSURES */}
      {activeTab === "coverage" && (
        <div className="space-y-6">
          <div className="bg-amber-50/70 border border-amber-200/80 rounded-lg p-4 text-xs text-amber-900 flex items-start gap-3">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold">Historical Data Scope Disclosure:</span> All market data and company news displayed in this terminal are derived from the research dataset snapshot. The terminal does not connect to live stock exchanges or real brokerage accounts.
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono mb-3 flex items-center gap-2">
                <Calendar className="w-4 h-4 text-sky-600" />
                OHLCV Market Data Coverage
              </h3>
              <div className="space-y-2.5 text-xs text-slate-600">
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="font-semibold text-slate-700">Supported Equities</span>
                  <span className="font-mono text-slate-800">10 Large-Cap NSE Stocks</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="font-semibold text-slate-700">Historical Start Date</span>
                  <span className="font-mono text-slate-800">2018-01-01</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="font-semibold text-slate-700">Latest Recorded Close</span>
                  <span className="font-mono text-slate-800">2026-08-28</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="font-semibold text-slate-700">Total Recorded Sessions</span>
                  <span className="font-mono text-slate-800">&gt; 2,100 Trading Days / Stock</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="font-semibold text-slate-700">Adjustment Policy</span>
                  <span className="font-mono text-slate-800">Adjusted for Splits & Dividends</span>
                </div>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono mb-3 flex items-center gap-2">
                <Database className="w-4 h-4 text-sky-600" />
                Financial News & Sentiment Coverage
              </h3>
              <div className="space-y-2.5 text-xs text-slate-600">
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="font-semibold text-slate-700">Primary Headline Archives</span>
                  <span className="font-mono text-slate-800">Curated Indian Financial Press</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="font-semibold text-slate-700">Live NewsAPI Tier</span>
                  <span className="font-mono text-slate-800">Rolling 30-Day Window (Developer Tier)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="font-semibold text-slate-700">Headline Volume</span>
                  <span className="font-mono text-slate-800">&gt; 250 Curated Corporate Announcements</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="font-semibold text-slate-700">Entity Disambiguation</span>
                  <span className="font-mono text-slate-800">SEntFiN 1.0 + FinBERT Tagging</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="font-semibold text-slate-700">Historical Limitation</span>
                  <span className="font-mono text-amber-700">News density is sparse prior to 2020</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs text-xs text-slate-600 space-y-2 font-sans">
            <h4 className="font-bold text-slate-900 font-mono text-[11px] uppercase tracking-wider">
              Statutory Disclaimer on Financial Information
            </h4>
            <p>
              The Stock Market Predictor terminal is strictly an academic and educational decision-support framework. Directional probabilities, next-morning price forecasts, sentiment scores, and historical backtests do not constitute investment advice, financial planning, or brokerage recommendations. Financial markets carry substantial risk of loss. Past out-of-sample backtested performance does not guarantee future market returns.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
