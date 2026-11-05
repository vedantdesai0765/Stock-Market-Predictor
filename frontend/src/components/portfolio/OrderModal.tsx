"use client";

import React, { useState, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { OrderPreviewResponse, ExecuteOrderResponse } from "@/types/api";
import { formatINR } from "@/lib/utils";
import {
  X,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  ShieldAlert,
  Loader2,
  Clock,
  Briefcase,
} from "lucide-react";

interface OrderModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultTicker?: string;
  defaultSide?: "BUY" | "SELL";
  defaultQuantity?: number;
  signalSource?: string;
  ownedQuantity?: number;
  availableCash?: number;
  onSuccess?: () => void;
}

const SUPPORTED_STOCKS = [
  { ticker: "ADANIENT", name: "Adani Enterprises" },
  { ticker: "ADANIGREEN", name: "Adani Green Energy" },
  { ticker: "ETERNAL", name: "Eternal Limited" },
  { ticker: "HDFCBANK", name: "HDFC Bank" },
  { ticker: "INFY", name: "Infosys" },
  { ticker: "ITC", name: "ITC Limited" },
  { ticker: "RELIANCE", name: "Reliance Industries" },
  { ticker: "SBI", name: "State Bank of India" },
  { ticker: "TCS", name: "Tata Consultancy Services" },
  { ticker: "VEDL", name: "Vedanta" },
];

export function OrderModal({
  isOpen,
  onClose,
  defaultTicker = "TCS",
  defaultSide = "BUY",
  defaultQuantity = 10,
  signalSource = "MANUAL",
  ownedQuantity = 0,
  availableCash = 100000.0,
  onSuccess,
}: OrderModalProps) {
  const queryClient = useQueryClient();

  const [ticker, setTicker] = useState(defaultTicker);
  const [side, setSide] = useState<"BUY" | "SELL">(defaultSide);
  const [quantity, setQuantity] = useState(defaultQuantity);
  const [preview, setPreview] = useState<OrderPreviewResponse | null>(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [step, setStep] = useState<"CONFIG" | "CONFIRM">("CONFIG");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successResult, setSuccessResult] = useState<ExecuteOrderResponse | null>(null);

  useEffect(() => {
    if (isOpen) {
      setTicker(defaultTicker);
      setSide(defaultSide);
      setQuantity(defaultQuantity);
      setStep("CONFIG");
      setErrorMsg(null);
      setSuccessResult(null);
    }
  }, [isOpen, defaultTicker, defaultSide, defaultQuantity]);

  // Fetch real-time preview on ticker / side / quantity changes
  useEffect(() => {
    if (!isOpen || quantity <= 0) return;

    let isMounted = true;
    setIsPreviewLoading(true);
    setErrorMsg(null);

    api
      .previewOrder({ ticker, side, quantity })
      .then((res) => {
        if (isMounted) {
          setPreview(res);
          setIsPreviewLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setErrorMsg(err.message || "Failed to load order price preview");
          setIsPreviewLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [isOpen, ticker, side, quantity]);

  if (!isOpen) return null;

  const handleExecute = async () => {
    setIsSubmitting(true);
    setErrorMsg(null);

    try {
      const idempotencyKey = `order-${ticker}-${side}-${quantity}-${Date.now()}`;
      const res = await api.executeOrder({
        ticker,
        side,
        quantity,
        signal_source: signalSource,
        idempotency_key: idempotencyKey,
      });

      setSuccessResult(res);
      queryClient.invalidateQueries({ queryKey: ["portfolio"] });
      queryClient.invalidateQueries({ queryKey: ["portfolio-analytics"] });
      if (onSuccess) onSuccess();
    } catch (err: any) {
      setErrorMsg(err.message || "Trade execution failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
      <div className="relative w-full max-w-lg bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden font-sans animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-6 py-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-sky-100 text-sky-700 flex items-center justify-center">
              <Briefcase className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900 leading-tight">
                Simulated Order Execution
              </h3>
              <p className="text-[11px] text-slate-500 font-mono">
                Statutory 7.5 bps friction · Virtual Capital
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            type="button"
            className="p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-200 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-4">
          {successResult ? (
            /* SUCCESS STATE */
            <div className="text-center py-4 space-y-3">
              <div className="w-12 h-12 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto">
                <CheckCircle2 className="w-7 h-7" />
              </div>
              <h4 className="text-base font-bold text-slate-900">Order Executed Successfully!</h4>
              <p className="text-xs text-slate-600 max-w-sm mx-auto font-sans leading-relaxed">
                {successResult.message}
              </p>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs font-mono text-slate-700 space-y-1.5 max-w-sm mx-auto text-left">
                <div className="flex justify-between">
                  <span>Execution Price:</span>
                  <span className="font-bold">{formatINR(successResult.execution_price)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Friction Paid:</span>
                  <span className="text-amber-700 font-bold">{formatINR(successResult.friction_amount)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Net Cash Impact:</span>
                  <span className="font-bold">{formatINR(successResult.net_amount)}</span>
                </div>
                <div className="flex justify-between border-t border-slate-200 pt-1 text-slate-900">
                  <span>Available Cash After:</span>
                  <span className="font-bold">{formatINR(successResult.cash_balance_after)}</span>
                </div>
              </div>
              <div className="pt-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-5 py-2 rounded-lg bg-sky-600 text-white font-medium text-xs hover:bg-sky-700 transition-colors shadow-xs"
                >
                  Return to Portfolio
                </button>
              </div>
            </div>
          ) : step === "CONFIG" ? (
            /* STEP 1: CONFIGURATION */
            <>
              {/* Buy / Sell Tabs */}
              <div className="flex rounded-lg bg-slate-100 p-1 border border-slate-200">
                <button
                  type="button"
                  onClick={() => setSide("BUY")}
                  className={`flex-1 py-1.5 rounded-md text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                    side === "BUY"
                      ? "bg-emerald-600 text-white shadow-xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <TrendingUp className="w-3.5 h-3.5" /> BUY ORDER
                </button>
                <button
                  type="button"
                  onClick={() => setSide("SELL")}
                  className={`flex-1 py-1.5 rounded-md text-xs font-bold transition-all flex items-center justify-center gap-1.5 ${
                    side === "SELL"
                      ? "bg-rose-600 text-white shadow-xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <TrendingDown className="w-3.5 h-3.5" /> SELL ORDER
                </button>
              </div>

              {/* Stock Selector */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1 font-mono uppercase tracking-wider">
                  Select Stock
                </label>
                <select
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-mono bg-slate-50 border border-slate-300 rounded-lg text-slate-900 outline-hidden focus:ring-2 focus:ring-sky-500/20"
                >
                  {SUPPORTED_STOCKS.map((s) => (
                    <option key={s.ticker} value={s.ticker}>
                      {s.ticker} · {s.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Price & Quantity Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <span className="text-[10px] uppercase font-mono font-bold text-slate-400 block mb-1">
                    Latest Market Close
                  </span>
                  <div className="text-base font-bold text-slate-900 font-mono">
                    {preview ? formatINR(preview.execution_price) : "..."}
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono flex items-center gap-1 mt-0.5">
                    <Clock className="w-3 h-3" /> As-of: {preview?.price_date || "..."}
                  </span>
                </div>

                <div>
                  <label className="block text-[10px] uppercase font-mono font-bold text-slate-400 mb-1">
                    Share Quantity
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={side === "SELL" ? preview?.owned_quantity || 1 : 100000}
                    value={quantity}
                    onChange={(e) => setQuantity(Math.max(1, parseInt(e.target.value) || 1))}
                    className="w-full px-3 py-2 text-sm font-bold font-mono bg-white border border-slate-300 rounded-lg text-slate-900 outline-hidden focus:ring-2 focus:ring-sky-500/20"
                  />
                  <div className="flex gap-1.5 mt-1.5">
                    {[10, 25, 50, 100].map((q) => (
                      <button
                        key={q}
                        type="button"
                        onClick={() => setQuantity(q)}
                        className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-slate-100 text-slate-600 hover:bg-slate-200"
                      >
                        +{q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Cost & Friction Summary Card */}
              {isPreviewLoading ? (
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-center text-xs text-slate-500 font-mono gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-sky-600" />
                  Calculating statutory friction...
                </div>
              ) : preview ? (
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs font-mono space-y-1.5">
                  <div className="flex justify-between text-slate-600">
                    <span>Gross Value:</span>
                    <span>{formatINR(preview.gross_amount)}</span>
                  </div>
                  <div className="flex justify-between text-slate-600">
                    <span className="flex items-center gap-1">
                      Statutory Friction (7.5 bps):
                    </span>
                    <span className="text-amber-700">+{formatINR(preview.friction_amount)}</span>
                  </div>
                  <div className="flex justify-between font-bold text-slate-900 border-t border-slate-200 pt-1.5">
                    <span>Total {side === "BUY" ? "Cost" : "Estimated Proceeds"}:</span>
                    <span className={side === "BUY" ? "text-emerald-700" : "text-rose-700"}>
                      {formatINR(preview.net_amount)}
                    </span>
                  </div>
                  <div className="flex justify-between text-[11px] text-slate-500 pt-0.5">
                    <span>{side === "BUY" ? "Available Cash:" : "Owned Shares:"}</span>
                    <span>
                      {side === "BUY"
                        ? formatINR(preview.available_cash)
                        : `${preview.owned_quantity} shares`}
                    </span>
                  </div>
                </div>
              ) : null}

              {/* Validation Warning */}
              {preview && !preview.is_valid && (
                <div className="p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800 font-sans flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                  <span>{preview.validation_error}</span>
                </div>
              )}

              {/* Proceed Button */}
              <button
                type="button"
                disabled={!preview || !preview.is_valid || isPreviewLoading}
                onClick={() => setStep("CONFIRM")}
                className={`w-full py-2.5 rounded-lg font-bold text-xs uppercase tracking-wider text-white shadow-xs transition-colors flex items-center justify-center gap-1.5 ${
                  !preview || !preview.is_valid
                    ? "bg-slate-300 cursor-not-allowed"
                    : side === "BUY"
                    ? "bg-emerald-600 hover:bg-emerald-700"
                    : "bg-rose-600 hover:bg-rose-700"
                }`}
              >
                Review Order Confirmation →
              </button>
            </>
          ) : (
            /* STEP 2: EXPLICIT CONFIRMATION */
            <div className="space-y-4">
              <div className="p-3.5 rounded-lg bg-amber-50 border border-amber-200/80 text-xs text-amber-900 font-sans flex items-start gap-2.5">
                <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <div className="font-bold font-mono uppercase tracking-wide text-amber-800">
                    Simulated Trading Notice
                  </div>
                  <p className="text-[11px] leading-relaxed text-amber-700">
                    This is a paper trading simulation with virtual capital. No actual funds are
                    debited, and no securities are traded on live exchanges. Model signals are
                    historical research outputs and carry no guaranteed future returns.
                  </p>
                </div>
              </div>

              {/* Order Final Summary */}
              <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 text-xs font-mono space-y-2">
                <div className="flex justify-between border-b border-slate-200 pb-1.5">
                  <span className="text-slate-500">Transaction:</span>
                  <span className={`font-bold ${side === "BUY" ? "text-emerald-700" : "text-rose-700"}`}>
                    {side} {quantity} shares of {ticker}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Execution Price:</span>
                  <span className="font-bold">{preview ? formatINR(preview.execution_price) : "-"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Gross Value:</span>
                  <span>{preview ? formatINR(preview.gross_amount) : "-"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Estimated Indian Friction (7.5 bps):</span>
                  <span className="text-amber-700">+{preview ? formatINR(preview.friction_amount) : "-"}</span>
                </div>
                <div className="flex justify-between border-t border-slate-200 pt-2 font-bold text-slate-900 text-sm">
                  <span>Total {side === "BUY" ? "Payment" : "Proceeds"}:</span>
                  <span className={side === "BUY" ? "text-emerald-700" : "text-rose-700"}>
                    {preview ? formatINR(preview.net_amount) : "-"}
                  </span>
                </div>
              </div>

              {errorMsg && (
                <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800">
                  {errorMsg}
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-2.5">
                <button
                  type="button"
                  disabled={isSubmitting}
                  onClick={() => setStep("CONFIG")}
                  className="flex-1 py-2 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 font-bold text-xs text-slate-700 transition-colors"
                >
                  ← Back
                </button>
                <button
                  type="button"
                  disabled={isSubmitting}
                  onClick={handleExecute}
                  className={`flex-2 py-2 rounded-lg font-bold text-xs uppercase tracking-wider text-white shadow-xs transition-colors flex items-center justify-center gap-1.5 ${
                    side === "BUY" ? "bg-emerald-600 hover:bg-emerald-700" : "bg-rose-600 hover:bg-rose-700"
                  }`}
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Executing...
                    </>
                  ) : (
                    `Confirm ${side} Order`
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
