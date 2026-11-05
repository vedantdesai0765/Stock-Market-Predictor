"use client";

import React, { useState } from "react";
import { AlertTriangle, RotateCcw, X, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface ResetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirmReset: () => Promise<void>;
  isResetting: boolean;
}

export function ResetModal({
  isOpen,
  onClose,
  onConfirmReset,
  isResetting,
}: ResetModalProps) {
  const [confirmationInput, setConfirmationInput] = useState("");
  const [hasError, setHasError] = useState(false);

  if (!isOpen) return null;

  const handleConfirm = async () => {
    if (confirmationInput.trim().toUpperCase() !== "RESET") {
      setHasError(true);
      return;
    }
    setHasError(false);
    await onConfirmReset();
    setConfirmationInput("");
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
      <div className="relative w-full max-w-md bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100 bg-amber-50/50">
          <div className="flex items-center gap-2 text-amber-800">
            <AlertTriangle className="w-5 h-5 text-amber-600" />
            <h3 className="font-bold text-sm font-sans">Reset Virtual Paper Wallet</h3>
          </div>
          <button
            onClick={onClose}
            disabled={isResetting}
            className="p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 space-y-4">
          <p className="text-xs text-slate-600 leading-relaxed font-sans">
            You are about to reset your simulated trading account. This operation is <strong>atomic</strong> and strictly resets the following virtual state:
          </p>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-2 text-xs text-slate-700 font-sans">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>Available Cash restored to <strong>₹1,00,000.00</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>All simulated active stock holdings cleared (0 shares)</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>Audit transaction history ledger cleared</span>
            </div>
          </div>

          <div className="p-3 bg-blue-50/70 border border-blue-200/70 rounded-lg text-[11px] text-blue-900">
            <strong>Safe Isolation Guarantee:</strong> Resetting the virtual wallet only touches the paper trading database table (<code className="font-mono">wallets</code>, <code className="font-mono">holdings</code>, <code className="font-mono">transactions</code>). It never alters historical OHLCV data, trained model artifacts, or research backtests.
          </div>

          <div className="space-y-1.5">
            <label className="block text-[11px] font-semibold text-slate-700 font-mono uppercase tracking-wider">
              Type &quot;RESET&quot; to confirm:
            </label>
            <input
              type="text"
              placeholder="RESET"
              value={confirmationInput}
              onChange={(e) => {
                setConfirmationInput(e.target.value);
                if (hasError) setHasError(false);
              }}
              className={cn(
                "w-full px-3 py-2 text-xs font-mono uppercase rounded-md border bg-slate-50 focus:outline-hidden focus:ring-1",
                hasError
                  ? "border-rose-400 focus:ring-rose-500"
                  : "border-slate-200 focus:ring-sky-500"
              )}
            />
            {hasError && (
              <p className="text-[11px] text-rose-600 font-medium">
                Please type &quot;RESET&quot; in capital letters to confirm.
              </p>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-5 py-3.5 bg-slate-50 border-t border-slate-100">
          <button
            type="button"
            onClick={onClose}
            disabled={isResetting}
            className="px-3 py-1.5 rounded-md text-xs font-semibold text-slate-600 hover:text-slate-800 hover:bg-slate-200 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            disabled={isResetting || confirmationInput.trim().toUpperCase() !== "RESET"}
            className={cn(
              "inline-flex items-center gap-1.5 px-4 py-1.5 rounded-md text-xs font-semibold transition-all",
              confirmationInput.trim().toUpperCase() === "RESET" && !isResetting
                ? "bg-rose-600 hover:bg-rose-700 text-white shadow-xs"
                : "bg-slate-200 text-slate-400 cursor-not-allowed"
            )}
          >
            <RotateCcw className={cn("w-3.5 h-3.5", isResetting && "animate-spin")} />
            {isResetting ? "Resetting Wallet..." : "Confirm Atomic Reset"}
          </button>
        </div>
      </div>
    </div>
  );
}
