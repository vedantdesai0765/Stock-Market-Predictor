import React from "react";
import Link from "next/link";
import { FlaskConical } from "lucide-react";

/**
 * Shown at the top of the Strategy Signals and Virtual Portfolio pages so nobody mistakes a
 * backtest simulation for a forecast.
 */
export function SimulatorNotice({ children }: { children?: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-900">
      <FlaskConical className="w-4 h-4 mt-0.5 shrink-0 text-amber-600" />
      <div className="leading-relaxed">
        <span className="font-semibold">Strategy simulator, not a forecast.</span>{" "}
        {children ?? (
          <>
            The BUY, SELL and HOLD labels here replay trading rules built on the 3-day model, which did not beat a
            simple guess for any of the 10 stocks in testing. Use this page to see how rules, costs and risk limits
            behave, not to decide trades. Backtest returns credit 5% yearly interest on idle cash. Averaged over the 10
            stocks, no fixed rule earned more than simply buying and holding.
          </>
        )}{" "}
        The project&rsquo;s tested forecast is the{" "}
        <Link href="/" className="font-semibold underline underline-offset-2 hover:text-amber-950">
          Morning Call
        </Link>
        .
      </div>
    </div>
  );
}
