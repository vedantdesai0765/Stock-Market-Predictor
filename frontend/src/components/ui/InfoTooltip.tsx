"use client";

import React, { useState, useRef, useEffect } from "react";
import { Info, HelpCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface InfoTooltipProps {
  term: string; // Plain-English term
  explanation: string; // Plain-English explanation
  technicalNote?: string; // Optional advanced/technical explanation
  icon?: "info" | "help";
  className?: string;
  side?: "top" | "bottom" | "left" | "right";
}

export function InfoTooltip({
  term,
  explanation,
  technicalNote,
  icon = "info",
  className,
  side = "top",
}: InfoTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative inline-flex items-center align-middle" ref={containerRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        aria-label={`Explanation for ${term}`}
        className={cn(
          "text-slate-400 hover:text-sky-600 focus:outline-hidden p-0.5 rounded transition-colors inline-flex",
          className
        )}
      >
        {icon === "help" ? (
          <HelpCircle className="w-3.5 h-3.5" />
        ) : (
          <Info className="w-3.5 h-3.5" />
        )}
      </button>

      {isOpen && (
        <div
          role="tooltip"
          className={cn(
            "absolute z-50 w-72 p-3 text-xs rounded-lg shadow-xl bg-white border border-slate-200 text-slate-800 pointer-events-none sm:pointer-events-auto animate-in fade-in-50 zoom-in-95 duration-100",
            side === "top" && "bottom-full left-1/2 -translate-x-1/2 mb-2",
            side === "bottom" && "top-full left-1/2 -translate-x-1/2 mt-2",
            side === "left" && "right-full top-1/2 -translate-y-1/2 mr-2",
            side === "right" && "left-full top-1/2 -translate-y-1/2 ml-2"
          )}
        >
          <div className="font-bold text-slate-900 border-b border-slate-100 pb-1 mb-1.5 flex items-center justify-between">
            <span>{term}</span>
            <span className="text-[10px] font-mono text-sky-600 uppercase tracking-wider">Guide</span>
          </div>

          <p className="text-slate-600 font-sans leading-relaxed text-[11px] mb-2">
            {explanation}
          </p>

          {technicalNote && (
            <div className="mt-1.5 pt-1.5 border-t border-slate-100 text-[10px] text-slate-500 font-sans bg-slate-50 -mx-3 -mb-3 p-2.5 rounded-b-lg">
              <span className="font-semibold text-slate-700 block mb-0.5 font-mono uppercase text-[9px]">
                Technical Definition:
              </span>
              {technicalNote}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
