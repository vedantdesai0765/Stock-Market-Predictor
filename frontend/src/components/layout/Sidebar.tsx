"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Activity,
  SunMedium,
  Compass,
  Zap,
  Briefcase,
  BookOpen,
  Settings,
  ExternalLink,
  ShieldCheck,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useHealth } from "@/hooks/useStockData";
import { Logo } from "@/components/brand/Logo";

const NAV_ITEMS = [
  { label: "Overview", href: "/", icon: LayoutDashboard },
  { label: "Prediction Terminal", href: "/prediction", icon: Activity },
  { label: "Next-Morning Forecast", href: "/forecast", icon: SunMedium },
  { label: "Why Did It Move?", href: "/explain", icon: Compass },
  { label: "Actionable Signals", href: "/signals", icon: Zap },
  { label: "Virtual Portfolio", href: "/portfolio", icon: Briefcase },
  { label: "References", href: "/references", icon: BookOpen },
  { label: "Settings", href: "/settings", icon: Settings },
];

interface SidebarProps {
  className?: string;
  onItemClick?: () => void;
}

export function Sidebar({ className, onItemClick }: SidebarProps) {
  const pathname = usePathname();
  const { data: health, isSuccess } = useHealth();

  return (
    <aside className={cn("w-64 bg-[#0B132B] text-slate-300 flex flex-col justify-between shrink-0 border-r border-[#1E293B] select-none min-h-screen", className)}>
      <div>
        {/* Brand Header */}
        <div className="h-16 flex items-center px-4 border-b border-[#1E293B]">
          <Logo variant="full" theme="dark" size="md" />
        </div>

        {/* Section Title */}
        <div className="px-5 pt-5 pb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500 font-mono">
          Platform Navigation
        </div>

        {/* Nav Links */}
        <nav className="px-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive =
              item.href === "/"
                ? pathname === "/"
                : pathname?.startsWith(item.href);
            const Icon = item.icon;

            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onItemClick}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-md text-xs font-medium transition-all group",
                  isActive
                    ? "bg-sky-600/20 text-white border-l-2 border-sky-400 font-semibold"
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                )}
              >
                <Icon
                  className={cn(
                    "w-4 h-4 transition-colors",
                    isActive ? "text-sky-400" : "text-slate-400 group-hover:text-slate-200"
                  )}
                />
                <span className="truncate">{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer System Status */}
      <div className="p-4 border-t border-[#1E293B] bg-[#070D1F]/50">
        <div className="flex items-center justify-between text-[11px] mb-2 font-mono">
          <span className="text-slate-400">FastAPI Backend</span>
          <span className="flex items-center gap-1.5">
            <span
              className={cn(
                "w-2 h-2 rounded-full",
                isSuccess ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              )}
            />
            <span className={isSuccess ? "text-emerald-400" : "text-amber-400"}>
              {isSuccess ? "Connected" : "Offline"}
            </span>
          </span>
        </div>

        <div className="rounded bg-[#162347]/60 border border-[#1E293B] p-2.5 text-[11px] text-slate-400 font-sans">
          <div className="flex items-center justify-between">
            <span className="font-medium text-slate-300">Legacy Streamlit</span>
            <a
              href="http://localhost:8501"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sky-400 hover:text-sky-300 inline-flex items-center gap-1 text-[10px]"
            >
              Port 8501 <ExternalLink className="w-2.5 h-2.5" />
            </a>
          </div>
          <p className="mt-1 text-[10px] text-slate-400 leading-snug">
            Research terminal remains fully operational alongside Next.js.
          </p>
        </div>
      </div>
    </aside>
  );
}
