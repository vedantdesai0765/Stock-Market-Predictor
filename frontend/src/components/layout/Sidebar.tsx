"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Activity,
  LineChart,
  Sunrise,
  Compass,
  Zap,
  Briefcase,
  BookOpen,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useHealth } from "@/hooks/useStockData";
import { Logo } from "@/components/brand/Logo";

// Three groups: today's live forecast, the research behind it, and the paper-trading simulator.
const NAV_GROUPS = [
  {
    title: "Today",
    items: [{ label: "Morning Call", href: "/", icon: Sunrise }],
  },
  {
    title: "Research",
    items: [
      { label: "Project Overview", href: "/overview", icon: LayoutDashboard },
      { label: "Forecast Track Record", href: "/forecast", icon: LineChart },
      { label: "3-Day Prediction", href: "/prediction", icon: Activity },
      { label: "Why Did It Move?", href: "/explain", icon: Compass },
      { label: "Methodology & References", href: "/references", icon: BookOpen },
    ],
  },
  {
    title: "Simulator",
    items: [
      { label: "Strategy Signals", href: "/signals", icon: Zap },
      { label: "Virtual Portfolio", href: "/portfolio", icon: Briefcase },
    ],
  },
  {
    title: "System",
    items: [{ label: "Settings", href: "/settings", icon: Settings }],
  },
];

interface SidebarProps {
  className?: string;
  onItemClick?: () => void;
}

export function Sidebar({ className, onItemClick }: SidebarProps) {
  const pathname = usePathname();
  const { data: health, isSuccess } = useHealth();

  return (
    <aside className={cn("w-64 bg-[#0B132B] text-slate-300 flex flex-col justify-between shrink-0 border-r border-[#1E293B] select-none min-h-screen overflow-y-auto", className)}>
      <div>
        {/* Brand Header */}
        <div className="h-16 flex items-center px-4 border-b border-[#1E293B]">
          <Logo variant="full" theme="dark" size="md" />
        </div>

        {/* Grouped navigation */}
        <nav className="px-3 pt-2 pb-4">
          {NAV_GROUPS.map((group) => (
            <div key={group.title}>
              <div className="px-2 pt-4 pb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500 font-mono">
                {group.title}
              </div>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const isActive =
                    item.href === "/"
                      ? pathname === "/"
                      : pathname?.startsWith(item.href) ||
                        (item.href === "/references" && pathname?.startsWith("/methodology"));
                  const Icon = item.icon;

                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={onItemClick}
                      aria-current={isActive ? "page" : undefined}
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
              </div>
            </div>
          ))}
        </nav>
      </div>

      {/* Footer: data service status and what this site is */}
      <div className="p-4 border-t border-[#1E293B] bg-[#070D1F]/50">
        <div className="flex items-center justify-between text-[11px] mb-2 font-mono">
          <span className="text-slate-400">Data service</span>
          <span className="flex items-center gap-1.5">
            <span
              className={cn(
                "w-2 h-2 rounded-full",
                isSuccess ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              )}
            />
            <span className={isSuccess ? "text-emerald-400" : "text-amber-400"}>
              {isSuccess ? "Connected" : "Connecting"}
            </span>
          </span>
        </div>
        <p className="text-[10px] text-slate-400 leading-snug font-sans">
          Final-year research project, Group C23, K J Somaiya School of Engineering. For education only. Not
          investment advice.
        </p>
      </div>
    </aside>
  );
}
