import React from "react";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface ModuleCardProps {
  title: string;
  description: string;
  href: string;
  tag: string;
  icon: React.ReactNode;
  statusText?: string;
  badge?: React.ReactNode;
  className?: string;
}

export function ModuleCard({
  title,
  description,
  href,
  tag,
  icon,
  statusText,
  badge,
  className,
}: ModuleCardProps) {
  return (
    <Link
      href={href}
      className={cn(
        "group relative flex flex-col justify-between p-5 rounded-lg border border-slate-200 bg-white hover:border-sky-300 hover:shadow-md transition-all",
        className
      )}
    >
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-md bg-slate-100 flex items-center justify-center text-slate-700 group-hover:bg-sky-50 group-hover:text-sky-600 transition-colors">
              {icon}
            </div>
            <span className="text-[11px] font-mono uppercase tracking-wider font-semibold text-slate-500">
              {tag}
            </span>
          </div>
          <ArrowUpRight className="w-4 h-4 text-slate-400 group-hover:text-sky-600 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
        </div>

        <h3 className="text-base font-bold text-slate-900 group-hover:text-sky-700 transition-colors">
          {title}
        </h3>
        <p className="mt-1 text-xs text-slate-600 leading-relaxed font-sans">
          {description}
        </p>
      </div>

      <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
        {statusText && (
          <span className="text-slate-600 font-mono text-[11px]">
            {statusText}
          </span>
        )}
        {badge}
      </div>
    </Link>
  );
}
