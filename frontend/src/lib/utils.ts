import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatINR(val: number, decimals: number = 2): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(val);
}

export function formatNumber(val: number, decimals: number = 2): string {
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(val);
}

export function formatPercent(val: number, decimals: number = 2): string {
  return `${(val * 100).toFixed(decimals)}%`;
}

export function formatBps(val: number): string {
  return `${(val * 10000).toFixed(1)} bps`;
}

/** "2026-11-16" -> "Monday, 16 November 2026" (dates from the API are plain calendar dates in India). */
export function formatDateLong(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(`${iso.slice(0, 10)}T00:00:00Z`);
  return d.toLocaleDateString("en-GB", {
    weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC",
  });
}

/** "2026-11-16" -> "Mon 16 Nov" */
export function formatDateShort(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(`${iso.slice(0, 10)}T00:00:00Z`);
  return d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" });
}

/** ISO timestamp -> "7:52 am" in Indian time */
export function formatTimeIST(iso?: string | null): string {
  if (!iso) return "";
  return new Date(iso)
    .toLocaleTimeString("en-IN", { hour: "numeric", minute: "2-digit", hour12: true, timeZone: "Asia/Kolkata" })
    .toLowerCase();
}

/** 0.706 -> "70.6%" */
export function formatShare(val?: number | null, decimals: number = 1): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "–";
  return `${(val * 100).toFixed(decimals)}%`;
}

/** 0.91 -> "+0.91%" */
export function formatSignedPct(val?: number | null, decimals: number = 2): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "–";
  return `${val > 0 ? "+" : ""}${val.toFixed(decimals)}%`;
}
