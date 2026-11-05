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
