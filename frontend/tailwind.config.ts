import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#F8FAFC",
        foreground: "#0F172A",
        card: {
          DEFAULT: "#FFFFFF",
          foreground: "#0F172A",
          border: "#E2E8F0",
        },
        navy: {
          900: "#0B132B",
          950: "#070D1F",
          sidebar: "#0B132B",
          sidebarHover: "#162347",
          sidebarActive: "#1C2D5A",
        },
        terminal: {
          border: "#E2E8F0",
          muted: "#64748B",
          accent: "#0284C7",
          teal: "#0D9488",
          up: "#16A34A",
          down: "#DC2626",
          warning: "#D97706",
          neutral: "#475569",
        },
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
      boxShadow: {
        card: "0 1px 3px 0 rgba(15, 23, 42, 0.05), 0 1px 2px -1px rgba(15, 23, 42, 0.05)",
        subtle: "0 1px 2px 0 rgba(0, 0, 0, 0.03)",
      },
    },
  },
  plugins: [],
};

export default config;
