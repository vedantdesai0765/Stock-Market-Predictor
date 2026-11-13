import type { Metadata, Viewport } from "next";
import "./globals.css";
import { AppProviders } from "@/components/providers/QueryProvider";
import { Shell } from "@/components/layout/Shell";

export const metadata: Metadata = {
  title: {
    default: "Stock Market Predictor | Morning Call",
    template: "%s | Stock Market Predictor",
  },
  description:
    "A daily forecast of whether 10 NSE stocks will open higher or lower, made before the market opens from overnight US and Asian markets, with a public track record. Final-year research project. Not investment advice.",
  applicationName: "Stock Market Predictor",
  appleWebApp: { capable: true, title: "Morning Call", statusBarStyle: "black-translucent" },
  openGraph: {
    title: "Stock Market Predictor: Morning Call",
    description:
      "Will each stock open higher or lower today? A forecast published every trading morning before 9:15 am IST, and graded against what really happened.",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#0B132B",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen">
        <AppProviders>
          <Shell>{children}</Shell>
        </AppProviders>
      </body>
    </html>
  );
}
