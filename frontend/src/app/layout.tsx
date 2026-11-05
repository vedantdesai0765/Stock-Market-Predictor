import type { Metadata } from "next";
import "./globals.css";
import { AppProviders } from "@/components/providers/QueryProvider";
import { Shell } from "@/components/layout/Shell";

export const metadata: Metadata = {
  title: "Stock Market Predictor | Institutional Financial Analytics",
  description:
    "Production-style Financial Analytics Terminal for Indian Equities. Directional predictions, walk-forward confidence gating, next-morning forecasting, and evidence-based trading signals.",
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
