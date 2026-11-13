import type { MetadataRoute } from "next";

// Lets phones and desktops install the site like an app ("Add to Home Screen").
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Stock Market Predictor",
    short_name: "Morning Call",
    description:
      "A daily forecast of whether 10 NSE stocks will open higher or lower, with a public track record. Educational project, not investment advice.",
    start_url: "/",
    display: "standalone",
    background_color: "#F8FAFC",
    theme_color: "#0B132B",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
