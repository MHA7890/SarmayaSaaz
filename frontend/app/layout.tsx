import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/Providers";
import { SyncGuard } from "@/components/SyncGuard";
import { Navbar } from "@/components/nav/Navbar";
import { TickerTape } from "@/components/nav/TickerTape";

export const metadata: Metadata = {
  metadataBase: new URL("https://sarmayasaaz.tech"),
  title: "SarmayaSaaz — AI-Powered Financial Forecasting",
  description:
    "SarmayaSaaz is an AI-powered financial forecasting platform for cryptocurrencies, PSX stocks, commodities, and mutual funds.",
  alternates: {
    canonical: "/",
  },
  openGraph: {
    title: "SarmayaSaaz — AI-Powered Financial Forecasting",
    description:
      "SarmayaSaaz is an AI-powered financial forecasting platform for cryptocurrencies, PSX stocks, commodities, and mutual funds.",
    url: "https://sarmayasaaz.tech",
    siteName: "SarmayaSaaz",
    locale: "en_US",
    type: "website",
    images: [
      {
        url: "/icon.png",
        width: 512,
        height: 512,
        alt: "SarmayaSaaz Logo",
      },
    ],
  },
  twitter: {
    card: "summary",
    title: "SarmayaSaaz — AI-Powered Financial Forecasting",
    description:
      "SarmayaSaaz is an AI-powered financial forecasting platform for cryptocurrencies, PSX stocks, commodities, and mutual funds.",
    images: ["/icon.png"],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  icons: {
    icon: [
      { url: "/favicon-32x32.png?v=3", sizes: "32x32", type: "image/png" },
      { url: "/favicon-16x16.png?v=3", sizes: "16x16", type: "image/png" },
      { url: "/favicon.ico?v=3", sizes: "any" },
    ],
    shortcut: "/favicon.ico?v=3",
    apple: "/apple-icon.png?v=3",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png?v=3" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png?v=3" />
        <link rel="shortcut icon" href="/favicon.ico?v=3" />
        <link rel="apple-touch-icon" sizes="180x180" href="/apple-icon.png?v=3" />
        {/* Applied before paint so the stored theme never flashes the wrong palette. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('theme');var d=t?t==='dark':true;document.documentElement.classList.toggle('dark',d);}catch(e){}})();`,
          }}
        />
      </head>
      <body>
        <Providers>
          <SyncGuard>
            <a
              href="#main"
              className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-accent focus:px-4 focus:py-2 focus:text-white"
            >
              Skip to content
            </a>
            <Navbar />
            <TickerTape />
            <main id="main" className="mx-auto w-full max-w-[1400px] px-4 py-6 sm:px-6">
              {children}
            </main>
            <footer className="mx-auto w-full max-w-[1400px] px-4 pb-12 pt-8 sm:px-6">
              <div className="rounded-xl border border-border/50 bg-card/40 p-4 sm:p-5 backdrop-blur-sm">
                <p className="text-xs leading-relaxed text-dim">
                  SarmayaSaaz provides AI-generated forecasts for informational purposes only.{" "}
                  <strong className="font-semibold text-neutral-200">
                    These forecasts are not financial, investment, or trading advice, and should not be relied upon as a recommendation to buy, sell, or hold any asset.
                  </strong>{" "}
                  Data and forecasts are based on available historical and market information, which may be delayed, incomplete, or not up to date at all times. Always verify current market data and conduct your own research before making any investment decisions.
                </p>
              </div>
            </footer>
          </SyncGuard>
        </Providers>
      </body>
    </html>
  );
}

