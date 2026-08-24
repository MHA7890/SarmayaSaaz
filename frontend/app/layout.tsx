import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/Providers";
import { SyncGuard } from "@/components/SyncGuard";
import { Navbar } from "@/components/nav/Navbar";
import { TickerTape } from "@/components/nav/TickerTape";

export const metadata: Metadata = {
  title: "SarmayaSaaz: AI Financial Forecasting",
  description:
    "Multi horizon AI forecasts across cryptocurrencies, commodities, PSX equities and MUFAP mutual funds.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
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

