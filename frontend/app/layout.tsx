import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/Providers";
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
          <footer className="mx-auto w-full max-w-[1400px] px-4 pb-10 pt-4 sm:px-6">
            <p className="text-xs leading-relaxed text-dim">
              Forecasts are model output, not investment advice. Figures derive from
              historical data and may be stale; check the &ldquo;as of&rdquo; date on each
              asset before acting on anything shown here.
            </p>
          </footer>
        </Providers>
      </body>
    </html>
  );
}
