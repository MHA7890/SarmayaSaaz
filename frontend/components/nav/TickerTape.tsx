"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";

import { api } from "@/lib/api/client";
import {
  cn,
  directionClass,
  directionGlyph,
  formatPercent,
  formatPrice,
  formatTicker,
} from "@/lib/format";

/**
 * Marquee of current quotes. The list is rendered twice and the track
 * translates -50%, which loops seamlessly without a JS scroll handler.
 * Hovering pauses it; prefers-reduced-motion disables it entirely.
 */
export function TickerTape() {
  const { data, isError } = useQuery({
    queryKey: ["market", "ticker"],
    queryFn: () => api.market(),
    staleTime: 60_000,
  });

  const quotes = (data?.quotes ?? []).filter((q) => q.change_24h_pct !== null).slice(0, 28);

  if (isError || quotes.length === 0) return null;

  const row = quotes.map((q) => (
    <Link
      key={`${q.asset_class}:${q.ticker}`}
      href={`/forecasts?ticker=${encodeURIComponent(q.ticker)}&class=${q.asset_class}`}
      className="group inline-flex items-baseline gap-1.5 px-4 py-1.5"
    >
      <span className="num text-[11px] font-semibold text-muted group-hover:text-text">
        {formatTicker(q.ticker)}
      </span>
      <span className="num text-[11px] text-dim">
        {formatPrice(q.current_price, q.currency, q.asset_class)}
      </span>
      <span className={cn("num text-[11px] font-medium", directionClass(q.change_24h_pct))}>
        {directionGlyph(q.change_24h_pct)} {formatPercent(q.change_24h_pct)}
      </span>
    </Link>
  ));

  return (
    <div className="ticker-mask overflow-hidden border-b border-line bg-surface-inset">
      <div className="ticker-track animate-ticker" aria-hidden="true">
        {row}
        {row}
      </div>
      <span className="sr-only">
        Live quote ticker showing {quotes.length} assets. Use the Markets page for a
        readable table.
      </span>
    </div>
  );
}
