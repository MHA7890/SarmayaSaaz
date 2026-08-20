"use client";

import { AlertTriangle, Info } from "lucide-react";

import type { Forecast, HorizonForecast } from "@/lib/api/types";
import { NotMeasured } from "@/components/ui/States";
import {
  actionClass,
  cn,
  directionClass,
  directionGlyph,
  formatGroup,
  formatPercent,
  formatPrice,
  formatTicker,
  relativeDate,
} from "@/lib/format";

/** Headline price, 24h move, action and data freshness. */
export function AssetHeader({ forecast }: { forecast: Forecast }) {
  return (
    <div className="card flex flex-wrap items-center justify-between gap-4 p-5">
      <div>
        <div className="flex flex-wrap items-baseline gap-2.5">
          <h1 className="text-xl font-bold">{forecast.name}</h1>
          {forecast.asset_class !== "commodity" && (
            <span className="num rounded border border-line px-1.5 py-0.5 text-[11px] text-dim">
              {formatTicker(forecast.ticker)}
            </span>
          )}
          <span className="label">{formatGroup(forecast.group)}</span>
        </div>
        <div className="mt-2 flex flex-wrap items-baseline gap-3">
          <span className="num text-3xl font-bold">
            {formatPrice(forecast.current_price, forecast.currency, forecast.asset_class)}
          </span>
          {forecast.change_24h_pct !== null && (
            <span
              className={cn(
                "num text-sm font-semibold",
                directionClass(forecast.change_24h_pct),
              )}
            >
              {directionGlyph(forecast.change_24h_pct)}{" "}
              {formatPercent(forecast.change_24h_pct)}
            </span>
          )}
          <span className="text-xs text-dim">{forecast.unit}</span>
        </div>
      </div>

      <div className="flex flex-col items-end gap-2">
        <div className="flex items-center gap-2">
          {forecast.price_source === "live" && (
            <span className="rounded-full border border-pos/40 bg-pos/10 px-2 py-0.5 text-[10px] font-bold tracking-wide text-pos">
              LIVE
            </span>
          )}
          <span
            className={cn(
              "rounded-full border px-3 py-1 text-xs font-bold tracking-wide",
              actionClass(forecast.action),
            )}
          >
            {forecast.action}
          </span>
        </div>
        <span className="text-[11px] text-dim">
          Data as of {forecast.as_of} ({relativeDate(forecast.as_of)})
        </span>
        {forecast.source_url && (
          <a
            href={forecast.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-dim underline decoration-dotted underline-offset-2 hover:text-fg"
          >
            Source: TradingView ↗
          </a>
        )}
      </div>
    </div>
  );
}

/** Full multi-horizon table with real bounds, or an explicit gap where none exist. */
export function HorizonMatrix({ forecast }: { forecast: Forecast }) {
  return (
    <section className="card overflow-hidden" aria-label="Multi horizon target matrix">
      <header className="border-b border-line px-5 py-4">
        <h2 className="text-sm font-semibold">Multi Horizon Target Matrix</h2>
        <p className="mt-0.5 text-xs text-dim">
          Each horizon is a separately trained model, not an extrapolation of the others.
        </p>
      </header>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-b border-line text-left">
              {["Horizon", "Expected", "Return", "Lower", "Upper", "Confidence", "Model"].map(
                (h) => (
                  <th key={h} scope="col" className="label px-5 py-2.5 font-medium">
                    {h}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {forecast.horizons.map((h) => (
              <tr key={h.horizon_days} className="border-b border-line/60 last:border-0">
                <th scope="row" className="num px-5 py-3 text-left font-semibold text-accent">
                  {h.horizon_days}D
                </th>
                <td className="num px-5 py-3 font-medium">
                  {formatPrice(h.projected_price, forecast.currency, forecast.asset_class)}
                </td>
                <td
                  className={cn(
                    "num px-5 py-3 font-semibold",
                    directionClass(h.predicted_return_pct),
                  )}
                >
                  {directionGlyph(h.predicted_return_pct)}{" "}
                  {formatPercent(h.predicted_return_pct)}
                </td>
                <td className="num px-5 py-3 text-dim">
                  {h.lower_bound !== null ? (
                    formatPrice(h.lower_bound, forecast.currency, forecast.asset_class)
                  ) : (
                    <NotMeasured label="—" />
                  )}
                </td>
                <td className="num px-5 py-3 text-dim">
                  {h.upper_bound !== null ? (
                    formatPrice(h.upper_bound, forecast.currency, forecast.asset_class)
                  ) : (
                    <NotMeasured label="—" />
                  )}
                </td>
                <td className="px-5 py-3">
                  {h.confidence_pct !== null ? (
                    <ConfidenceBar value={h.confidence_pct} />
                  ) : (
                    <NotMeasured />
                  )}
                </td>
                <td className="px-5 py-3">
                  <span className="num rounded border border-line px-1.5 py-0.5 text-[11px]">
                    {h.primary_model}
                  </span>
                  {h.contributions.length > 1 && (
                    <span className="ml-1 text-[11px] text-dim">
                      +{h.contributions.length - 1}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <span className="flex items-center gap-2">
      <span className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-high">
        <span className="block h-full rounded-full bg-accent" style={{ width: `${pct}%` }} />
      </span>
      <span className="num text-[11px] text-dim">{pct.toFixed(2)}%</span>
    </span>
  );
}

/** The ensemble behind one horizon, so a number is never unattributed. */
export function EnsembleBreakdown({
  horizon,
  currency,
  assetClass,
}: {
  horizon: HorizonForecast;
  currency: string;
  assetClass?: string;
}) {
  return (
    <section className="card p-5" aria-label="Ensemble composition">
      <header className="mb-3">
        <h2 className="text-sm font-semibold">
          Ensemble at {horizon.horizon_days}D
        </h2>
        <p className="mt-0.5 text-xs text-dim">
          {horizon.contributions.length} model
          {horizon.contributions.length === 1 ? "" : "s"} voted, weighted by held out
          performance.
        </p>
      </header>

      <ul className="space-y-2">
        {[...horizon.contributions]
          .sort((a, b) => b.predicted_return_pct - a.predicted_return_pct)
          .map((c) => {
            const maxAbs = Math.max(
              ...horizon.contributions.map((x) => Math.abs(x.predicted_return_pct)),
              1e-9,
            );
            return (
              <li key={c.model_name} className="flex items-center gap-3">
                <span className="num w-28 shrink-0 truncate text-xs">{c.model_name}</span>
                <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-high">
                  <span
                    className="block h-full rounded-full bg-accent"
                    style={{
                      width: `${Math.min(100, (Math.abs(c.predicted_return_pct) / maxAbs) * 100)}%`,
                    }}
                  />
                </span>
                <span
                  className={cn(
                    "num w-16 shrink-0 text-right text-[11px] font-medium",
                    directionClass(c.predicted_return_pct),
                  )}
                >
                  {formatPercent(c.predicted_return_pct)}
                </span>
              </li>
            );
          })}
      </ul>

      <p className="mt-3 flex items-start gap-1.5 text-[11px] text-dim">
        <Info size={12} className="mt-0.5 shrink-0" />
        Consensus {formatPercent(horizon.predicted_return_pct)} →{" "}
        {formatPrice(horizon.projected_price, currency, assetClass)}
      </p>
    </section>
  );
}

/** Degradations the engine reported, surfaced instead of hidden. */
export function WarningsPanel({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <details className="card p-4">
      <summary className="flex cursor-pointer items-center gap-2 text-xs font-medium text-warn">
        <AlertTriangle size={14} />
        {warnings.length} model{warnings.length === 1 ? "" : "s"} unavailable for this
        asset
      </summary>
      <ul className="mt-2.5 space-y-1">
        {warnings.map((w, i) => (
          <li key={i} className="text-[11px] leading-relaxed text-dim">
            {w}
          </li>
        ))}
      </ul>
    </details>
  );
}
