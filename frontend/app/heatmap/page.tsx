"use client";

import { useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";
import { useState } from "react";

import { ErrorState, NotMeasured, Skeleton } from "@/components/ui/States";
import { api } from "@/lib/api/client";
import {
  ASSET_CLASS_LABEL,
  ASSET_CLASS_SHORT,
  HORIZONS,
  type AssetClass,
  type LeaderboardEntry,
} from "@/lib/api/types";
import { cn, formatTicker, isInternalId } from "@/lib/format";

const CLASSES: AssetClass[] = ["commodity", "crypto", "stock", "mutual_fund"];

/**
 * Sequential magnitude ramp, low -> high. The colours live in globals.css as
 * --heat-N / --heat-N-fg so light and dark can run genuinely different ramps:
 * on dark, magnitude reads as brighter and ends at yellow; on white a yellow
 * maximum is nearly invisible, so light runs the same green family the other
 * way and ends deep. Magnitude only, never identity.
 */
const RAMP_STEPS = 6;

/**
 * Domain floor and ceiling, deliberately not 0 and 100.
 *
 * Scores across every class and horizon span 38.25 to 93.07, so a 0-100 domain
 * left the bottom two of six steps permanently unreachable and squeezed the
 * commodity board into three shades - 35 of its 54 cells landed on the same
 * one, which is why a 38 and a 72 looked alike. Over 45-85 all six steps are
 * reachable and that board separates into four.
 *
 * The floor is low enough that the first step covers roughly the coin-flip
 * band, so a directional model at or below chance stays colourless slate
 * rather than being shaded as if it had a magnitude worth reading.
 */
const RAMP_MIN = 45;
const RAMP_MAX = 85;

function rampIndex(value: number): number {
  const t = (value - RAMP_MIN) / (RAMP_MAX - RAMP_MIN);
  return Math.min(RAMP_STEPS - 1, Math.max(0, Math.floor(t * RAMP_STEPS)));
}

function rampFor(value: number | null): string {
  if (value === null) return "transparent";
  return `var(--heat-${rampIndex(value)})`;
}

function rampTextFor(value: number | null): string | undefined {
  if (value === null) return undefined;
  return `var(--heat-${rampIndex(value)}-fg)`;
}

/** One score per row/model cell, normalized across the 3 metric shapes the
 * backend actually records (see backend/routers/models.py's module docstring):
 * commodities have real directional accuracy, crypto has a per-cluster win
 * rate, stock/mutual-fund have only the winning model's MAE. */
function cellValue(score: LeaderboardEntry["scores"][number] | undefined) {
  if (!score) return { display: "—", ramp: null as number | null, title: "not trained for this cell" };
  if (score.directional_accuracy !== null) {
    const pct = score.directional_accuracy;
    return { display: pct.toFixed(2), ramp: pct, title: `${pct.toFixed(2)}% directional accuracy` };
  }
  if (score.win_rate !== null) {
    const pct = score.win_rate * 100;
    return {
      display: pct.toFixed(2),
      ramp: pct,
      title: `${pct.toFixed(2)}% win rate across this cluster (not per asset accuracy)`,
    };
  }
  if (score.mae !== null) {
    return { display: score.mae.toFixed(4), ramp: null, title: `MAE ${score.mae.toFixed(4)} (only model recorded for this group)` };
  }
  return { display: "—", ramp: null, title: "no score recorded" };
}

export default function HeatmapPage() {
  const [horizon, setHorizon] = useState(60);
  const [view, setView] = useState<AssetClass | "overall">("overall");

  const commodityQ = useQuery({
    queryKey: ["leaderboard", "commodity", horizon],
    queryFn: () => api.leaderboard({ horizon_days: horizon, asset_class: "commodity" }),
  });
  const cryptoQ = useQuery({
    queryKey: ["leaderboard", "crypto", horizon],
    queryFn: () => api.leaderboard({ horizon_days: horizon, asset_class: "crypto" }),
  });
  const stockQ = useQuery({
    queryKey: ["leaderboard", "stock", horizon],
    queryFn: () => api.leaderboard({ horizon_days: horizon, asset_class: "stock" }),
  });
  const mutualFundQ = useQuery({
    queryKey: ["leaderboard", "mutual_fund", horizon],
    queryFn: () => api.leaderboard({ horizon_days: horizon, asset_class: "mutual_fund" }),
  });
  const byClass = {
    commodity: commodityQ,
    crypto: cryptoQ,
    stock: stockQ,
    mutual_fund: mutualFundQ,
  } as const;

  const klass = view === "overall" ? "commodity" : view;
  const { data, isLoading, error } = byClass[klass];
  const perAsset = klass === "commodity";

  const overallEntries = CLASSES.flatMap((c) =>
    (byClass[c].data?.entries ?? []).map((e) => ({ ...e, _class: c })),
  );
  const overallLoading = CLASSES.some((c) => byClass[c].isLoading);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-bold">Model Performance Heatmap</h1>
        <p className="mt-0.5 text-sm text-dim">
          Held out directional accuracy by model, measured during training.
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Asset class">
          <button
            onClick={() => setView("overall")}
            aria-pressed={view === "overall"}
            className={cn("pill", view === "overall" && "pill-active")}
          >
            Overall
          </button>
          {CLASSES.map((c) => (
            <button
              key={c}
              onClick={() => setView(c)}
              aria-pressed={view === c}
              className={cn("pill", view === c && "pill-active")}
            >
              {ASSET_CLASS_LABEL[c]}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5" role="group" aria-label="Horizon">
          <span className="label mr-1">Horizon</span>
          {HORIZONS.map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              aria-pressed={horizon === h}
              className={cn("pill", horizon === h && "pill-active")}
            >
              {h}D
            </button>
          ))}
        </div>
      </div>

      {view !== "overall" && !perAsset && (
        <p className="inset flex items-start gap-2 px-4 py-3 text-xs leading-relaxed text-muted">
          <Info size={14} className="mt-0.5 shrink-0 text-accent" />
          <span>
            Only the commodities engine recorded per model metrics at asset granularity.{" "}
            {view === "crypto"
              ? "Crypto recorded a win rate per cluster, shared by every asset in that cluster rather than measured per asset, shown below as a real percentage rather than a checkmark."
              : "This engine recorded only the winning model and its MAE per group. The other candidate models were never scored during training, so those cells stay empty rather than a fabricated number."}
          </span>
        </p>
      )}

      {view === "overall" ? (
        <>
          {overallLoading && <Skeleton className="h-80 w-full" />}
          {overallEntries.length > 0 && (
            <div className="card overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[640px] text-sm">
                  <caption className="sr-only">
                    Best model per group across all asset classes at the {horizon} day horizon
                  </caption>
                  <thead>
                    <tr className="border-b border-line">
                      <th scope="col" className="label px-4 py-2.5 text-left font-medium">
                        Group
                      </th>
                      <th scope="col" className="label px-4 py-2.5 text-left font-medium">
                        Class
                      </th>
                      <th scope="col" className="label px-4 py-2.5 text-left font-medium">
                        Best Model
                      </th>
                      <th scope="col" className="label px-4 py-2.5 text-right font-medium">
                        Score
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {overallEntries.map((e) => {
                      const bestScore = e.scores.find((s) => s.model_name === e.best_model);
                      const cell = cellValue(bestScore);
                      return (
                        <tr
                          key={`${e._class}:${e.ticker}`}
                          className="cursor-pointer border-b border-line/60 last:border-0 hover:bg-surface-high/50"
                          onClick={() => setView(e._class)}
                        >
                          <th scope="row" className="px-4 py-2.5 text-left">
                            <span className="num block text-[13px] font-semibold">{e.name}</span>
                          </th>
                          <td className="px-4 py-2.5">
                            <span className="label">{ASSET_CLASS_SHORT[e._class]}</span>
                          </td>
                          <td className="px-4 py-2.5">
                            {e.best_model ? (
                              <span className="num text-[11px] font-semibold text-accent">
                                {e.best_model}
                              </span>
                            ) : (
                              <NotMeasured />
                            )}
                          </td>
                          <td className="px-4 py-2.5 text-right">
                            <span
                              className="num inline-flex h-7 min-w-[56px] items-center justify-center rounded-md px-2 text-[11px] font-medium"
                              style={{ background: rampFor(cell.ramp), color: rampTextFor(cell.ramp) }}
                              title={cell.title}
                            >
                              {cell.display}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          {isLoading && <Skeleton className="h-80 w-full" />}
          {error && <ErrorState error={error} />}

          {data && data.entries.length > 0 && (
            <div className="card overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[760px] text-sm">
                  <caption className="sr-only">
                    Model performance by asset at the {horizon} day horizon
                  </caption>
                  <thead>
                    <tr className="border-b border-line">
                      <th scope="col" className="label px-4 py-2.5 text-left font-medium">
                        Asset
                      </th>
                      {data.models.map((m) => (
                        <th
                          key={m}
                          scope="col"
                          className="label px-2 py-2.5 text-center font-medium"
                        >
                          {m}
                        </th>
                      ))}
                      <th scope="col" className="label px-4 py-2.5 text-left font-medium">
                        Best
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.entries.map((e) => {
                      const byName = new Map(e.scores.map((s) => [s.model_name, s]));
                      return (
                        <tr key={e.ticker} className="border-b border-line/60 last:border-0">
                          <th scope="row" className="px-4 py-2 text-left">
                            <span className="block max-w-[200px] truncate text-[13px] font-semibold">
                              {e.name}
                            </span>
                            {!isInternalId(e.ticker) && e.ticker.toLowerCase() !== e.name.toLowerCase() && (
                              <span className="num block truncate text-[11px] text-dim">
                                {formatTicker(e.ticker)}
                              </span>
                            )}
                          </th>
                          {data.models.map((m) => {
                            const score = byName.get(m);
                            const cell = cellValue(score);
                            const isBest = e.best_model === m;
                            return (
                              <td key={m} className="p-1 text-center">
                                <span
                                  className={cn(
                                    "num flex h-9 items-center justify-center rounded-md text-[11px] font-medium",
                                    isBest && "ring-2 ring-accent",
                                    cell.ramp === null && "text-dim",
                                  )}
                                  style={{ background: rampFor(cell.ramp), color: rampTextFor(cell.ramp) }}
                                  title={score ? `${m}: ${cell.title}` : `${m}: not trained for this cell`}
                                >
                                  {cell.display}
                                </span>
                              </td>
                            );
                          })}
                          <td className="px-4 py-2">
                            {e.best_model ? (
                              <span className="num text-[11px] font-semibold text-accent">
                                {e.best_model}
                              </span>
                            ) : (
                              <NotMeasured />
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Sequential legend, magnitude only, not identity. */}
          {perAsset && (
            <div className="flex items-center gap-3 text-xs text-dim">
              <span>Directional accuracy</span>
              <span className="flex items-center gap-0.5">
                {Array.from({ length: RAMP_STEPS }, (_, i) => (
                  <span
                    key={i}
                    className="h-3.5 w-7 rounded-sm border border-line/50"
                    style={{ background: `var(--heat-${i})` }}
                  />
                ))}
              </span>
              <span className="num">45% → 85%+</span>
              <span className="ml-2 flex items-center gap-1.5">
                <span className="h-3.5 w-3.5 rounded-sm ring-2 ring-accent" />
                best per asset
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
