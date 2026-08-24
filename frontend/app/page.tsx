"use client";

import { useQuery } from "@tanstack/react-query";
import { TrendingDown, TrendingUp } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { EmptyState, ErrorState, Skeleton } from "@/components/ui/States";
import { api } from "@/lib/api/client";
import {
  ASSET_CLASS_LABEL,
  ASSET_CLASS_SHORT,
  HORIZONS,
  type AssetClass,
  type Mover,
} from "@/lib/api/types";
import {
  cn,
  directionGlyph,
  formatGroup,
  formatInt,
  formatPercent,
  formatPrice,
  relativeDate,
} from "@/lib/format";

const CLASSES: (AssetClass | "all")[] = ["all", "crypto", "stock", "mutual_fund", "commodity"];

export default function DashboardPage() {
  const [horizon, setHorizon] = useState<number>(28);
  const [klass, setKlass] = useState<AssetClass | "all">("all");

  const { data: stats } = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const { data, isLoading, error } = useQuery({
    queryKey: ["movers", horizon, klass],
    queryFn: () =>
      api.movers({
        horizon_days: horizon,
        asset_class: klass === "all" ? undefined : klass,
        limit: 10,
      }),
  });

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-bold">Dashboard</h1>
        <p className="mt-0.5 text-sm text-dim">
          Largest predicted moves at the selected horizon. Model output, not realised
          price movement.
          {data?.generated_at && ` Snapshot ${relativeDate(data.generated_at)}.`}
        </p>
        {/* Carried over from the forecast dashboard this page replaced - it was
            the only place the system's own state was visible. */}
        <p className="mt-0.5 text-xs text-dim">
          {stats
            ? `${stats.total_assets} assets · ${stats.engines_online}/${stats.engines_total} engines online · ${formatInt(stats.artifacts_on_disk)} trained artifacts`
            : "Loading system status…"}
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by asset class">
          {CLASSES.map((c) => (
            <button
              key={c}
              onClick={() => setKlass(c)}
              aria-pressed={klass === c}
              className={cn("pill", klass === c && "pill-active")}
            >
              {c === "all" ? "All Assets" : ASSET_CLASS_LABEL[c]}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5" role="group" aria-label="Select horizon">
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

      {isLoading && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Skeleton className="h-[460px]" />
          <Skeleton className="h-[460px]" />
        </div>
      )}
      {error && (
        <ErrorState
          error={error}
          hint="The dashboard reads a precomputed snapshot. Build one with: uv run python scripts/build_snapshot.py"
        />
      )}

      {data && (
        <div className="grid gap-5 lg:grid-cols-2">
          <MoverColumn
            title="Top Gainers"
            movers={data.gainers}
            horizon={horizon}
            tone="pos"
          />
          <MoverColumn
            title="Top Losers"
            movers={data.losers}
            horizon={horizon}
            tone="neg"
          />
        </div>
      )}
    </div>
  );
}

function MoverColumn({
  title,
  movers,
  horizon,
  tone,
}: {
  title: string;
  movers: Mover[];
  horizon: number;
  tone: "pos" | "neg";
}) {
  const Icon = tone === "pos" ? TrendingUp : TrendingDown;

  return (
    <section className="card overflow-hidden" aria-label={title}>
      <header className="flex items-center justify-between border-b border-line px-5 py-3.5">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <Icon size={15} className={tone === "pos" ? "text-pos" : "text-neg"} />
          {title}
        </h2>
        <span className="num rounded-full border border-line px-2 py-0.5 text-[11px] text-dim">
          {horizon}D horizon
        </span>
      </header>

      {movers.length === 0 ? (
        <EmptyState
          title={`No ${tone === "pos" ? "positive" : "negative"} forecasts`}
          detail={`No asset in this filter is predicted to move ${tone === "pos" ? "up" : "down"} at ${horizon}D.`}
        />
      ) : (
        <ol className="divide-y divide-line/60">
          {movers.map((m, i) => (
            <li key={`${m.asset_class}:${m.ticker}`}>
              <Link
                href={`/forecasts?ticker=${encodeURIComponent(m.ticker)}&class=${m.asset_class}`}
                className="flex items-center gap-2 sm:gap-3 px-3 sm:px-5 py-2.5 sm:py-3 hover:bg-surface-high/50"
              >
                <span className="num w-4 sm:w-5 shrink-0 text-center text-xs text-dim">{i + 1}</span>

                <span className="w-[62px] sm:w-[78px] shrink-0 self-center">
                  <span className="label block truncate rounded border border-line px-1 py-0.5 text-center text-[9px] sm:text-[10px] tracking-normal">
                    {ASSET_CLASS_SHORT[m.asset_class]}
                  </span>
                </span>

                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs sm:text-[13px] font-semibold">{m.name}</span>
                  {/* "Commodity" as a group is just the class badge repeated -
                      every other class's group (sector/cluster) is real context. */}
                  {formatGroup(m.group).toLowerCase() !== "commodity" && (
                    <span className="block truncate text-[10px] sm:text-[11px] text-dim">
                      {formatGroup(m.group)}
                    </span>
                  )}
                </span>

                <span className="num hidden md:inline-block shrink-0 whitespace-nowrap text-right text-[12px]">
                  <span className="text-dim">
                    {formatPrice(m.current_price, m.currency, m.asset_class)}
                  </span>
                  <span className="mx-1 text-dim">→</span>
                  <span className="font-medium">
                    {formatPrice(m.projected_price, m.currency, m.asset_class)}
                  </span>
                </span>

                <span
                  className={cn(
                    "num shrink-0 whitespace-nowrap rounded-md border px-1.5 sm:px-2 py-0.5 sm:py-1 text-right text-[11px] sm:text-xs font-bold w-auto sm:w-[92px]",
                    tone === "pos"
                      ? "border-pos/40 bg-pos/10 text-pos"
                      : "border-neg/40 bg-neg/10 text-neg",
                  )}
                >
                  {directionGlyph(m.predicted_return_pct)} {formatPercent(m.predicted_return_pct)}
                </span>
              </Link>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
