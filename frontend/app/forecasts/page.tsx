"use client";

import { useQuery } from "@tanstack/react-query";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { AssetPicker } from "@/components/AssetPicker";
import { ForecastChart } from "@/components/charts/ForecastChart";
import {
  AssetHeader,
  EnsembleBreakdown,
  HorizonMatrix,
  WarningsPanel,
} from "@/components/forecast/ForecastPanels";
import { ErrorState, Skeleton } from "@/components/ui/States";
import { api } from "@/lib/api/client";
import type { AssetClass } from "@/lib/api/types";
import { cn } from "@/lib/format";

function Workspace() {
  const params = useSearchParams();
  const router = useRouter();

  const ticker = params.get("ticker") ?? "gold";
  const assetClass = (params.get("class") as AssetClass | null) ?? "commodity";
  const [selected, setSelected] = useState<number | null>(null);

  function select(t: string, c: AssetClass) {
    router.push(`/forecasts?ticker=${encodeURIComponent(t)}&class=${c}`);
  }

  const { data, isLoading, error } = useQuery({
    queryKey: ["forecast", assetClass, ticker],
    queryFn: () => api.forecast(ticker, assetClass),
  });

  const active =
    data?.horizons.find((h) => h.horizon_days === selected) ??
    data?.horizons.find((h) => h.horizon_days === 28) ??
    data?.horizons[0];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-lg font-bold">Forecast Workspace</h1>
          <p className="mt-0.5 text-sm text-dim">
            Multi horizon targets, ensemble composition and confidence intervals.
          </p>
        </div>
        <AssetPicker value={ticker} assetClass={assetClass} onChange={select} />
      </div>

      {isLoading && (
        <div className="grid gap-5 lg:grid-cols-[1.6fr_1fr]">
          <Skeleton className="h-[420px]" />
          <Skeleton className="h-[420px]" />
        </div>
      )}
      {error && <ErrorState error={error} />}

      {data && (
        <>
          <AssetHeader forecast={data} />

          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Select horizon">
            {data.horizons.map((h) => (
              <button
                key={h.horizon_days}
                onClick={() => setSelected(h.horizon_days)}
                aria-pressed={active?.horizon_days === h.horizon_days}
                className={cn(
                  "pill",
                  active?.horizon_days === h.horizon_days && "pill-active",
                )}
              >
                {h.horizon_days}D
              </button>
            ))}
          </div>

          <div className="grid gap-5 lg:grid-cols-[1.6fr_1fr]">
            <ForecastChart forecast={data} selectedHorizonDays={active?.horizon_days ?? null} />
            <div className="space-y-5">
              {active && (
                <EnsembleBreakdown
                  horizon={active}
                  currency={data.currency}
                  assetClass={data.asset_class}
                />
              )}
            </div>
          </div>

          <HorizonMatrix forecast={data} />
          <WarningsPanel warnings={data.warnings} />
        </>
      )}
    </div>
  );
}

export default function ForecastsPage() {
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <Workspace />
    </Suspense>
  );
}
