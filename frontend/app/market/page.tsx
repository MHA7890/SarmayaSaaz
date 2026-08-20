"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowUpDown, Search } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { EmptyState, ErrorState, Skeleton } from "@/components/ui/States";
import { api } from "@/lib/api/client";
import { ASSET_CLASS_LABEL, ASSET_CLASS_SHORT, type AssetClass } from "@/lib/api/types";
import {
  cn,
  directionClass,
  directionGlyph,
  formatGroup,
  formatPercent,
  formatPrice,
  formatTicker,
  isInternalId,
  relativeDate,
} from "@/lib/format";

const CLASSES: (AssetClass | "all")[] = ["all", "crypto", "stock", "mutual_fund", "commodity"];
type SortKey = "ticker" | "price" | "change";

export default function MarketPage() {
  const [klass, setKlass] = useState<AssetClass | "all">("all");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortKey>("ticker");
  const [asc, setAsc] = useState(true);

  const { data, isLoading, error } = useQuery({
    queryKey: ["market", klass],
    queryFn: () => api.market(klass === "all" ? undefined : klass),
  });

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const list = (data?.quotes ?? []).filter(
      (q) =>
        !needle ||
        q.ticker.toLowerCase().includes(needle) ||
        q.name.toLowerCase().includes(needle),
    );
    const dir = asc ? 1 : -1;
    return [...list].sort((a, b) => {
      if (sort === "price") return (a.current_price - b.current_price) * dir;
      if (sort === "change")
        return ((a.change_24h_pct ?? 0) - (b.change_24h_pct ?? 0)) * dir;
      return a.ticker.localeCompare(b.ticker) * dir;
    });
  }, [data, query, sort, asc]);

  function toggleSort(key: SortKey) {
    if (sort === key) setAsc((v) => !v);
    else {
      setSort(key);
      setAsc(key === "ticker");
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-bold">Market Explorer</h1>
        <p className="mt-0.5 text-sm text-dim">
          Every asset with a trained model.
          {data?.generated_at && ` Snapshot ${relativeDate(data.generated_at)}.`}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
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
        <div className="relative ml-auto">
          <Search
            size={14}
            className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-dim"
          />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search symbol or name…"
            aria-label="Search the market table"
            className="focus-ring w-full rounded-lg border border-line bg-surface py-1.5 pl-8 pr-3 text-[13px] placeholder:text-dim sm:w-64"
          />
        </div>
      </div>

      {isLoading && <Skeleton className="h-96 w-full" />}
      {error && (
        <ErrorState
          error={error}
          hint="The market table reads a precomputed snapshot. Build one with: uv run python scripts/build_snapshot.py"
        />
      )}

      {data && rows.length === 0 && (
        <EmptyState title="No assets match" detail="Try clearing the search or filter." />
      )}

      {rows.length > 0 && (
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[680px] text-sm">
              <caption className="sr-only">
                Assets with current price and 24 hour change
              </caption>
              <thead>
                <tr className="border-b border-line text-left">
                  <th
                    scope="col"
                    className="px-5 py-2.5"
                    aria-sort={sort === "ticker" ? (asc ? "ascending" : "descending") : "none"}
                  >
                    <SortButton label="Asset" active={sort === "ticker"} asc={asc} onClick={() => toggleSort("ticker")} />
                  </th>
                  <th scope="col" className="label px-5 py-2.5 font-medium">Class</th>
                  <th scope="col" className="label px-5 py-2.5 font-medium">Group</th>
                  <th
                    scope="col"
                    className="px-5 py-2.5 text-right"
                    aria-sort={sort === "price" ? (asc ? "ascending" : "descending") : "none"}
                  >
                    <SortButton label="Price" active={sort === "price"} asc={asc} onClick={() => toggleSort("price")} align="right" />
                  </th>
                  <th
                    scope="col"
                    className="px-5 py-2.5 text-right"
                    aria-sort={sort === "change" ? (asc ? "ascending" : "descending") : "none"}
                  >
                    <SortButton label="24H" active={sort === "change"} asc={asc} onClick={() => toggleSort("change")} align="right" />
                  </th>
                  <th scope="col" className="label min-w-[92px] px-5 py-2.5 text-right font-medium">Action</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((q) => (
                  <tr
                    key={`${q.asset_class}:${q.ticker}`}
                    className="border-b border-line/60 last:border-0 hover:bg-surface-high/50"
                  >
                    <th scope="row" className="px-5 py-3 text-left font-normal">
                      <span className="block max-w-[240px] truncate text-[13px] font-semibold">
                        {q.name}
                      </span>
                      {q.ticker.toLowerCase() !== q.name.toLowerCase() &&
                        !isInternalId(q.ticker) && (
                          <span className="num block truncate text-[11px] text-dim">
                            {formatTicker(q.ticker)}
                          </span>
                        )}
                    </th>
                    <td className="px-5 py-3">
                      <span className="label">{ASSET_CLASS_SHORT[q.asset_class]}</span>
                    </td>
                    <td className="max-w-[160px] truncate px-5 py-3 text-xs text-dim">
                      {formatGroup(q.group)}
                    </td>
                    <td className="num px-5 py-3 text-right font-medium">
                      {formatPrice(q.current_price, q.currency, q.asset_class)}
                    </td>
                    <td
                      className={cn(
                        "num px-5 py-3 text-right font-semibold",
                        directionClass(q.change_24h_pct),
                      )}
                    >
                      {q.change_24h_pct === null ? (
                        <span className="text-dim">—</span>
                      ) : (
                        <>
                          {directionGlyph(q.change_24h_pct)} {formatPercent(q.change_24h_pct)}
                        </>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-5 py-3 text-right">
                      <Link
                        href={`/forecasts?ticker=${encodeURIComponent(q.ticker)}&class=${q.asset_class}`}
                        className="focus-ring inline-block whitespace-nowrap rounded-md border border-line px-2.5 py-1 text-[11px] font-medium text-accent hover:bg-accent-soft"
                      >
                        Forecast →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function SortButton({
  label,
  active,
  asc,
  onClick,
  align = "left",
}: {
  label: string;
  active: boolean;
  asc: boolean;
  onClick: () => void;
  align?: "left" | "right";
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "label focus-ring inline-flex items-center gap-1 rounded font-medium hover:text-text",
        active && "text-text",
        align === "right" && "flex-row-reverse",
      )}
    >
      {label}
      <ArrowUpDown size={11} className={cn(active ? "opacity-100" : "opacity-40")} />
    </button>
  );
}
