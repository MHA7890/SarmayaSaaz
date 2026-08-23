"use client";

import { type MouseEvent, useMemo, useRef, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Forecast, NewsCatalyst } from "@/lib/api/types";
import { cn, formatPrice } from "@/lib/format";

/** Series colours come from the validated categorical ramp in globals.css. */
const HISTORICAL = "rgb(var(--series-1))";
const PROJECTED = "rgb(var(--series-2))";

type Row = {
  date: string;
  pos: number;
  price?: number;
  projected?: number;
  band?: [number, number];
  isNextOpen?: boolean;
};

export function isMarketClosed(assetClass: string): boolean {
  const now = new Date();
  const utcDay = now.getUTCDay();
  const utcHour = now.getUTCHours();
  const utcMinutes = utcHour * 60 + now.getUTCMinutes();
  const ac = (assetClass || "").toLowerCase();


  if (ac === "stock" || ac === "psx") {
    if (utcDay === 0 || utcDay === 6) return true;
    const openMinutes = 4 * 60 + 30; // 04:30 UTC (09:30 PKT)
    const closeMinutes = 10 * 60 + 30; // 10:30 UTC (15:30 PKT)
    return utcMinutes < openMinutes || utcMinutes >= closeMinutes;
  }

  if (ac === "commodity") {
    if (utcDay === 6) return true; // Sat
    if (utcDay === 5 && utcHour >= 21) return true; // Fri after 21:00 UTC
    if (utcDay === 0 && utcHour < 22) return true; // Sun before 22:00 UTC
    if (utcHour === 21) return true; // Daily 21:00-22:00 UTC break
    return false;
  }

  return false;
}

function getNextTradingDate(asOf: string, assetClass: string): string {
  const dt = new Date(asOf);
  dt.setDate(dt.getDate() + 1);
  const day = dt.getDay();
  const ac = (assetClass || "").toLowerCase();
  if (ac === "stock" || ac === "psx") {
    if (day === 6) dt.setDate(dt.getDate() + 2);
    else if (day === 0) dt.setDate(dt.getDate() + 1);
  }
  return dt.toISOString().slice(0, 10);
}

/**
 * History gets a fixed 70% of the plot width and the projection cone gets
 * the other 30%, regardless of how many points fall in each - the x-axis is
 * a synthetic [0, 1] position, not a real time scale.
 */
const HISTORY_FRACTION = 0.7;

function buildSeries(
  forecast: Forecast,
  historyDays: number,
  selectedHorizonDays: number | null,
): { rows: Row[]; histLen: number; toPos: (date: string) => number; nextOpenPrice?: number } {
  const history = forecast.history.slice(-historyDays);
  const historyRows: Omit<Row, "pos">[] = history.map((p) => ({ date: p.date, price: p.price }));

  const lastStored = historyRows.at(-1);
  if (!lastStored || lastStored.date !== forecast.as_of) {
    historyRows.push({ date: forecast.as_of, price: forecast.current_price });
  }

  const firstDate = historyRows[0] ? new Date(historyRows[0].date).getTime() : null;
  const todayMs = new Date(forecast.as_of).getTime();
  const histSpanMs = firstDate !== null ? todayMs - firstDate : 0;

  const toPos = (date: string): number => {
    if (firstDate === null || histSpanMs <= 0) return HISTORY_FRACTION;
    const frac = (new Date(date).getTime() - firstDate) / histSpanMs;
    return HISTORY_FRACTION * Math.max(0, Math.min(1, frac));
  };

  const rows: Row[] = historyRows.map((r) => ({ ...r, pos: toPos(r.date) }));

  const anchor = rows.at(-1);
  if (!anchor) return { rows, histLen: rows.length, toPos };
  anchor.pos = HISTORY_FRACTION;
  anchor.projected = forecast.current_price;
  anchor.band = [forecast.current_price, forecast.current_price];
  const histLen = rows.length;

  const sortedHorizons = [...forecast.horizons]
    .sort((a, b) => a.horizon_days - b.horizon_days)
    .filter((h) => selectedHorizonDays === null || h.horizon_days <= selectedHorizonDays);
  const maxHorizon = sortedHorizons.at(-1)?.horizon_days ?? 0;
  const start = new Date(forecast.as_of);

  const marketClosed = isMarketClosed(forecast.asset_class);
  const isEligibleAsset = forecast.asset_class === "commodity" || forecast.asset_class === "stock";
  let nextOpenPrice: number | undefined;

  if (marketClosed && isEligibleAsset && sortedHorizons.length > 0) {
    const h0 = sortedHorizons[0];
    if (h0) {
      const dailyRate = Math.pow(h0.projected_price / forecast.current_price, 1 / Math.max(1, h0.horizon_days));
      nextOpenPrice = forecast.current_price * dailyRate;
      const nextDate = getNextTradingDate(forecast.as_of, forecast.asset_class);
      const nextOpenPos = HISTORY_FRACTION + (maxHorizon > 0 ? (1 - HISTORY_FRACTION) * (1 / maxHorizon) : 0.05);

      rows.push({
        date: `${nextDate} (Next Open)`,
        pos: nextOpenPos,
        projected: nextOpenPrice,
        isNextOpen: true,
        band: [nextOpenPrice, nextOpenPrice],
      });
    }
  }


  for (const h of sortedHorizons) {
    const at = new Date(start);
    at.setDate(at.getDate() + h.horizon_days);
    const pos =
      maxHorizon > 0
        ? HISTORY_FRACTION + (1 - HISTORY_FRACTION) * (h.horizon_days / maxHorizon)
        : HISTORY_FRACTION;
    rows.push({
      date: at.toISOString().slice(0, 10),
      pos,
      projected: h.projected_price,
      band:
        h.lower_bound !== null && h.upper_bound !== null
          ? [h.lower_bound, h.upper_bound]
          : undefined,
    });
  }
  return { rows, histLen, toPos, nextOpenPrice };
}

/** Evenly-spaced sample of up to `n` rows from `arr`, endpoints included. */
function sampleRows(arr: Row[], n: number): Row[] {
  if (arr.length <= n) return arr;
  if (n <= 1) return arr.slice(0, 1);
  const out: Row[] = [];
  for (let i = 0; i < n; i++) {
    const row = arr[Math.round((i / (n - 1)) * (arr.length - 1))];
    if (row) out.push(row);
  }
  return out;
}

const RANGES = [
  { label: "3M", days: 90 },
  { label: "6M", days: 180 },
  { label: "1Y", days: 365 },
];

const DEFAULT_RANGE_DAYS = 90;
const MAX_CATALYST_DATES = 10;

type CatalystGroup = { date: string; pos: number; items: NewsCatalyst[] };

export function ForecastChart({
  forecast,
  selectedHorizonDays = null,
}: {
  forecast: Forecast;
  selectedHorizonDays?: number | null;
}) {
  const [range, setRange] = useState(DEFAULT_RANGE_DAYS);
  const { rows: data, histLen, toPos, nextOpenPrice } = useMemo(
    () => buildSeries(forecast, range, selectedHorizonDays),
    [forecast, range, selectedHorizonDays],
  );
  const junction = forecast.as_of;
  const marketClosed = isMarketClosed(forecast.asset_class);
  const isEligibleAsset = forecast.asset_class === "commodity" || forecast.asset_class === "stock";

  const hasBand = forecast.horizons.some(
    (h) => h.lower_bound !== null && h.upper_bound !== null,
  );

  const { tickPositions, tickLabelMap } = useMemo(() => {
    const chosen = [...sampleRows(data.slice(0, histLen), 5), ...sampleRows(data.slice(histLen), 3)];
    const map = new Map<number, string>();
    const positions: number[] = [];
    for (const r of chosen) {
      if (!map.has(r.pos)) {
        map.set(r.pos, r.date.slice(2, 7));
        positions.push(r.pos);
      }
    }
    positions.sort((a, b) => a - b);
    return { tickPositions: positions, tickLabelMap: map };
  }, [data, histLen]);

  const visibleCatalysts = useMemo(() => {
    const earliest = data[0]?.date;
    if (!earliest || !junction) return [];
    const byDate = new Map<string, NewsCatalyst[]>();
    for (const c of forecast.catalysts) {
      if (c.date < earliest || c.date > junction) continue;
      const list = byDate.get(c.date) ?? [];
      list.push(c);
      byDate.set(c.date, list);
    }
    for (const items of byDate.values()) {
      items.sort((a, b) => Number(a.market_wide) - Number(b.market_wide));
    }

    const groups = [...byDate.entries()]
      .map(([date, items]) => ({ date, items, pos: toPos(date) }))
      .sort((a, b) => (a.date < b.date ? -1 : 1));

    if (groups.length <= MAX_CATALYST_DATES) return groups;
    const step = (groups.length - 1) / (MAX_CATALYST_DATES - 1);
    const picked: typeof groups = [];
    for (let i = 0; i < MAX_CATALYST_DATES; i++) {
      const g = groups[Math.round(i * step)];
      if (g && picked[picked.length - 1] !== g) picked.push(g);
    }
    return picked;
  }, [forecast.catalysts, data, junction, toPos]);

  const containerRef = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState<{ group: CatalystGroup; x: number; y: number } | null>(
    null,
  );

  function showCatalyst(group: CatalystGroup, e: MouseEvent) {
    const bounds = containerRef.current?.getBoundingClientRect();
    if (!bounds) return;
    setHovered({
      group,
      x: Math.min(Math.max(e.clientX - bounds.left, 90), bounds.width - 90),
      y: e.clientY - bounds.top,
    });
  }

  return (
    <section className="card p-5" aria-label="Price history and forecast projection">
      <header className="mb-1 flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold">Price History &amp; AI Projection</h2>
            {marketClosed && isEligibleAsset && nextOpenPrice !== undefined && (
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-amber-500/10 border border-amber-500/30 text-amber-300 flex items-center gap-1.5 shadow-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                Market Closed · Projected Next Open: {formatPrice(nextOpenPrice, forecast.currency, forecast.asset_class)}
              </span>
            )}
          </div>
          <p className="mt-0.5 text-xs text-dim">
            {forecast.unit} · observed through {forecast.as_of}
          </p>
        </div>
        <div className="flex gap-1" role="group" aria-label="History range">
          {RANGES.map((r) => (
            <button
              key={r.label}
              onClick={() => setRange(r.days)}
              aria-pressed={range === r.days}
              className={cn("pill", range === r.days && "pill-active")}
            >
              {r.label}
            </button>
          ))}
        </div>
      </header>

      {/* Legend is always present for >= 2 series */}
      <div className="mb-3 flex flex-wrap items-center gap-4 text-xs">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded" style={{ background: HISTORICAL }} />
          <span className="text-muted">Historical price</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="16" height="2" aria-hidden="true">
            <line
              x1="0" y1="1" x2="16" y2="1"
              stroke={PROJECTED} strokeWidth="2" strokeDasharray="4 3"
            />
          </svg>
          <span className="text-muted">AI projection</span>
        </span>
        {hasBand && (
          <span className="inline-flex items-center gap-1.5">
            <span
              className="h-3 w-4 rounded-sm"
              style={{ background: PROJECTED, opacity: 0.18 }}
            />
            <span className="text-muted">Confidence range</span>
          </span>
        )}
      </div>

      <div ref={containerRef} className="relative h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid
              stroke="rgb(var(--line))"
              strokeDasharray="3 3"
              vertical={false}
            />
            <XAxis
              dataKey="pos"
              type="number"
              domain={[0, 1]}
              ticks={tickPositions}
              tick={{ fill: "rgb(var(--dim))", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "rgb(var(--line))" }}
              tickFormatter={(p: number) => tickLabelMap.get(p) ?? ""}
            />
            <YAxis
              tick={{ fill: "rgb(var(--dim))", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={72}
              domain={["auto", "auto"]}
              tickFormatter={(v: number) => formatPrice(v, forecast.currency, forecast.asset_class)}
            />

            {junction && (
              <ReferenceLine
                x={HISTORY_FRACTION}
                stroke="rgb(var(--dim))"
                strokeDasharray="2 4"
              />
            )}

            {hasBand && (
              <Area
                dataKey="band"
                stroke="none"
                fill={PROJECTED}
                fillOpacity={0.18}
                isAnimationActive={false}
                connectNulls
              />
            )}
            <Line
              dataKey="price"
              stroke={HISTORICAL}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              name="Historical price"
            />
            <Line
              dataKey="projected"
              stroke={PROJECTED}
              strokeWidth={2}
              strokeDasharray="4 3"
              dot={{ r: 3, fill: PROJECTED, strokeWidth: 0 }}
              isAnimationActive={false}
              connectNulls
              name="AI projection"
            />

            {visibleCatalysts.map((g) => (
              <ReferenceLine
                key={g.date}
                x={g.pos}
                stroke="rgb(var(--warn))"
                strokeDasharray="2 4"
                strokeOpacity={0.5}
                label={(props: { viewBox: { x: number; y: number } }) => (
                  <circle
                    data-testid="catalyst-marker"
                    cx={props.viewBox.x}
                    cy={props.viewBox.y + 6}
                    r={4}
                    fill="rgb(var(--warn))"
                    stroke="rgb(var(--surface))"
                    strokeWidth={1.5}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={(e) => showCatalyst(g, e)}
                    onMouseLeave={() => setHovered(null)}
                    onClick={() => {
                      const url = g.items[0]?.url;
                      if (url) window.open(url, "_blank", "noopener,noreferrer");
                    }}
                  />
                )}
              />
            ))}

            <Tooltip
              cursor={{ stroke: "rgb(var(--dim))", strokeDasharray: "3 3" }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const row = payload[0]?.payload as Row | undefined;
                if (!row) return null;
                return (
                  <div className="rounded-lg border border-line bg-surface px-3 py-2 shadow-xl">
                    <p className="num mb-1 text-[11px] text-dim">{row.date}</p>
                    {row.price !== undefined && (
                      <p className="num text-xs">
                        <span className="text-dim">Actual </span>
                        {formatPrice(row.price, forecast.currency, forecast.asset_class)}
                      </p>
                    )}
                    {row.isNextOpen ? (
                      <p className="num text-xs font-semibold text-amber-300">
                        <span className="text-amber-400/80">Projected Next Open </span>
                        {formatPrice(row.projected ?? 0, forecast.currency, forecast.asset_class)}
                      </p>
                    ) : (
                      row.projected !== undefined && (
                        <p className="num text-xs">
                          <span className="text-dim">Projected </span>
                          {formatPrice(row.projected, forecast.currency, forecast.asset_class)}
                        </p>
                      )
                    )}
                    {row.band && row.band[0] !== row.band[1] && !row.isNextOpen && (
                      <p className="num mt-0.5 text-[11px] text-dim">
                        Range {formatPrice(row.band[0], forecast.currency, forecast.asset_class)}{" "}
                        – {formatPrice(row.band[1], forecast.currency, forecast.asset_class)}
                      </p>
                    )}
                  </div>
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>

        {hovered && (
          <div
            className="pointer-events-none absolute z-10 w-[240px] -translate-x-1/2 rounded-lg border border-line bg-surface p-3 shadow-xl"
            style={{ left: hovered.x, top: Math.max(hovered.y - 12, 4) }}
          >
            <p className="num text-[10px] text-dim">
              {hovered.group.date}
              {hovered.group.items[0]?.source ? ` · ${hovered.group.items[0].source}` : ""}
            </p>
            <p className="mt-1 text-xs font-medium leading-snug">
              {hovered.group.items[0]?.headline}
            </p>
            {hovered.group.items.length > 1 && (
              <p className="mt-1 text-[11px] text-dim">
                +{hovered.group.items.length - 1} more headline
                {hovered.group.items.length - 1 === 1 ? "" : "s"} this day
              </p>
            )}
            {hovered.group.items[0]?.url && (
              <p className="mt-1.5 text-[11px] text-accent">Click marker to read</p>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

