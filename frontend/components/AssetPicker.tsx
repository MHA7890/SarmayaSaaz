"use client";

import { useQuery } from "@tanstack/react-query";
import { ChevronDown } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api/client";
import { ASSET_CLASS_LABEL, type Asset, type AssetClass } from "@/lib/api/types";
import { cn, formatGroup, formatTicker, isInternalId } from "@/lib/format";

const CLASS_FILTERS: { label: string; value: AssetClass | "all" }[] = [
  { label: "All", value: "all" },
  { label: "Crypto", value: "crypto" },
  { label: "PSX Stock", value: "stock" },
  { label: "Mutual Fund", value: "mutual_fund" },
  { label: "Commodity", value: "commodity" },
];

export function AssetPicker({
  value,
  assetClass,
  onChange,
}: {
  value: string;
  assetClass?: AssetClass;
  onChange: (ticker: string, assetClass: AssetClass) => void;
}) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const [classFilter, setClassFilter] = useState<AssetClass | "all">("all");
  const [cursor, setCursor] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const activeRef = useRef<HTMLButtonElement>(null);

  const { data } = useQuery({
    queryKey: ["assets", "all"],
    queryFn: () => api.assets({ limit: 2000 }),
  });

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const grouped = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    const all = (data?.assets ?? []).filter(
      (a) =>
        (classFilter === "all" || a.asset_class === classFilter) &&
        (!needle ||
          a.ticker.toLowerCase().includes(needle) ||
          a.name.toLowerCase().includes(needle)),
    );
    const out = new Map<AssetClass, Asset[]>();
    for (const a of all) {
      const list = out.get(a.asset_class) ?? [];
      list.push(a);
      out.set(a.asset_class, list);
    }
    return out;
  }, [data, filter, classFilter]);

  const flat = useMemo(() => [...grouped.values()].flat(), [grouped]);

  useEffect(() => setCursor(0), [filter, classFilter]);
  useEffect(() => {
    if (open) activeRef.current?.scrollIntoView({ block: "nearest" });
  }, [cursor, open]);

  const current = (data?.assets ?? []).find(
    (a) => a.ticker === value && (!assetClass || a.asset_class === assetClass),
  );

  function select(a: Asset) {
    onChange(a.ticker, a.asset_class);
    setOpen(false);
    setFilter("");
  }

  return (
    <div ref={ref} className="relative w-full sm:w-auto">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="focus-ring flex w-full sm:w-auto min-w-0 sm:min-w-[220px] items-center justify-between gap-3 rounded-lg border border-line bg-surface px-3 py-2 text-left"
      >
        <span className="min-w-0">
          <span className="block truncate text-sm font-semibold">
            {current?.name ?? value ?? "Select an asset"}
          </span>
          {current && (
            <span className="block truncate text-[11px] text-dim">
              {!isInternalId(current.ticker) && current.ticker.toLowerCase() !== current.name.toLowerCase() && (
                <span className="num">{formatTicker(current.ticker)} · </span>
              )}
              {formatGroup(current.group)}
            </span>
          )}
        </span>
        <ChevronDown size={15} className="shrink-0 text-dim" />
      </button>

      {open && (
        <div className="absolute left-0 sm:left-auto right-0 sm:right-auto z-50 mt-1.5 max-h-[420px] w-[calc(100vw-2rem)] max-w-[340px] sm:w-[340px] overflow-hidden rounded-lg border border-line bg-surface shadow-2xl">
          <div className="space-y-2 border-b border-line p-2">
            <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by asset class">
              {CLASS_FILTERS.map((f) => (
                <button
                  key={f.value}
                  onClick={() => setClassFilter(f.value)}
                  aria-pressed={classFilter === f.value}
                  className={cn("pill", classFilter === f.value && "pill-active")}
                >
                  {f.label}
                </button>
              ))}
            </div>
            <input
              ref={inputRef}
              autoFocus
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setCursor((c) => Math.min(c + 1, flat.length - 1));
                } else if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setCursor((c) => Math.max(c - 1, 0));
                } else if (e.key === "Enter") {
                  e.preventDefault();
                  const hit = flat[cursor];
                  if (hit) select(hit);
                } else if (e.key === "Escape") {
                  setOpen(false);
                }
              }}
              placeholder="Filter assets…"
              aria-label="Filter assets"
              className="focus-ring w-full rounded-md border border-line bg-surface-inset px-2.5 py-1.5 text-[13px] placeholder:text-dim"
            />
          </div>
          <ul role="listbox" className="max-h-[350px] overflow-y-auto p-1">
            {[...grouped.entries()].map(([klass, items]) => (
              <li key={klass}>
                <p className="label px-2 py-1.5">
                  {ASSET_CLASS_LABEL[klass]} ({items.length})
                </p>
                <ul>
                  {items.map((a) => {
                    const flatIndex = flat.indexOf(a);
                    const active = flatIndex === cursor;
                    return (
                    <li key={`${a.asset_class}:${a.ticker}`}>
                      <button
                        ref={active ? activeRef : undefined}
                        role="option"
                        aria-selected={a.ticker === value}
                        onMouseEnter={() => setCursor(flatIndex)}
                        onClick={() => select(a)}
                        className={cn(
                          "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-surface-high",
                          active && "bg-surface-high",
                          a.ticker === value && "bg-accent-soft",
                        )}
                      >
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-xs font-semibold">{a.name}</span>
                          {!isInternalId(a.ticker) && a.ticker.toLowerCase() !== a.name.toLowerCase() && (
                            <span className="num block truncate text-[11px] text-dim">
                              {formatTicker(a.ticker)}
                            </span>
                          )}
                        </span>
                      </button>
                    </li>
                    );
                  })}
                </ul>
              </li>
            ))}
            {grouped.size === 0 && (
              <li className="px-3 py-6 text-center text-xs text-dim">No assets match.</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
