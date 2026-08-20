"use client";

import { useQuery } from "@tanstack/react-query";
import {
  BookOpen,
  Grid3x3,
  LayoutDashboard,
  LineChart,
  Moon,
  Search,
  Store,
  Sun,
  X,
} from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import logoDark from "@/public/logo-dark.png";
import logoLight from "@/public/logo-light.png";

import { api } from "@/lib/api/client";
import { ASSET_CLASS_SHORT } from "@/lib/api/types";
import { cn, formatTicker } from "@/lib/format";

const LINKS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/market", label: "Markets", icon: Store },
  { href: "/forecasts", label: "Forecast", icon: LineChart },
  { href: "/heatmap", label: "Model Heatmap", icon: Grid3x3 },
  { href: "/methodology", label: "Methodology", icon: BookOpen },
];

export function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const [dark, setDark] = useState(true);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggleTheme() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      /* private mode */
    }
  }

  const { data } = useQuery({
    queryKey: ["assets", "all"],
    queryFn: () => api.assets({ limit: 2000 }),
  });

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (needle.length < 1) return [];
    const all = data?.assets ?? [];
    const scored = all
      .map((a) => {
        const t = a.ticker.toLowerCase();
        const n = a.name.toLowerCase();
        if (t === needle) return { a, score: 0 };
        if (t.startsWith(needle)) return { a, score: 1 };
        if (n.startsWith(needle)) return { a, score: 2 };
        if (t.includes(needle) || n.includes(needle)) return { a, score: 3 };
        return null;
      })
      .filter((x): x is { a: (typeof all)[number]; score: number } => x !== null)
      .sort((x, y) => x.score - y.score)
      .slice(0, 8);
    return scored.map((s) => s.a);
  }, [query, data]);

  useEffect(() => setCursor(0), [query]);

  // Cmd/Ctrl-K focuses search, Escape dismisses it.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        inputRef.current?.focus();
        setOpen(true);
      }
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  function go(ticker: string, assetClass: string) {
    setOpen(false);
    setQuery("");
    inputRef.current?.blur();
    router.push(`/forecasts?ticker=${encodeURIComponent(ticker)}&class=${assetClass}`);
  }

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/85 backdrop-blur">
      <div className="mx-auto flex h-14 w-full max-w-[1400px] items-center gap-3 px-4 sm:px-6">
        <div className="flex flex-1 items-center">
          {/* Two files rather than one plus a CSS filter: the wordmark's green
              and the arrow's gold both need lifting for the dark surface, and
              swapping via `dark:` keeps it flash-free on first paint. The
              Link carries the accessible name so neither copy is announced. */}
          <Link
            href="/"
            aria-label="SarmayaSaaz — home"
            className="focus-ring flex shrink-0 items-center rounded-md"
          >
            <Image
              src={logoLight}
              alt=""
              aria-hidden
              priority
              className="h-8 w-auto dark:hidden"
            />
            <Image
              src={logoDark}
              alt=""
              aria-hidden
              priority
              className="hidden h-8 w-auto dark:block"
            />
          </Link>
        </div>

        <nav className="hidden shrink-0 items-center gap-0.5 lg:flex">
          {LINKS.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "focus-ring flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[13px] font-medium transition-colors",
                  active
                    ? "bg-accent-soft text-accent"
                    : "text-muted hover:bg-surface-high hover:text-text",
                )}
              >
                <Icon size={14} />
                {label}
              </Link>
            );
          })}
        </nav>

        <div className="flex flex-1 items-center justify-end gap-3">
          <div ref={boxRef} className="relative w-full max-w-[280px]">
            <Search
              size={14}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-dim"
            />
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setOpen(true);
              }}
              onFocus={() => setOpen(true)}
              onKeyDown={(e) => {
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setCursor((c) => Math.min(c + 1, results.length - 1));
                } else if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setCursor((c) => Math.max(c - 1, 0));
                } else if (e.key === "Enter") {
                  const hit = results[cursor];
                  if (hit) go(hit.ticker, hit.asset_class);
                }
              }}
              placeholder="Search BTC, gold, OGDC…"
              aria-label="Search assets"
              className="focus-ring w-full rounded-lg border border-line bg-surface py-1.5 pl-8 pr-8 text-[13px] placeholder:text-dim"
            />
            {query ? (
              <button
                onClick={() => {
                  setQuery("");
                  inputRef.current?.focus();
                }}
                aria-label="Clear search"
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-dim hover:text-text"
              >
                <X size={13} />
              </button>
            ) : (
              <kbd
                aria-hidden="true"
                className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded border border-line bg-surface-inset px-1.5 py-0.5 text-[10px] font-medium text-dim"
              >
                ⌘K
              </kbd>
            )}

            {open && results.length > 0 && (
              <ul
                role="listbox"
                className="absolute left-0 right-0 top-full z-50 mt-1.5 overflow-hidden rounded-lg border border-line bg-surface shadow-xl"
              >
                {results.map((a, i) => (
                  <li
                    key={`${a.asset_class}:${a.ticker}`}
                    role="option"
                    aria-selected={i === cursor}
                  >
                    <button
                      onMouseEnter={() => setCursor(i)}
                      onClick={() => go(a.ticker, a.asset_class)}
                      className={cn(
                        "flex w-full items-center gap-2 px-3 py-2 text-left",
                        i === cursor ? "bg-surface-high" : "",
                      )}
                    >
                      <span className="num shrink-0 text-[13px] font-semibold">
                        {formatTicker(a.ticker)}
                      </span>
                      <span className="truncate text-xs text-dim">{a.name}</span>
                      <span className="label ml-auto shrink-0">
                        {ASSET_CLASS_SHORT[a.asset_class]}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <button
            onClick={toggleTheme}
            aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
            className="focus-ring shrink-0 rounded-md border border-line p-1.5 text-muted hover:text-text"
          >
            {dark ? <Sun size={15} /> : <Moon size={15} />}
          </button>
        </div>
      </div>
    </header>
  );
}
