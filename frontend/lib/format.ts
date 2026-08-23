import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Price formatting adapts precision to magnitude - a memecoin at $0.000018 and
 * Bitcoin at $62,739 cannot share a decimal count.
 *
 * Mutual fund NAVs always render at a fixed 4 decimals (MUFAP quotes them
 * that way, and the dataset is stored at that precision).
 *
 * Crypto gets a floor of 4 decimals - never the 2 the >=1000/>=1 buckets
 * below would give a $60k BTC or a $3k ETH print - but sub-$0.0001 coins
 * (SHIB, PEPE, BONK, FLOKI all trade in the 1e-6 range) still scale up past
 * 4, since flooring *everything* at 4 would round every one of those to a
 * flat, wrong-looking "$0.0000".
 */
export function formatPrice(
  value: number | null | undefined,
  currency = "USD",
  assetClass?: string,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";

  const symbol = currency === "PKR" ? "Rs " : "$";
  const abs = Math.abs(value);

  let decimals: number;
  if (assetClass === "mutual_fund") decimals = 4;
  else if (assetClass === "crypto") decimals = abs >= 0.0001 ? 4 : 8;
  else if (abs >= 1000) decimals = 2;
  else if (abs >= 1) decimals = 2;
  else if (abs >= 0.01) decimals = 4;
  else decimals = 6;

  return (
    symbol +
    value.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })
  );
}

export function formatPercent(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(decimals)}%`;
}

export function formatCompact(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(
    value,
  );
}

/** Arrow glyph paired with every directional colour so hue never stands alone. */
export function directionGlyph(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value) || value === 0) return "→";
  return value > 0 ? "▲" : "▼";
}

export function directionClass(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "text-dim";
  if (value > 0) return "text-pos";
  if (value < 0) return "text-neg";
  return "text-dim";
}

export function actionClass(action: string): string {
  switch (action) {
    case "STRONG BUY":
    case "BUY":
      return "text-pos border-pos/40 bg-pos/10";
    case "SELL":
    case "STRONG SELL":
      return "text-neg border-neg/40 bg-neg/10";
    default:
      return "text-dim border-line bg-surface-high";
  }
}

export function relativeDate(iso: string | null | undefined, assetClass?: string): string {
  if (!iso) return "unknown";
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return iso;

  const now = new Date();
  const days = Math.floor((now.getTime() - then.getTime()) / 86_400_000);
  if (days <= 0) return iso.slice(0, 10);
  if (days === 1) return "yesterday";
  if (days === 2 && (now.getDay() === 0 || now.getDay() === 6) && assetClass !== "crypto") {
    return "Fri close";
  }
  if (days < 30) return `${days} days ago`;
  const months = Math.floor(days / 30);
  return months === 1 ? "1 month ago" : `${months} months ago`;
}



export function horizonLabel(days: number): string {
  return `${days}D`;
}

export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US");
}

/**
 * Display names for tickers whose raw form isn't presentable
 * (backend keys like "crude_oil" and "natural_gas" carry underscores because
 * they're also file/column identifiers). Falls back to a straight
 * underscore-to-space swap for anything not explicitly mapped.
 */
const TICKER_DISPLAY: Record<string, string> = {
  crude_oil: "WTI Crude",
  natural_gas: "Nat Gas",
};

export function formatTicker(ticker: string): string {
  return TICKER_DISPLAY[ticker] ?? ticker.replace(/_/g, " ");
}

/**
 * Group/sector strings are filesystem-derived (e.g. a data directory named
 * "Consumer_Autos") and carry that underscore straight through to the API -
 * display-only cleanup, not a semantic transform.
 */
export function formatGroup(group: string): string {
  // Groups arrive in three shapes: underscored PSX sectors
  // ("Cement_Construction"), spaced crypto clusters ("High Volatility Alts"),
  // and one camel-cased MUFAP cluster ("MoneyMarket"). Normalise all three.
  return group.replace(/_/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2");
}

/** True for a raw internal id (e.g. "natural_gas") rather than a real market
 * symbol - these exist to key data files, not to be shown next to a name. */
export function isInternalId(ticker: string): boolean {
  return ticker.includes("_");
}

/**
 * Signed, non-scientific formatting for small model-attribution magnitudes
 * (SHAP impact, etc). These sit around 1e-3 to 1e-4, so a flat two-decimal
 * rule would round every one to "0.00" - decimals scale up as the value
 * shrinks instead, the same magnitude-adaptive idea as formatPrice.
 */
export function formatSigned(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";

  const abs = Math.abs(value);
  let decimals: number;
  if (abs >= 1) decimals = 2;
  else if (abs >= 0.01) decimals = 4;
  else if (abs >= 0.0001) decimals = 6;
  else decimals = 8;

  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}`;
}
