/**
 * Mirrors backend/schemas.py. Keep in sync - the backend publishes an OpenAPI
 * document at /openapi.json if these ever drift.
 */

export type AssetClass = "crypto" | "commodity" | "mutual_fund" | "stock";
export type Direction = "up" | "down" | "neutral";
export type Action = "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL";

export interface Asset {
  ticker: string;
  name: string;
  asset_class: AssetClass;
  group: string;
  currency: string;
  unit: string;
}

export interface AssetList {
  count: number;
  assets: Asset[];
}

export interface ModelContribution {
  model_name: string;
  predicted_return_pct: number;
  weight: number;
}

export interface HorizonForecast {
  horizon_days: number;
  predicted_return_pct: number;
  projected_price: number;
  direction: Direction;
  /** Null when the metrics needed to derive a band were never recorded. */
  lower_bound: number | null;
  upper_bound: number | null;
  confidence_pct: number | null;
  primary_model: string;
  contributions: ModelContribution[];
}

export interface MarketDriver {
  feature: string;
  impact: number;
  direction: Direction;
  headline: string | null;
  url: string | null;
}

export interface PricePoint {
  date: string;
  price: number;
}

export interface NewsCatalyst {
  date: string;
  headline: string;
  url: string | null;
  source: string | null;
  market_wide: boolean;
}

export interface Forecast {
  ticker: string;
  name: string;
  asset_class: AssetClass;
  group: string;
  currency: string;
  unit: string;
  current_price: number;
  change_24h_pct: number | null;
  as_of: string;
  price_source: string;
  source_url: string | null;
  action: Action;
  sentiment_score: number | null;
  horizons: HorizonForecast[];
  history: PricePoint[];
  drivers: MarketDriver[];
  catalysts: NewsCatalyst[];
  warnings: string[];
}

export interface Quote {
  ticker: string;
  name: string;
  asset_class: AssetClass;
  group: string;
  currency: string;
  unit: string;
  current_price: number;
  change_24h_pct: number | null;
}

export interface MarketResponse {
  count: number;
  generated_at: string;
  quotes: Quote[];
}

export interface Mover {
  ticker: string;
  name: string;
  asset_class: AssetClass;
  group: string;
  currency: string;
  current_price: number;
  projected_price: number;
  predicted_return_pct: number;
  horizon_days: number;
  primary_model: string;
  confidence_pct: number | null;
}

export interface MoversResponse {
  horizon_days: number;
  generated_at: string;
  gainers: Mover[];
  losers: Mover[];
}

export interface ModelScore {
  model_name: string;
  directional_accuracy: number | null;
  mae: number | null;
  rmse: number | null;
  r2: number | null;
  win_rate: number | null;
}

export interface LeaderboardEntry {
  ticker: string;
  name: string;
  asset_class: AssetClass;
  group: string;
  horizon_days: number;
  scores: ModelScore[];
  best_model: string | null;
}

export interface Leaderboard {
  horizon_days: number;
  metric: string;
  models: string[];
  entries: LeaderboardEntry[];
}

export interface EngineStatus {
  name: string;
  online: boolean;
  asset_count: number;
  detail: string | null;
}

export interface Health {
  status: string;
  version: string;
  engines: EngineStatus[];
  model_cache: Record<string, number>;
}

export interface SystemStats {
  total_assets: number;
  engines_online: number;
  engines_total: number;
  artifacts_on_disk: number;
  horizons: number[];
}

export interface SnapshotMeta {
  available: boolean;
  building: boolean;
  generated_at?: string;
  asset_count?: number;
  failed?: number;
}

export const HORIZONS = [7, 14, 28, 42, 60, 90, 120] as const;

export const ASSET_CLASS_LABEL: Record<AssetClass, string> = {
  crypto: "Cryptocurrencies",
  stock: "PSX Equities",
  mutual_fund: "Mutual Funds",
  commodity: "Commodities",
};

export const ASSET_CLASS_SHORT: Record<AssetClass, string> = {
  crypto: "CRYPTO",
  stock: "PSX",
  mutual_fund: "FUND",
  commodity: "COMMODITY",
};
