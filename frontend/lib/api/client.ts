import type {
  AssetClass,
  AssetList,
  Forecast,
  Health,
  Leaderboard,
  MarketResponse,
  MoversResponse,
  SnapshotMeta,
  SystemStats,
} from "./types";

/**
 * Requests go to a relative /api path, which next.config.mjs rewrites to the
 * FastAPI process. That keeps the browser same-origin so there is no CORS
 * preflight on every call.
 */
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, params?: Record<string, unknown>): Promise<T> {
  const url = new URL(`${BASE}/api${path}`, typeof window === "undefined" ? "http://127.0.0.1:3000" : window.location.origin);

  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }
  }

  let res: Response;
  try {
    res = await fetch(url.toString(), { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError(
      "Cannot reach the forecasting API. Is the backend running on port 8000?",
      0,
    );
  }

  if (!res.ok) {
    let detail: string | undefined;
    try {
      detail = (await res.json())?.detail;
    } catch {
      /* response had no JSON body */
    }
    throw new ApiError(detail ?? `Request failed (${res.status})`, res.status, detail);
  }

  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  stats: () => request<SystemStats>("/stats"),
  snapshot: () => request<SnapshotMeta>("/snapshot"),

  assets: (params?: { asset_class?: AssetClass; search?: string; limit?: number }) =>
    request<AssetList>("/assets", params),

  forecast: (ticker: string, assetClass?: AssetClass) =>
    request<Forecast>(`/forecast/${encodeURIComponent(ticker)}`, {
      asset_class: assetClass,
    }),

  market: (assetClass?: AssetClass) =>
    request<MarketResponse>("/market", { asset_class: assetClass }),

  movers: (params: { horizon_days: number; asset_class?: AssetClass; limit?: number }) =>
    request<MoversResponse>("/movers", params),

  leaderboard: (params: {
    horizon_days: number;
    asset_class: AssetClass;
    metric?: string;
  }) => request<Leaderboard>("/models/leaderboard", params),
};
