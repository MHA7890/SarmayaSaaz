"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Boxes, Cpu, GitBranch, ShieldCheck } from "lucide-react";

import { api } from "@/lib/api/client";
import { formatInt } from "@/lib/format";

const ENGINES = [
  {
    name: "Commodities",
    universe: "6 assets: gold, silver, copper, WTI crude, natural gas, wheat",
    grouping: "Per asset (no clustering)",
    models: "XGBoost, LightGBM, CatBoost, RandomForest, LSTM, GRU, Transformer, N-BEATS, TFT",
    ensemble: "Top 3 by directional accuracy, accuracy weighted",
    bands: "Accuracy weighted validation MAE, in absolute price units",
  },
  {
    name: "Crypto",
    universe: "23 served of 29: six withheld for stale price history",
    grouping: "4 K-Means clusters, pinned at training time",
    models: "Same nine architectures, with lite N-BEATS and TFT variants",
    ensemble: "All models above a 0.4 validated win rate floor, win rate weighted",
    bands: "True min/max dispersion across the models that voted",
  },
  {
    name: "PSX Equities",
    universe: "97 Pakistan Stock Exchange tickers",
    grouping: "7 super sectors",
    models: "XGBoost, LightGBM, LSTM",
    ensemble: "Single winning model per sector/horizon",
    bands: "Winning model's validation MAE, as a fraction of price",
  },
  {
    name: "MUFAP Mutual Funds",
    universe: "198 funds that cleared a 200-day age filter",
    grouping: "5 super clusters by underlying asset class",
    models: "XGBoost, LightGBM, LSTM",
    ensemble: "Single winning model per cluster/horizon",
    bands: "Winning model's validation MAE, as a fraction of NAV",
  },
];

export default function MethodologyPage() {
  const { data: stats } = useQuery({ queryKey: ["stats"], queryFn: api.stats });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-bold">Methodology</h1>
        <p className="mt-1 text-sm leading-relaxed text-dim">
          How SarmayaSaaz produces a forecast, and just as importantly, what it does not
          claim to know.
          {stats &&
            ` ${formatInt(stats.artifacts_on_disk)} trained artifacts currently on disk across ${stats.total_assets} assets.`}
        </p>
      </div>

      <section className="card-pad">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <GitBranch size={15} className="text-accent" />
          Direct multi horizon forecasting
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Each horizon has its own independently trained model. A 120 day forecast is not
          a 7 day forecast extrapolated forward. It is a separate model fitted against a{" "}
          <code className="num rounded bg-surface-high px-1 text-xs">shift(-120)</code>{" "}
          target. Horizons are 7, 14, 28, 42, 60, 90 and 120 days.
        </p>
      </section>

      <section className="card-pad">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <ShieldCheck size={15} className="text-accent" />
          Leakage prevention
        </h2>
        <ul className="mt-2 space-y-1.5 text-sm leading-relaxed text-muted">
          <li>
            <strong className="text-text">Purged chronological splits.</strong> Train,
            validation and test are separated in time with an embargo gap, so overlapping
            target windows cannot leak across the boundary.
          </li>
          <li>
            <strong className="text-text">Stationary features only.</strong> Raw prices and
            volumes are dropped before training; models see ratios, log returns and
            normalised indicators.
          </li>
          <li>
            <strong className="text-text">Frozen scalers.</strong> Each group&rsquo;s scaler
            is fitted on training data alone and serialized, then reused verbatim at
            inference.
          </li>
        </ul>
      </section>

      <section className="card-pad">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <Boxes size={15} className="text-accent" />
          The four engines
        </h2>
        <div className="mt-3 space-y-3">
          {ENGINES.map((e) => (
            <div key={e.name} className="inset p-4">
              <h3 className="text-[13px] font-semibold">{e.name}</h3>
              <dl className="mt-2 grid gap-x-4 gap-y-1.5 text-xs sm:grid-cols-[130px_1fr]">
                {[
                  ["Universe", e.universe],
                  ["Grouping", e.grouping],
                  ["Architectures", e.models],
                  ["Ensemble", e.ensemble],
                  ["Confidence band", e.bands],
                ].map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="label pt-0.5">{k}</dt>
                    <dd className="text-muted">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>
      </section>

      <section className="card-pad">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <Cpu size={15} className="text-accent" />
          Explainability
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Where the leading model in an ensemble is tree based, SHAP (Shapley Additive
          exPlanations) attributes the prediction to individual features, and those
          contributions are shown directly. Neural networks are not attributed. No
          surrogate explanation is substituted in their place, so the drivers panel is
          simply absent rather than approximated.
        </p>
      </section>

      <section className="card-pad border-warn/40">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-warn">
          <AlertTriangle size={15} />
          Known limitations
        </h2>
        <ul className="mt-2 space-y-1.5 text-sm leading-relaxed text-muted">
          <li>
            <strong className="text-text">Static weights.</strong> Models are trained
            offline. They do not adapt to a regime shift until retrained.
          </li>
          <li>
            <strong className="text-text">Mixed feature contracts.</strong> 154 commodity
            artifacts were reverted to a feature set without sentiment inputs during
            champion/challenger testing, so an ensemble can average models observing
            slightly different inputs.
          </li>
          <li>
            <strong className="text-text">Price freshness &amp; candle integrity.</strong>{" "}
            Traditional equity (PSX) and commodity feeds strictly enforce completed past sessions
            (&lt; today), dropping unclosed intraday bars during market hours to prevent forming candles
            from polluting feature matrices. Crypto assets operate 24/7 and close daily bars at 00:00 UTC.
            PSX collection includes automated TradingView fallback for high availability. Every asset displays
            its exact &ldquo;as of&rdquo; date and relative time elapsed.
          </li>
          <li>
            <strong className="text-text">Withheld assets.</strong> Six crypto datasets end
            490–1,596 days behind the rest of the universe and are excluded rather than
            forecast from stale or mislabeled history. One mutual fund with missing commodity scaling data
            is permanently excluded.
          </li>
          <li>
            <strong className="text-text">Point estimates.</strong> Bands are derived from
            historical error, not quantile regression, so they describe typical past error
            rather than a calibrated predictive interval.
          </li>
        </ul>
      </section>
    </div>
  );
}
