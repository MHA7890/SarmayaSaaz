"""
Shared helpers for the data-new/ collection scripts.

Each collector (crypto/psx/mufap/commodities) pulls raw daily series from its
one designated source, then runs it through here for a consistent null/outlier
policy before writing CSVs into data-new/<asset-class>-data/.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_NEW = PROJECT_ROOT / "data-new"

# How far a bar's High/Low may contradict its Open/Close, as a fraction of
# the close, before the bar is discarded instead of clamped. 0.05% is far
# below any meaningful price move and well above tick-rounding noise.
OHLC_CLAMP_TOLERANCE = 0.0005

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def clean_ohlcv(df: pd.DataFrame, *, return_threshold: float, price_cols=("Open", "High", "Low", "Close")) -> tuple[pd.DataFrame, int]:
    """Null removal + outlier removal for an OHLC(V) frame indexed by date.

    - drops rows with any null in the price columns (or Volume, if present)
    - drops rows with non-positive prices
    - drops rows that violate basic OHLC consistency (High is not the max,
      Low is not the min)
    - drops rows whose single-day Close-to-Close return exceeds
      `return_threshold` in magnitude (data glitches / bad prints), which is
      a source-specific tolerance since PSX has circuit breakers and crypto
      does not.
    """
    cols = [c for c in price_cols if c in df.columns]
    if "Volume" in df.columns:
        cols = cols + ["Volume"]

    before = len(df)
    df = df.dropna(subset=cols)

    for c in [c for c in price_cols if c in df.columns]:
        df = df[df[c] > 0]

    if all(c in df.columns for c in ("High", "Low", "Open", "Close")):
        # OHLC consistency. A bar whose High is below max(Open, Close), or Low
        # above min(Open, Close), is self-contradictory and normally dropped.
        #
        # But two different things produce that shape. PSX publishes genuine
        # tick-rounding artifacts on thin days - IBFL 2026-08-19 closed at
        # 265.03 with a Low of 265.05, over by 0.02, which is 0.008% of the
        # close on 43 shares traded. It also publishes genuinely broken prints:
        # across IBFL's history the violations run to 104% of the close.
        # Dropping the whole session treats both the same and silently loses a
        # real trading day, which is how one ticker ends up stuck a day behind
        # the rest of the market.
        #
        # So: violations within OHLC_CLAMP_TOLERANCE of the close are clamped -
        # High/Low widened to contain Open and Close, which preserves the close
        # and makes the bar consistent - and anything larger is still dropped.
        # For IBFL that admits the 12 rounding rows and rejects the other 214.
        row_max = df[["Open", "Close"]].max(axis=1)
        row_min = df[["Open", "Close"]].min(axis=1)

        tol = df["Close"].abs() * OHLC_CLAMP_TOLERANCE
        hi_off = row_max - df["High"]          # > 0 when High sits too low
        lo_off = df["Low"] - row_min           # > 0 when Low sits too high
        clampable = ((hi_off <= tol) & (lo_off <= tol))

        df = df.copy()
        df.loc[clampable, "High"] = df.loc[clampable, ["High"]].join(row_max.rename("m")).max(axis=1)
        df.loc[clampable, "Low"] = df.loc[clampable, ["Low"]].join(row_min.rename("m")).min(axis=1)

        row_max = df[["Open", "Close"]].max(axis=1)
        row_min = df[["Open", "Close"]].min(axis=1)
        consistent = (df["High"] >= row_max) & (df["Low"] <= row_min) & (df["High"] >= df["Low"])
        df = df[consistent]

    if "Close" in df.columns:
        df = df.sort_index()
        ret = df["Close"].pct_change()
        df = df[(ret.abs() <= return_threshold) | ret.isna()]

    removed = before - len(df)
    return df, removed


def clean_price_series(df: pd.DataFrame, price_col: str, *, return_threshold: float) -> tuple[pd.DataFrame, int]:
    """Null/outlier removal for a single-price series (e.g. mutual fund NAV)."""
    before = len(df)
    df = df.dropna(subset=[price_col])
    df = df[df[price_col] > 0]
    df = df.sort_index()
    ret = df[price_col].pct_change()
    df = df[(ret.abs() <= return_threshold) | ret.isna()]
    removed = before - len(df)
    return df, removed


def drop_unclosed_sessions(
    df: pd.DataFrame,
    *,
    tz: str,
    session_close: str | None = None,
    buffer_minutes: int = 30,
) -> tuple[pd.DataFrame, int]:
    """
    Drop trailing rows for sessions that have not finished yet.

    A source will happily hand back a row for a session still in progress.
    Verified live: PSX's DPS endpoint queried at 11:25 PKT with the market open
    returned HBL dated that same day whose "Close" was the live price (322.25)
    on a fraction of the usual volume (112k against a typical 600k-1.8M), and
    six minutes later the same "close" read 322.01. Writing that as the day's
    close silently poisons the series and the feature frames built from it.

    `tz` is the timezone the source's dates are expressed in - Asia/Karachi for
    PSX and MUFAP, UTC for the 24h crypto and commodity feeds.

    `session_close` ("HH:MM", in `tz`) is when today's session actually ends.
    Given it, today's row is kept once that time has passed plus
    `buffer_minutes`, because the session is genuinely over and the close is
    final - which is what lets a collector run right after the closing bell
    publish the same day instead of waiting for tomorrow.

    Left as None, the rule is the conservative one: only sessions dated
    strictly before today are final. That is the right default for feeds whose
    trading day ends after local midnight - a crypto UTC day closes at 00:00
    UTC the following day, so "today" is never final in UTC terms.
    """
    if df.empty:
        return df, 0

    now = pd.Timestamp.now(tz=tz)
    today = now.normalize().tz_localize(None)

    cutoff = today  # default: everything dated today or later is unfinished
    if session_close is not None:
        hh, mm = (int(x) for x in session_close.split(":"))
        closes_at = (
            now.normalize()
            + pd.Timedelta(hours=hh, minutes=mm)
            + pd.Timedelta(minutes=buffer_minutes)
        )
        if now >= closes_at:
            cutoff = today + pd.Timedelta(days=1)  # today's session is over; keep it

    keep = df.index < cutoff
    return df[keep], int((~keep).sum())


def clip_last_n_years(df: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = df.index.max() - pd.DateOffset(years=years)
    return df[df.index >= cutoff]


def safe_filename(name: str) -> str:
    return "".join(c for c in name if c not in '<>:"/\\|?*').strip()
