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


def adjust_splits(
    df: pd.DataFrame,
    *,
    price_cols: tuple[str, ...] = ("Close", "NAV"),
    min_drop_ratio: float = 0.65,
    min_jump_ratio: float = 1.50,
) -> pd.DataFrame:
    """
    Detects unadjusted corporate actions (stock splits, bonus shares, reverse splits)
    and retroactively adjusts past prices so that historical charts and features are smooth.
    """
    col = next((c for c in price_cols if c in df.columns), None)
    if col is None or df.empty or len(df) < 10:
        return df

    df = df.copy()
    changed = True
    while changed:
        changed = False
        prices = df[col].values
        n = len(prices)
        for i in range(1, n):
            p_prev = prices[i - 1]
            p_curr = prices[i]
            if p_prev <= 0 or p_curr <= 0 or np.isnan(p_prev) or np.isnan(p_curr):
                continue

            ratio = p_curr / p_prev

            # Stock split / bonus shares (e.g. 5-for-1 split drops price by ~80%, ratio <= 0.65)
            if ratio <= min_drop_ratio:
                future = prices[i : min(i + 10, n)]
                if len(future) >= 3 and np.mean(future) < p_prev * 0.75:
                    for c in ["Open", "High", "Low", "Close", "NAV"]:
                        if c in df.columns:
                            df.iloc[:i, df.columns.get_loc(c)] *= ratio
                    if "Volume" in df.columns:
                        df.iloc[:i, df.columns.get_loc("Volume")] /= ratio
                    changed = True
                    break

            # Reverse split (e.g. 1-for-2 reverse split doubles price, ratio >= 1.50)
            elif ratio >= min_jump_ratio:
                future = prices[i : min(i + 10, n)]
                if len(future) >= 3 and np.mean(future) > p_prev * 1.35:
                    for c in ["Open", "High", "Low", "Close", "NAV"]:
                        if c in df.columns:
                            df.iloc[:i, df.columns.get_loc(c)] *= ratio
                    if "Volume" in df.columns:
                        df.iloc[:i, df.columns.get_loc("Volume")] /= ratio
                    changed = True
                    break

    return df


def clean_ohlcv(
    df: pd.DataFrame,
    *,
    return_threshold: float,
    price_cols=("Open", "High", "Low", "Close"),
    adjust_stock_splits: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Null removal + outlier removal for an OHLC(V) frame indexed by date.

    - drops rows with any null in the price columns (or Volume, if present)
    - drops rows with non-positive prices
    - drops rows that violate basic OHLC consistency (High is not the max,
      Low is not the min)
    - optionally retroactively adjusts corporate action splits/bonus shares
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

    if adjust_stock_splits:
        df = adjust_splits(df, price_cols=price_cols)

    if all(c in df.columns for c in ("High", "Low", "Open", "Close")):
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


def clean_price_series(
    df: pd.DataFrame,
    price_col: str,
    *,
    return_threshold: float,
    adjust_stock_splits: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Null/outlier removal for a single-price series (e.g. mutual fund NAV)."""
    before = len(df)
    df = df.dropna(subset=[price_col])
    df = df[df[price_col] > 0]
    df = df.sort_index()
    if adjust_stock_splits:
        df = adjust_splits(df, price_cols=(price_col,))
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


def read_existing_csv(path: Path) -> pd.DataFrame:
    """Read an existing CSV file as a Date-indexed DataFrame if it exists."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def trim_tail(df: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
    """Trim the last n_rows from a DataFrame as a safety buffer for incremental updates."""
    if df.empty or len(df) <= n_rows:
        return pd.DataFrame()
    return df.iloc[:-n_rows].copy()


def merge_incremental(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge old existing DataFrame with newly fetched DataFrame.
    New rows overwrite old rows on date overlap.
    """
    if old_df.empty:
        return new_df.sort_index() if not new_df.empty else new_df
    if new_df.empty:
        return old_df.sort_index()

    combined = pd.concat([old_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def is_already_current(
    df: pd.DataFrame,
    *,
    tz: str = "Asia/Karachi",
    session_close: str | None = None,
) -> bool:
    """
    Check if existing DataFrame already has the latest expected closed trading session date.
    Returns True if data is up to date, avoiding redundant network calls.
    """
    if df.empty:
        return False

    now = pd.Timestamp.now(tz=tz)
    # Determine the most recent closed trading day date
    if now.weekday() == 5:  # Saturday -> Friday
        expected = (now - pd.Timedelta(days=1)).normalize().tz_localize(None)
    elif now.weekday() == 6:  # Sunday -> Friday
        expected = (now - pd.Timedelta(days=2)).normalize().tz_localize(None)
    else:
        today = now.normalize().tz_localize(None)
        if session_close is not None:
            hh, mm = (int(x) for x in session_close.split(":"))
            closes_at = now.normalize() + pd.Timedelta(hours=hh, minutes=mm)
            if now < closes_at:
                expected = today - pd.Timedelta(days=3 if now.weekday() == 0 else 1)
            else:
                expected = today
        else:
            expected = today - pd.Timedelta(days=3 if now.weekday() == 0 else 1)

    max_date = df.index.max().normalize()
    if max_date.tz is not None:
        max_date = max_date.tz_localize(None)

    return max_date >= expected


