"""
Leak-safe chronological splitting, shared by every asset class.

Every original pipeline split by TIME, never randomly, and every one used a
purge/embargo gap around the split boundary sized to the prediction horizon -
without it, a training row's target window would overlap dates that appear
in the validation/test rows, letting the model see (indirectly) what it's
being asked to predict. This module centralizes that logic so no per-asset
script can accidentally regress to a random split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame | None = None


def chronological_split(
    df: pd.DataFrame,
    horizon_days: int,
    *,
    date_col: str | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    purge: bool = True,
) -> Split:
    """
    70/10/20 chronological split (matches the commodities notebook
    generators) with a purge gap of `horizon_days` at each boundary so no
    training row's forward-looking target window bleeds into val/test, and
    no val row's target window bleeds into test.

    `df` must already be sorted by date. Pass `date_col=None` to use the
    index (crypto/mufap/commodities keep Date as the index); pass a column
    name for frames where the date is a plain column.
    """
    dates = df.index if date_col is None else df[date_col]
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    values = dates.to_numpy()  # plain array - positional indexing regardless of source (Index vs Series)

    n = len(df)
    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + val_frac))

    train_end_date = values[max(0, train_end_idx - 1)]
    val_end_date = values[max(0, val_end_idx - 1)]

    gap = pd.Timedelta(days=horizon_days) if purge else pd.Timedelta(days=0)

    train_mask = dates <= train_end_date
    val_mask = (dates > train_end_date + gap) & (dates <= val_end_date)
    test_mask = dates > val_end_date + gap

    return Split(train=df[train_mask], val=df[val_mask], test=df[test_mask])


def two_way_chronological_split(
    df: pd.DataFrame,
    horizon_days: int,
    *,
    date_col: str | None = None,
    train_frac: float = 0.8,
) -> Split:
    """
    80/20 train/val split with a purge gap (matches stocks/mufap stage4:
    those pipelines have no separate held-out test set - validation MAE
    doubles as the model-selection and reported-confidence metric).
    """
    dates = df.index if date_col is None else df[date_col]
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    values = dates.to_numpy()

    n = len(df)
    split_idx = int(n * train_frac)
    split_date = values[max(0, split_idx - 1)]
    gap = pd.Timedelta(days=horizon_days)

    train_mask = dates <= split_date
    val_mask = dates > split_date + gap

    train_df = df[train_mask]
    val_df = df[val_mask]
    if len(val_df) < 50:
        # Fallback for small universes: drop the purge gap rather than end up
        # with too few validation rows to trust the metric.
        val_df = df[dates > split_date]

    return Split(train=train_df, val=val_df)


def assert_no_overlap(split: Split) -> None:
    """Defensive check: train/val (/test) index ranges must not intersect."""
    train_idx = set(split.train.index)
    val_idx = set(split.val.index)
    assert train_idx.isdisjoint(val_idx), "train/val date overlap - leakage!"
    if split.test is not None and len(split.test):
        test_idx = set(split.test.index)
        assert train_idx.isdisjoint(test_idx), "train/test date overlap - leakage!"
        assert val_idx.isdisjoint(test_idx), "val/test date overlap - leakage!"
