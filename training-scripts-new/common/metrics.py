"""
Shared metric computation - identical definitions to the original pipelines
(src/commodities/src/generators/create_*_nb_*.py `calculate_metrics`, and the
MAE-only scoring used by src/stocks, src/mufap, src/crypto stage4 scripts).

All prediction targets across every asset class are *forward returns*
(`(future - now) / now`), so every metric here is computed the same way:
reconstruct the predicted price from the predicted return, then compare
price-to-price. This keeps MAE in the asset's native price units (USD/oz,
PKR/share, PKR/unit, USD/token) - directly usable as a confidence band, and
comparable to what the existing engines already expect.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def price_metrics(current_price: np.ndarray, true_return: np.ndarray, pred_return: np.ndarray) -> dict:
    """
    current_price: price at prediction time (t0), one per sample
    true_return:   actual forward return realised at t0+h
    pred_return:   model's predicted forward return

    Mirrors calculate_metrics() from the commodities notebook generators:
    MAE/RMSE/MAPE/R2 computed on reconstructed price, Dir_Acc on the sign of
    the move relative to t0, plus improvement over a naive "no change" guess.
    """
    true_price = current_price * (1.0 + true_return)
    pred_price = current_price * (1.0 + pred_return)

    mae = float(mean_absolute_error(true_price, pred_price))
    rmse = float(np.sqrt(mean_squared_error(true_price, pred_price)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((true_price - pred_price) / np.where(true_price == 0, np.nan, true_price)))) * 100
        mape = 0.0 if not np.isfinite(mape) else mape
    r2 = float(r2_score(true_price, pred_price)) if len(true_price) > 1 else 0.0

    actual_dir = np.sign(true_price - current_price)
    pred_dir = np.sign(pred_price - current_price)
    dir_acc = float(np.mean(actual_dir == pred_dir)) * 100.0

    naive_mae = float(mean_absolute_error(true_price, current_price))
    improvement_pct = 0.0 if naive_mae == 0 else (naive_mae - mae) / naive_mae * 100.0

    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "MAPE": round(mape, 4),
        "Dir_Acc": round(dir_acc, 4),
        "R2": round(r2, 6),
        "Naive_MAE": round(naive_mae, 6),
        "Improvement_Pct": round(improvement_pct, 4),
    }


def return_mae(true_return: np.ndarray, pred_return: np.ndarray) -> float:
    """Plain MAE on the return itself (used where the original pipeline
    scored on returns directly, e.g. crypto/stocks/mufap win-rate selection,
    rather than reconstructing price)."""
    return float(mean_absolute_error(true_return, pred_return))


def win_rate_from_mae(mae: float, floor: float = 0.4) -> float:
    """Crypto ensemble weight: 1 - MAE, floored - see stage4_master_training.py."""
    return round(max(floor, 1.0 - mae), 4)
