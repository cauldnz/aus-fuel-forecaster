"""Regression metrics for fuel-price prediction.

Per spec §8.5, every test-fold report carries: MAE, RMSE, MAPE,
median absolute error, and 90th-percentile absolute error. Both
models are evaluated on the *same* held-out rows so these metrics
are directly comparable.

Each function:
- Accepts any array-like (numpy, pandas, list).
- Coerces to ``np.float64`` and aligns lengths.
- Drops paired rows where either side is NaN — silent dropping rather
  than raising, because the eval harness routinely scores partial
  test folds (e.g. when a station has missing days).
- Raises ``ValueError`` if no valid rows remain.

Outputs are plain floats. Cents-per-litre for MAE/RMSE/median/p90,
percent for MAPE.

Spec: spec.md §8.5.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _aligned_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to float arrays, validate same length, drop pair-wise NaNs."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {yt.shape} vs {yp.shape}"
        )
    if yt.size == 0:
        raise ValueError("y_true / y_pred are empty")
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        raise ValueError("no valid (finite) row pairs after dropping NaNs")
    return yt[mask], yp[mask]


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error. Same units as the input prices (cents/L)."""
    yt, yp = _aligned_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error. Same units as the input prices (cents/L).

    Penalises large errors more than MAE — useful as a secondary metric
    even though our LightGBM objective is L1 (MAE-aligned per spec §8.2).
    """
    yt, yp = _aligned_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error, expressed as a percent (5.0 means 5%).

    Rows where ``y_true == 0`` are dropped — division by zero is
    undefined, and a retail price of zero shouldn't appear in real
    fuel data anyway (it's a data-quality outlier). If every y_true
    is zero, raises ``ValueError`` rather than silently returning NaN.
    """
    yt, yp = _aligned_arrays(y_true, y_pred)
    nonzero = yt != 0
    if not nonzero.any():
        raise ValueError("MAPE: every y_true is zero")
    yt, yp = yt[nonzero], yp[nonzero]
    return float(np.mean(np.abs((yt - yp) / yt)) * 100)


def median_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Median of |y_true - y_pred|. Robust to outliers; pairs with MAE."""
    yt, yp = _aligned_arrays(y_true, y_pred)
    return float(np.median(np.abs(yt - yp)))


def quantile_absolute_error(
    y_true: ArrayLike, y_pred: ArrayLike, q: float = 0.9
) -> float:
    """q-th percentile of the absolute errors (default p90 per spec §8.5).

    p90 is the spec's tail-risk metric — the worst 10% of errors are
    where crisis-period stress shows up. p99 / p99.9 are also useful
    once we have enough rows; ``q`` is a kwarg so eval reports can
    quote multiple quantiles from the same call site.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0, 1]; got {q!r}")
    yt, yp = _aligned_arrays(y_true, y_pred)
    return float(np.quantile(np.abs(yt - yp), q))


def all_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """One-shot convenience: return the full §8.5 metric set as a dict.

    Useful for the comparison-report writer, which builds Markdown tables
    of (model, fold) → metrics. Quantile defaults to p90 per spec.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "median_abs_error": median_absolute_error(y_true, y_pred),
        "p90_abs_error": quantile_absolute_error(y_true, y_pred, q=0.9),
    }
