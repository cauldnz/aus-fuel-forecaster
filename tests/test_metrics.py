"""Tests for evaluate.metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fuel_pred.evaluate import metrics as m

# ----------------------------- happy-path tests -----------------------------


def test_mae_zero_when_perfect() -> None:
    y = [180.0, 181.5, 179.0, 182.0]
    assert m.mae(y, y) == 0.0


def test_mae_known_value() -> None:
    """Hand-computed: mean(|10|, |20|) = 15."""
    assert m.mae([100.0, 200.0], [110.0, 220.0]) == 15.0


def test_rmse_known_value() -> None:
    """Hand-computed: sqrt(mean(100, 400)) = sqrt(250)."""
    assert m.rmse([100.0, 200.0], [110.0, 220.0]) == pytest.approx(250 ** 0.5)


def test_mape_known_value() -> None:
    """|10/100| + |20/200| = 0.1 + 0.1 -> mean 0.1 -> 10.0%."""
    assert m.mape([100.0, 200.0], [110.0, 220.0]) == pytest.approx(10.0)


def test_median_absolute_error_pulls_centre() -> None:
    """Median ignores the one large outlier — that's the whole point of pairing it with MAE."""
    err = [1.0, 2.0, 3.0, 100.0]
    assert m.median_absolute_error([0.0] * 4, err) == pytest.approx(2.5)


def test_p90_is_higher_than_median() -> None:
    """Sanity: q=0.9 >= q=0.5 by construction, except in pathological tied cases."""
    rng = np.random.default_rng(42)
    y_true = rng.normal(180.0, 10.0, size=500)
    y_pred = y_true + rng.normal(0.0, 5.0, size=500)
    assert m.quantile_absolute_error(y_true, y_pred, q=0.9) >= m.median_absolute_error(
        y_true, y_pred
    )


def test_quantile_zero_is_min_quantile_one_is_max() -> None:
    """Boundary check on q."""
    errs_signed = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_true = [0.0] * 5
    assert m.quantile_absolute_error(y_true, errs_signed, q=0.0) == 1.0
    assert m.quantile_absolute_error(y_true, errs_signed, q=1.0) == 5.0


# ----------------------------- input handling -----------------------------


@pytest.mark.parametrize(
    "container",
    [
        list,
        tuple,
        np.array,
        pd.Series,
    ],
)
def test_accepts_common_array_likes(container: type) -> None:
    """numpy arrays, pandas Series, lists, tuples — all should work."""
    yt = container([100.0, 200.0])
    yp = container([110.0, 220.0])
    assert m.mae(yt, yp) == 15.0


def test_drops_pairwise_nans_silently() -> None:
    """The eval harness routinely scores partial folds — silent NaN drop, not raise."""
    yt = [100.0, 200.0, np.nan, 300.0]
    yp = [110.0, np.nan, 250.0, 330.0]
    # Only rows 0 and 3 are pairwise non-NaN: |10|, |30| -> mean 20.
    assert m.mae(yt, yp) == 20.0


def test_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        m.mae([1.0, 2.0, 3.0], [1.0, 2.0])


def test_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        m.mae([], [])


def test_raises_on_all_nan_pairs() -> None:
    with pytest.raises(ValueError, match="no valid"):
        m.mae([np.nan, np.nan], [1.0, np.nan])


# ----------------------------- MAPE-specific -----------------------------


def test_mape_handles_zeros_in_y_true() -> None:
    """y_true == 0 rows are silently dropped; the rest compute normally."""
    yt = [100.0, 0.0, 200.0]
    yp = [110.0, 50.0, 220.0]
    # Middle row drops; |10/100| + |20/200| -> mean 0.1 -> 10%.
    assert m.mape(yt, yp) == pytest.approx(10.0)


def test_mape_raises_when_all_y_true_zero() -> None:
    with pytest.raises(ValueError, match="every y_true is zero"):
        m.mape([0.0, 0.0], [10.0, 20.0])


# ----------------------------- quantile validation -----------------------------


@pytest.mark.parametrize("bad_q", [-0.1, 1.1, 2.0, -1.0])
def test_quantile_q_must_be_in_unit_interval(bad_q: float) -> None:
    with pytest.raises(ValueError, match=r"q must be in \[0, 1\]"):
        m.quantile_absolute_error([1.0, 2.0], [1.0, 2.0], q=bad_q)


# ----------------------------- one-shot convenience -----------------------------


def test_all_metrics_returns_full_spec_set() -> None:
    """Names + ordering exactly match what the comparison-report writer needs."""
    yt = [100.0, 200.0, 150.0, 175.0]
    yp = [110.0, 195.0, 152.0, 180.0]
    out = m.all_metrics(yt, yp)
    assert set(out.keys()) == {
        "mae",
        "rmse",
        "mape",
        "median_abs_error",
        "p90_abs_error",
    }
    # Spot-check one value matches the standalone computation.
    assert out["mae"] == m.mae(yt, yp)


def test_all_metrics_floats_throughout() -> None:
    """No numpy scalars leaking into the output dict — keeps Markdown formatting clean."""
    yt = [100.0, 200.0]
    yp = [110.0, 220.0]
    out = m.all_metrics(yt, yp)
    for k, v in out.items():
        assert isinstance(v, float), f"{k} is {type(v).__name__}, want float"
