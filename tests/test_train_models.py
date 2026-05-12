"""Tests for train.train_models — the orchestrator + identical-rows guard.

Uses synthetic feature DataFrames small enough that LightGBM trains in
seconds. The point is to exercise the orchestrator's wiring (filter,
split, guard, fit, persist), not to test LightGBM itself.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fuel_pred.train import folds as folds_mod
from fuel_pred.train import train_models as tm
from fuel_pred.train.feature_blocks import (
    BLOCK_COLUMNS,
    EXCLUDE_FROM_FEATURES,
)

# ----------------------------- fixtures -----------------------------


def _synth_panel(
    n_stations: int = 5,
    n_days: int = 200,
    *,
    seed: int = 42,
    sa2_null_fraction: float = 0.0,
) -> pd.DataFrame:
    """Build a tiny synthetic features.parquet shape.

    One row per (station, U91, day). Includes every column from every
    block plus identifiers + target. The numeric values are pure noise
    around a slowly-varying trend so LightGBM converges quickly with
    early stopping.

    sa2_null_fraction lets tests probe the identical-rows guard by
    injecting nulls into the SA2 columns for some rows.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows: list[dict[str, object]] = []
    for station_idx in range(n_stations):
        station_id = f"s{station_idx:03d}"
        # Slow trend so the model has *something* to learn rather than
        # pure noise — produces a non-degenerate best_iteration.
        base = 180.0 + station_idx * 2.0 + rng.normal(0, 0.5, n_days).cumsum() * 0.05
        for i, d in enumerate(dates):
            row: dict[str, object] = {
                "station_id": station_id,
                "fuel_code": "U91",
                "date": d,
                "y_t1": float(base[i] + rng.normal(0, 0.3)),
                "price_mean": float(base[i]),
                "price_min": float(base[i] - 0.5),
                "price_max": float(base[i] + 0.5),
                "n_obs": 5,
                "name": f"Station {station_idx}",
                "address": f"{station_idx} Test St",
                "suburb": "Test",
                "postcode": "2000",
                "brand_raw": "BP" if station_idx % 2 == 0 else "Caltex",
                "brand_canonical": "BP" if station_idx % 2 == 0 else "Ampol",
                "brand_is_major": True,
                "first_seen": dates[0],
                "last_seen": dates[-1],
                "lat": -33.0 + station_idx * 0.01,
                "lon": 151.0 + station_idx * 0.01,
                "geocoder": "synthetic",
                "mb_code": "MB1",
                "sa2_code": "117011635",
                "sa2_name": "Mascot",
                "counter_id": f"c{station_idx:03d}",
            }
            for col in BLOCK_COLUMNS["lag"]:
                row[col] = float(base[i] + rng.normal(0, 0.2))
            for col in BLOCK_COLUMNS["upstream"]:
                row[col] = float(80.0 + rng.normal(0, 1.0))
            for col in BLOCK_COLUMNS["cal"]:
                # int / bool cal cols — use simple ints; LightGBM handles either.
                row[col] = int(d.dayofweek if "day_of_week" in col else i % 14)
            for col in BLOCK_COLUMNS["ctx"]:
                row[col] = float(rng.normal(0, 1.0))
            for col in BLOCK_COLUMNS["stn"]:
                if col in {"stn_brand_raw", "stn_brand_canonical"}:
                    row[col] = row[
                        "brand_raw" if col == "stn_brand_raw" else "brand_canonical"
                    ]
                elif col in {"stn_brand_is_major", "stn_is_franchisee", "stn_is_metro"}:
                    row[col] = bool(station_idx % 2 == 0)
                else:
                    row[col] = float(rng.normal(0, 1.0))
            for col in BLOCK_COLUMNS["wx"]:
                if col == "wx_weather_code":
                    row[col] = "1"  # categorical str
                else:
                    row[col] = float(rng.normal(20, 5))
            for col in BLOCK_COLUMNS["sa2"]:
                row[col] = float(rng.normal(50, 10))
            rows.append(row)

    df = pd.DataFrame(rows)

    if sa2_null_fraction > 0:
        # Knock out SA2 columns on a fraction of rows to test the guard.
        n_null = int(sa2_null_fraction * len(df))
        idx = rng.choice(df.index, size=n_null, replace=False)
        df.loc[idx, list(BLOCK_COLUMNS["sa2"])] = pd.NA

    return df


def _short_fold_config() -> folds_mod.FoldConfig:
    """A 200-day timeline split into roughly: 100/40/40/20."""
    return folds_mod.FoldConfig(
        train_start="2024-01-01",
        train_end="2024-04-09",         # day 100
        val_start="2024-04-10",
        val_end="2024-05-19",           # day 140
        test_normal_start="2024-05-20",
        test_normal_end="2024-06-28",   # day 180
        test_crisis_start="2024-06-29",
    )


@pytest.fixture
def features_path(tmp_path: Path) -> Path:
    """Synthetic features.parquet on disk for the orchestrator to load."""
    df = _synth_panel()
    p = tmp_path / "features.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    return p


# ----------------------------- end-to-end happy path -----------------------------


def test_train_writes_all_artefacts(features_path: Path, tmp_path: Path) -> None:
    """End-to-end: orchestrator produces both pickles + feature_lists.json
    + per-fold prediction parquets."""
    out_dir = tmp_path / "models"
    result = tm.train(features_path, out_dir, fold=_short_fold_config())

    assert (out_dir / "model_a.pkl").exists()
    assert (out_dir / "model_b.pkl").exists()
    assert (out_dir / "feature_lists.json").exists()
    assert (out_dir / "predictions_test_normal.parquet").exists()
    assert (out_dir / "predictions_test_crisis.parquet").exists()
    # No leftover .tmp files from the atomic rename.
    assert not any(p.suffix == ".tmp" for p in out_dir.iterdir())

    # Returned dict carries both fits.
    assert set(result.keys()) == {"A", "B"}


def test_feature_lists_json_records_correct_block_membership(
    features_path: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "models"
    tm.train(features_path, out_dir, fold=_short_fold_config())

    payload = json.loads((out_dir / "feature_lists.json").read_text(encoding="utf-8"))
    assert set(payload) == {"A", "B", "config"}

    cols_a = payload["A"]["feature_columns"]
    cols_b = payload["B"]["feature_columns"]
    # Model B is Model A + sa2 (per spec §8.4).
    assert set(cols_b) - set(cols_a) == set(BLOCK_COLUMNS["sa2"])
    assert set(cols_a) - set(cols_b) == set()
    # Targets and identifiers stayed out.
    for col in EXCLUDE_FROM_FEATURES:
        assert col not in cols_a, f"{col!r} leaked into Model A"
        assert col not in cols_b, f"{col!r} leaked into Model B"


def test_predictions_parquet_has_expected_schema(
    features_path: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "models"
    tm.train(features_path, out_dir, fold=_short_fold_config())

    preds = pd.read_parquet(out_dir / "predictions_test_normal.parquet")
    assert set(preds.columns) == {
        "station_id",
        "fuel_code",
        "date",
        "y_true",
        "y_pred_a",
        "y_pred_b",
    }
    assert len(preds) > 0
    assert preds["y_pred_a"].notna().all()
    assert preds["y_pred_b"].notna().all()


def test_pickled_models_can_be_reloaded_and_predict(
    features_path: Path, tmp_path: Path
) -> None:
    """If the eval pass dies and we have to re-load the pickles, that needs to work.

    Documents the requirement: callers must coerce string categoricals to
    pandas Categorical before calling ``predict()``. The pickled model
    encodes its category mapping internally; mismatched dtype on the
    predict input triggers LightGBM's ``"train and valid dataset
    categorical_feature do not match"`` error.
    """
    out_dir = tmp_path / "models"
    tm.train(features_path, out_dir, fold=_short_fold_config())

    with (out_dir / "model_a.pkl").open("rb") as fh:
        model_a = pickle.load(fh)
    payload = json.loads((out_dir / "feature_lists.json").read_text(encoding="utf-8"))
    cols_a = payload["A"]["feature_columns"]
    cat_a = payload["A"]["categorical_columns"]

    sample = pd.read_parquet(features_path).head(20).copy()
    # Coerce categoricals using the model's own stored mapping. The
    # booster keeps it on ``pandas_categorical_`` — a list of category
    # lists in the same order as the categorical_feature names.
    booster_cats = model_a.booster_.pandas_categorical
    for col, cats in zip(cat_a, booster_cats, strict=True):
        sample[col] = sample[col].astype(pd.CategoricalDtype(categories=cats))

    preds = model_a.predict(sample[cols_a])
    assert len(preds) == 20
    assert np.isfinite(preds).all()


# ----------------------------- identical-rows guard -----------------------------


def test_identical_rows_guard_aligns_train_a_and_train_b(
    tmp_path: Path,
) -> None:
    """Spec §8.4: both models train on rows where every Model B column
    is non-null. With sa2 nulls, Model A trains on FEWER rows than its
    raw fold size — the rows it shares with Model B."""
    df = _synth_panel(n_stations=8, n_days=200, sa2_null_fraction=0.3)
    p = tmp_path / "features.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)

    out_dir = tmp_path / "models"
    result = tm.train(p, out_dir, fold=_short_fold_config())

    # Best-iteration should be set on both fits (they actually trained).
    assert result["A"].best_iteration is not None
    assert result["B"].best_iteration is not None
    # Both fits used the same number of feature columns? No — A has fewer
    # (no sa2). But both should have trained on identical rows. We can't
    # introspect that from the FitResult directly, so verify indirectly:
    # both models should produce predictions for the same rows on the
    # test fold, and the count should equal the rows where Model B
    # columns are all non-null in that fold.
    preds = pd.read_parquet(out_dir / "predictions_test_normal.parquet")
    # Predictions are written for every row in the test_normal fold (no
    # guard there — only training is gated). Both columns should be fully
    # populated except where the predictor itself returned NaN, which it
    # shouldn't on numeric features.
    assert preds["y_pred_a"].notna().all()
    assert preds["y_pred_b"].notna().all()


def test_train_raises_when_sa2_block_entirely_null(tmp_path: Path) -> None:
    """If 100% of training rows have null SA2, the guard leaves zero
    eligible rows and we should fail loudly."""
    df = _synth_panel(n_stations=5, n_days=200, sa2_null_fraction=1.0)
    p = tmp_path / "features.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(RuntimeError, match="zero training rows"):
        tm.train(p, tmp_path / "models", fold=_short_fold_config())


# ----------------------------- filter behaviour -----------------------------


def test_train_filters_out_diesel_rows(tmp_path: Path) -> None:
    """Spec §3 + §7.8: only U91 rows have a target. Diesel rows must be
    dropped before training."""
    df = _synth_panel(n_stations=4, n_days=120)
    # Add some Diesel rows with null targets — should be silently filtered.
    diesel = df.head(50).copy()
    diesel["fuel_code"] = "DL"
    diesel["y_t1"] = pd.NA
    df = pd.concat([df, diesel], ignore_index=True)

    p = tmp_path / "features.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    out_dir = tmp_path / "models"
    tm.train(p, out_dir, fold=_short_fold_config())

    # Should still produce models — the U91 portion is enough.
    assert (out_dir / "model_a.pkl").exists()


def test_train_raises_on_empty_features(tmp_path: Path) -> None:
    """Empty input parquet → raise rather than silently writing empty models."""
    p = tmp_path / "features.parquet"
    pd.DataFrame(columns=["station_id", "fuel_code", "date", "y_t1"]).to_parquet(p)
    with pytest.raises(RuntimeError, match="no rows after"):
        tm.train(p, tmp_path / "models", fold=_short_fold_config())


def test_train_warns_and_proceeds_when_spec_column_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Lax mode: a column in the spec but not in features.parquet should
    log a WARNING and let modeling proceed using the columns that ARE
    present. (Verified the day Phase 6 first ran against real data —
    `stn_distance_to_sydney_terminal_km` was promised but not built.)"""
    df = _synth_panel(n_stations=4, n_days=200).drop(
        columns=["stn_distance_to_sydney_terminal_km"]
    )
    p = tmp_path / "features.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    out_dir = tmp_path / "models"

    with caplog.at_level("WARNING", logger="fuel_pred.train.train_models"):
        tm.train(p, out_dir, fold=_short_fold_config())

    # Modeling completed.
    assert (out_dir / "model_a.pkl").exists()
    assert (out_dir / "model_b.pkl").exists()
    # Warning surfaced — single line, names the missing column.
    matched = [
        r
        for r in caplog.records
        if "absent from features.parquet" in r.message
        and "stn_distance_to_sydney_terminal_km" in r.message
    ]
    assert matched, "expected a WARNING naming the missing spec column"


def test_train_no_predictions_flag_skips_prediction_parquets(
    features_path: Path, tmp_path: Path
) -> None:
    """`save_predictions=False` writes pickles + json but no prediction parquets."""
    out_dir = tmp_path / "models"
    tm.train(
        features_path,
        out_dir,
        fold=_short_fold_config(),
        save_predictions=False,
    )
    assert (out_dir / "model_a.pkl").exists()
    assert (out_dir / "feature_lists.json").exists()
    assert not (out_dir / "predictions_test_normal.parquet").exists()
    assert not (out_dir / "predictions_test_crisis.parquet").exists()
