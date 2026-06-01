"""Hermetic tests for the v3.0 k-fold orchestrator (train.cv).

Tests the orchestrator's behaviour (fold loop, audit file, predictions
output dirs) by stubbing the inner ``_train_one_fold``. End-to-end fits
against the real LightGBM live in the smoke milestone test (longer
runtime, gated by data availability).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fuel_pred.train.cv import KFoldRunResult, train_kfold
from fuel_pred.train.folds import KFoldConfig


@pytest.fixture
def features_path(tmp_path: Path) -> Path:
    """Synthetic features.parquet — enough rows + columns for the
    orchestrator to slice into k folds without errors. Real-fit
    correctness isn't tested here (stubs replace _train_one_fold)."""
    dates = pd.date_range("2016-09-01", "2026-04-30", freq="D")
    rows = []
    for d in dates:
        for sid in ("s1", "s2"):
            rows.append({
                "station_id": sid,
                "fuel_code": "U91",
                "date": d.date(),
                "price_mean": 180.0,
                "y_t1": 181.0,  # non-null target
                # Token feature columns — _train_one_fold is stubbed so
                # content doesn't matter; size matters.
                "lag_price_1": 179.0,
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "features.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    return p


def _fake_fit_result() -> object:
    """Stand-in for a FitResult; mimics the attributes train_kfold's
    audit writer reads."""
    fit = MagicMock()
    fit.best_iteration = 100
    fit.best_score = 4.5
    fit.feature_columns = ["lag_price_1"]
    fit.categorical_columns = []
    return fit


@pytest.fixture
def stub_train_one_fold():
    """Patch _train_one_fold so the orchestrator runs without LightGBM."""

    def fake(*, train_full, val_full, test_folds, out_dir, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "A": _fake_fit_result(),
            "B": _fake_fit_result(),
            "B_PRIME": _fake_fit_result(),
        }

    with patch("fuel_pred.train.cv._train_one_fold", side_effect=fake) as p:
        yield p


# --------------------------- orchestrator shape ---------------------------


def test_train_kfold_calls_inner_once_per_fold(
    tmp_path: Path, features_path: Path, stub_train_one_fold,
) -> None:
    out = tmp_path / "models_kfold"
    cfg = KFoldConfig.default()  # k=6
    result = train_kfold(features_path, out, kfold_config=cfg)
    assert isinstance(result, KFoldRunResult)
    assert stub_train_one_fold.call_count == cfg.k
    assert len(result.per_fold_results) == cfg.k
    assert len(result.per_fold_out_dirs) == cfg.k


def test_train_kfold_creates_per_fold_subdirs(
    tmp_path: Path, features_path: Path, stub_train_one_fold,
) -> None:
    out = tmp_path / "models_kfold"
    cfg = KFoldConfig.default()
    train_kfold(features_path, out, kfold_config=cfg)
    for fold_idx in range(1, cfg.k + 1):
        assert (out / f"fold_{fold_idx}").is_dir(), (
            f"fold_{fold_idx} subdir not created"
        )


def test_train_kfold_passes_test_dict_with_only_test_key(
    tmp_path: Path, features_path: Path, stub_train_one_fold,
) -> None:
    """k-fold's test_folds is {'test': df} — no crisis, no test_normal."""
    out = tmp_path / "models_kfold"
    train_kfold(features_path, out, kfold_config=KFoldConfig.default())
    for call in stub_train_one_fold.call_args_list:
        test_folds = call.kwargs["test_folds"]
        assert set(test_folds.keys()) == {"test"}, (
            f"k-fold should pass single 'test' key, got {test_folds.keys()}"
        )


def test_train_kfold_forwards_train_val_test_per_fold(
    tmp_path: Path, features_path: Path, stub_train_one_fold,
) -> None:
    """Each call gets non-empty train/val/test (synthetic panel covers
    enough span)."""
    out = tmp_path / "models_kfold"
    train_kfold(features_path, out, kfold_config=KFoldConfig.default())
    for call in stub_train_one_fold.call_args_list:
        assert len(call.kwargs["train_full"]) > 0
        assert len(call.kwargs["val_full"]) > 0
        assert len(call.kwargs["test_folds"]["test"]) > 0


def test_train_kfold_writes_kfold_audit_json(
    tmp_path: Path, features_path: Path, stub_train_one_fold,
) -> None:
    out = tmp_path / "models_kfold"
    cfg = KFoldConfig.default()
    train_kfold(features_path, out, kfold_config=cfg)
    audit_path = out / "kfold_audit.json"
    assert audit_path.exists()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["kfold_config"]["k"] == cfg.k
    assert payload["kfold_config"]["gap_days"] == cfg.gap_days
    assert payload["kfold_config"]["panel_end"] == cfg.panel_end
    assert len(payload["folds"]) == cfg.k
    for fold_idx, fold_summary in enumerate(payload["folds"], start=1):
        assert fold_summary["fold"] == fold_idx
        assert "models" in fold_summary
        assert set(fold_summary["models"].keys()) == {"A", "B", "B_PRIME"}
        for summary in fold_summary["models"].values():
            assert "best_iteration" in summary
            assert "best_val_mae" in summary
            assert "n_features" in summary


def test_train_kfold_forwards_overrides_to_inner(
    tmp_path: Path, features_path: Path, stub_train_one_fold,
) -> None:
    """target / save_predictions / log_period / n_estimators reach _train_one_fold."""
    out = tmp_path / "models_kfold"
    train_kfold(
        features_path, out,
        kfold_config=KFoldConfig.default(),
        target="y_t1",
        save_predictions=False,
        log_period=10,
        n_estimators=42,
    )
    for call in stub_train_one_fold.call_args_list:
        assert call.kwargs["target"] == "y_t1"
        assert call.kwargs["save_predictions"] is False
        assert call.kwargs["log_period"] == 10
        assert call.kwargs["n_estimators"] == 42


def test_train_kfold_returns_per_fold_out_dirs_in_order(
    tmp_path: Path, features_path: Path, stub_train_one_fold,
) -> None:
    out = tmp_path / "models_kfold"
    cfg = KFoldConfig.default()
    result = train_kfold(features_path, out, kfold_config=cfg)
    expected = [out / f"fold_{i}" for i in range(1, cfg.k + 1)]
    assert result.per_fold_out_dirs == expected


def test_train_kfold_uses_default_config_when_none_passed(
    tmp_path: Path, features_path: Path, stub_train_one_fold,
) -> None:
    out = tmp_path / "models_kfold"
    result = train_kfold(features_path, out)  # no kfold_config arg
    assert result.kfold_config.k == 6
    assert stub_train_one_fold.call_count == 6


def test_train_kfold_honours_custom_k(
    tmp_path: Path, features_path: Path, stub_train_one_fold,
) -> None:
    out = tmp_path / "models_kfold"
    cfg = KFoldConfig(k=3, test_window_months=6, val_window_days=180)
    train_kfold(features_path, out, kfold_config=cfg)
    assert stub_train_one_fold.call_count == 3
    for fold_idx in (1, 2, 3):
        assert (out / f"fold_{fold_idx}").is_dir()
    assert not (out / "fold_4").exists()
