"""Hermetic tests for the v3.0 k-fold comparison report (evaluate.compare_kfold).

Synthesise per-fold predictions_test.parquet files + a kfold_audit.json,
run compare_kfold, assert the rendered Markdown has the right shape:
per-fold + aggregate rows, no crisis-as-separate framing, segmentation
tables over the union of folds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fuel_pred.evaluate.compare_kfold import compare_kfold


def _make_features(tmp_path: Path) -> Path:
    """Minimum-viable features.parquet with segmentation columns."""
    rows = []
    for sid in ("s1", "s2"):
        for fuel in ("U91",):
            for d in pd.date_range("2020-01-01", "2026-04-30", freq="D"):
                rows.append({
                    "station_id": sid,
                    "fuel_code": fuel,
                    "date": d.date(),
                    # Segmentation columns _load_segmentation_slice reads
                    "stn_is_metro": (sid == "s1"),
                    "stn_brand_canonical": "BP" if sid == "s1" else "Ampol",
                    "sa2_seifa_irsd_score": 1000.0,
                })
    df = pd.DataFrame(rows)
    p = tmp_path / "features.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    return p


def _make_fold_predictions(
    fold_dir: Path,
    test_dates: tuple[str, str],
    mae_a: float,
    mae_b: float,
    *,
    include_b_prime: bool = True,
) -> None:
    """Synthetic predictions_test.parquet — y_true is constant, y_pred_X
    set so the residual magnitudes match the requested MAE."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    start, end = test_dates
    rows = []
    for sid in ("s1", "s2"):
        for d in pd.date_range(start, end, freq="D"):
            rows.append({
                "station_id": sid,
                "fuel_code": "U91",
                "date": d.date(),
                "y_true": 180.0,
                "y_pred_a": 180.0 + mae_a,
                "y_pred_b": 180.0 + mae_b,
            })
            if include_b_prime:
                rows[-1]["y_pred_b_prime"] = 180.0 + mae_b + 0.1
    pd.DataFrame(rows).to_parquet(
        fold_dir / "predictions_test.parquet",
        engine="pyarrow", compression="zstd", index=False,
    )


def _make_kfold_audit(root: Path, k: int = 6) -> None:
    """Audit JSON the renderer's header reads."""
    payload = {
        "kfold_config": {
            "k": k,
            "test_window_months": 12,
            "val_window_days": 365,
            "gap_days": 1,
            "horizon_days": 1,
            "warmup_end": "2016-12-31",
            "panel_end": "2026-04-30",
        },
        "folds": [
            {
                "fold": i,
                "out_dir": str(root / f"fold_{i}"),
                "models": {
                    "A": {"best_iteration": 100, "best_val_mae": 4.5,
                          "n_features": 80, "n_categorical": 3},
                    "B": {"best_iteration": 100, "best_val_mae": 4.5,
                          "n_features": 95, "n_categorical": 3},
                    "B_PRIME": {"best_iteration": 100, "best_val_mae": 4.5,
                                "n_features": 100, "n_categorical": 4},
                },
            }
            for i in range(1, k + 1)
        ],
    }
    (root / "kfold_audit.json").write_text(json.dumps(payload, indent=2))


@pytest.fixture
def kfold_setup(tmp_path: Path) -> tuple[Path, Path]:
    """Build a 3-fold synthetic kfold output root + matching features file."""
    features_path = _make_features(tmp_path)
    root = tmp_path / "models_kfold"
    root.mkdir(parents=True)
    _make_fold_predictions(root / "fold_1", ("2020-05-01", "2021-04-30"), 1.0, 0.8)
    _make_fold_predictions(root / "fold_2", ("2021-05-01", "2022-04-30"), 1.2, 0.9)
    _make_fold_predictions(root / "fold_3", ("2022-05-01", "2023-04-30"), 1.1, 0.7)
    _make_kfold_audit(root, k=3)
    return features_path, root


# --------------------------- behaviour ---------------------------


def test_compare_kfold_writes_output_file(kfold_setup: tuple[Path, Path], tmp_path: Path) -> None:
    features_path, root = kfold_setup
    out = tmp_path / "comparison_kfold.md"
    compare_kfold(features_path, root, out)
    assert out.exists()
    assert out.stat().st_size > 100


def test_compare_kfold_headline_has_per_fold_rows_then_aggregate(
    kfold_setup: tuple[Path, Path], tmp_path: Path,
) -> None:
    features_path, root = kfold_setup
    out = tmp_path / "comparison_kfold.md"
    compare_kfold(features_path, root, out)
    text = out.read_text(encoding="utf-8")
    # Per-fold rows
    assert "| fold_1 |" in text
    assert "| fold_2 |" in text
    assert "| fold_3 |" in text
    # Aggregate rows (Mean bolded; Stdev/Min/Max plain)
    assert "**Mean**" in text
    assert "| Stdev |" in text
    assert "| Min |" in text
    assert "| Max |" in text


def test_compare_kfold_no_crisis_or_test_normal_framing(
    kfold_setup: tuple[Path, Path], tmp_path: Path,
) -> None:
    """v3.0 dropped crisis-as-separate; the rendered report shouldn't
    use the v2.x fold-name vocabulary."""
    features_path, root = kfold_setup
    out = tmp_path / "comparison_kfold.md"
    compare_kfold(features_path, root, out)
    text = out.read_text(encoding="utf-8")
    # Not strict — the word "crisis" could appear in a discussion of
    # methodology. But specific v2.x fold rows like "test_normal"
    # should not appear as table-row labels.
    assert "| test_normal |" not in text
    assert "| test_crisis |" not in text


def test_compare_kfold_includes_b_prime_table_when_predictions_carry_it(
    kfold_setup: tuple[Path, Path], tmp_path: Path,
) -> None:
    features_path, root = kfold_setup
    out = tmp_path / "comparison_kfold.md"
    compare_kfold(features_path, root, out)
    text = out.read_text(encoding="utf-8")
    assert "B vs B'" in text  # second headline section
    assert "Δ MAE (B'−B)" in text


def test_compare_kfold_segments_render_across_union_of_folds(
    kfold_setup: tuple[Path, Path], tmp_path: Path,
) -> None:
    """Segmentation tables aggregate across all folds (one row per
    segment, not per-fold-per-segment)."""
    features_path, root = kfold_setup
    out = tmp_path / "comparison_kfold.md"
    compare_kfold(features_path, root, out)
    text = out.read_text(encoding="utf-8")
    assert "Metro / regional" in text
    assert "Brand" in text
    assert "Fuel type" in text
    assert "SEIFA quintile" in text


def test_compare_kfold_header_quotes_kfold_config_from_audit(
    kfold_setup: tuple[Path, Path], tmp_path: Path,
) -> None:
    features_path, root = kfold_setup
    out = tmp_path / "comparison_kfold.md"
    compare_kfold(features_path, root, out)
    text = out.read_text(encoding="utf-8")
    assert "k=3" in text  # our synthetic setup uses k=3
    assert "gap_days=1" in text
    assert "panel_end=2026-04-30" in text


def test_compare_kfold_raises_when_no_fold_subdirs(tmp_path: Path) -> None:
    features_path = _make_features(tmp_path)
    empty_root = tmp_path / "models_kfold_empty"
    empty_root.mkdir()
    with pytest.raises(RuntimeError, match="no fold_"):
        compare_kfold(features_path, empty_root, tmp_path / "out.md")


def test_compare_kfold_skips_folds_without_predictions(
    tmp_path: Path,
) -> None:
    """A fold_N subdir without predictions_test.parquet is warned + skipped."""
    features_path = _make_features(tmp_path)
    root = tmp_path / "models_kfold"
    root.mkdir()
    # fold_1 has predictions, fold_2 is empty
    _make_fold_predictions(root / "fold_1", ("2020-05-01", "2021-04-30"), 1.0, 0.8)
    (root / "fold_2").mkdir()
    _make_kfold_audit(root, k=2)
    out = tmp_path / "comparison_kfold.md"
    compare_kfold(features_path, root, out)
    text = out.read_text(encoding="utf-8")
    assert "| fold_1 |" in text
    assert "| fold_2 |" not in text


def test_compare_kfold_handles_missing_kfold_audit(tmp_path: Path) -> None:
    """No kfold_audit.json → warn + render without geometry annotations."""
    features_path = _make_features(tmp_path)
    root = tmp_path / "models_kfold"
    root.mkdir()
    _make_fold_predictions(root / "fold_1", ("2020-05-01", "2021-04-30"), 1.0, 0.8)
    out = tmp_path / "comparison_kfold.md"
    compare_kfold(features_path, root, out)
    text = out.read_text(encoding="utf-8")
    assert "| fold_1 |" in text
    # Without audit, k-fold config header line shouldn't appear
    assert "K-fold config:" not in text


def test_compare_kfold_aggregate_mean_matches_per_fold_mean(
    kfold_setup: tuple[Path, Path], tmp_path: Path,
) -> None:
    """Sanity: the Mean row's Δ MAE equals the arithmetic mean of per-fold Δ MAE."""
    features_path, root = kfold_setup
    out = tmp_path / "comparison_kfold.md"
    compare_kfold(features_path, root, out)
    text = out.read_text(encoding="utf-8")
    # Per-fold Δ MAE: fold 1 = 0.8-1.0 = -0.2; fold 2 = 0.9-1.2 = -0.3;
    # fold 3 = 0.7-1.1 = -0.4. Mean = -0.3
    assert "-0.300" in text  # the Mean Δ MAE row should show -0.300
