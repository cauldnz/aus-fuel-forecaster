"""Hermetic tests for evaluate.compare."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fuel_pred.evaluate import compare as cmp

# ----------------------------- fixtures -----------------------------


def _synth_predictions(
    n_stations: int = 6,
    n_days: int = 60,
    *,
    a_better_at_low_seifa: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Build synthetic predictions for one fold.

    Schema matches what train.train_models writes:
        station_id, fuel_code, date, y_true, y_pred_a, y_pred_b.

    Each station has a random base price + mild noise. Predictions
    deviate by station_idx-derived amounts so segmentation has signal.

    a_better_at_low_seifa=True: Model A wins on the first half of
    stations (proxy for "low SEIFA"); Model B wins on the second half.
    Used to verify the comparison report shows the per-segment delta
    going in different directions.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows: list[dict[str, object]] = []
    for s in range(n_stations):
        base = 180.0 + s * 0.5
        for d in dates:
            y_true = base + rng.normal(0, 0.5)
            # B is generally slightly better, but flip per-station if requested
            if a_better_at_low_seifa and s < n_stations // 2:
                y_pred_a = y_true + rng.normal(0, 0.5)
                y_pred_b = y_true + rng.normal(0, 1.0)  # B worse
            else:
                y_pred_a = y_true + rng.normal(0, 1.0)
                y_pred_b = y_true + rng.normal(0, 0.5)  # B better
            rows.append(
                {
                    "station_id": f"s{s:03d}",
                    "fuel_code": "U91",
                    "date": d,
                    "y_true": float(y_true),
                    "y_pred_a": float(y_pred_a),
                    "y_pred_b": float(y_pred_b),
                }
            )
    return pd.DataFrame(rows)


def _synth_features(n_stations: int = 6, n_days: int = 60) -> pd.DataFrame:
    """Synthetic features.parquet that carries just the segmentation columns."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows: list[dict[str, object]] = []
    for s in range(n_stations):
        for d in dates:
            rows.append(
                {
                    "station_id": f"s{s:03d}",
                    "fuel_code": "U91",
                    "date": d,
                    "stn_is_metro": bool(s % 2 == 0),
                    "stn_brand_canonical": ["BP", "Ampol", "Shell", "7-Eleven"][s % 4],
                    "sa2_seifa_irsd_score": 900.0 + s * 50.0,  # spreads quintiles
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def models_dir_with_predictions(tmp_path: Path) -> Path:
    """Create a tmp models/ dir containing both fold parquets."""
    out = tmp_path / "models"
    out.mkdir()
    _synth_predictions(seed=42).to_parquet(
        out / "predictions_test_normal.parquet",
        engine="pyarrow", compression="zstd", index=False,
    )
    _synth_predictions(seed=99).to_parquet(
        out / "predictions_test_crisis.parquet",
        engine="pyarrow", compression="zstd", index=False,
    )
    return out


@pytest.fixture
def features_path(tmp_path: Path) -> Path:
    p = tmp_path / "features.parquet"
    _synth_features().to_parquet(
        p, engine="pyarrow", compression="zstd", index=False
    )
    return p


# ----------------------------- end-to-end -----------------------------


def test_compare_writes_output_file(
    features_path: Path, models_dir_with_predictions: Path, tmp_path: Path
) -> None:
    out = tmp_path / "results" / "comparison.md"
    cmp.compare(features_path, models_dir_with_predictions, out)
    assert out.exists(), "comparison.md should land at the requested path"
    # Atomic write: no leftover .tmp file.
    assert not (out.parent / (out.name + ".tmp")).exists()


def test_compare_output_is_well_formed_markdown(
    features_path: Path, models_dir_with_predictions: Path, tmp_path: Path
) -> None:
    """Smoke check: every expected section header present."""
    out = tmp_path / "results" / "comparison.md"
    cmp.compare(features_path, models_dir_with_predictions, out)
    text = out.read_text(encoding="utf-8")

    assert "# Model A vs Model B" in text
    assert "## Headline (overall)" in text
    assert "## Segmented by Metro / regional" in text
    assert "## Segmented by Brand" in text
    assert "## Segmented by Fuel type" in text
    assert "## Segmented by SEIFA quintile" in text
    # Per-fold subsections.
    assert "### test_normal" in text
    assert "### test_crisis" in text
    # The headline column header.
    assert "| Fold | n | MAE A | MAE B | Δ MAE |" in text


def test_compare_renders_signed_delta_for_b_minus_a(
    features_path: Path, models_dir_with_predictions: Path, tmp_path: Path
) -> None:
    """Δ MAE should appear with explicit + / - sign."""
    out = tmp_path / "results" / "comparison.md"
    cmp.compare(features_path, models_dir_with_predictions, out)
    text = out.read_text(encoding="utf-8")
    # At least one signed delta cell — either positive or negative —
    # should appear in the body.
    has_signed = any(
        line.startswith("| ") and (("| +" in line) or ("| -" in line))
        for line in text.splitlines()
    )
    assert has_signed, "expected at least one signed +/- delta in the table body"


def test_compare_segments_by_seifa_quintile(
    features_path: Path, models_dir_with_predictions: Path, tmp_path: Path
) -> None:
    """SEIFA quintile section should produce Q1..Q5 rows when scores
    span the range."""
    out = tmp_path / "results" / "comparison.md"
    cmp.compare(features_path, models_dir_with_predictions, out)
    text = out.read_text(encoding="utf-8")
    # Find the SEIFA section and check it has quintile rows.
    seifa_section = text.split("## Segmented by SEIFA quintile", 1)[1]
    # We have 6 stations spanning scores 900..1150 (50 apart) → qcut
    # produces 5 unique bin edges → some Q labels appear.
    assert any(q in seifa_section for q in ("Q1", "Q2", "Q3", "Q4", "Q5"))


def test_compare_handles_missing_crisis_fold(
    features_path: Path, tmp_path: Path
) -> None:
    """Missing test_crisis parquet → log a warning but still produce
    a report from test_normal alone."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _synth_predictions().to_parquet(
        models_dir / "predictions_test_normal.parquet",
        engine="pyarrow", compression="zstd", index=False,
    )
    # No test_crisis parquet on disk.
    out = tmp_path / "results" / "comparison.md"
    cmp.compare(features_path, models_dir, out)
    text = out.read_text(encoding="utf-8")
    assert "### test_normal" in text
    # No test_crisis subsection appears anywhere.
    assert "### test_crisis" not in text


def test_compare_raises_when_no_predictions_present(
    features_path: Path, tmp_path: Path
) -> None:
    """Empty models/ dir → loud failure (vs. silently writing an empty report)."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out = tmp_path / "results" / "comparison.md"
    with pytest.raises(RuntimeError, match="no prediction parquets"):
        cmp.compare(features_path, models_dir, out)


def test_compare_brand_bucket_collapses_to_top_n_plus_other(
    tmp_path: Path,
) -> None:
    """When brand cardinality > TOP_N_BRANDS, low-frequency brands collapse
    into 'Other'."""
    n_stations = cmp.TOP_N_BRANDS + 4  # forces some 'Other' bucket members
    n_days = 30
    preds = _synth_predictions(n_stations=n_stations, n_days=n_days)
    # Build a features frame with one brand per station — that yields
    # n_stations distinct brands → n - TOP_N collapse to 'Other'.
    feats_rows: list[dict[str, object]] = []
    for s in range(n_stations):
        for d in pd.date_range("2024-01-01", periods=n_days, freq="D"):
            feats_rows.append(
                {
                    "station_id": f"s{s:03d}",
                    "fuel_code": "U91",
                    "date": d,
                    "stn_is_metro": True,
                    "stn_brand_canonical": f"Brand{s:02d}",
                    "sa2_seifa_irsd_score": 1000.0,
                }
            )
    feats_path = tmp_path / "features.parquet"
    pd.DataFrame(feats_rows).to_parquet(feats_path)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    preds.to_parquet(models_dir / "predictions_test_normal.parquet")
    preds.to_parquet(models_dir / "predictions_test_crisis.parquet")

    out = tmp_path / "results" / "comparison.md"
    cmp.compare(feats_path, models_dir, out)
    text = out.read_text(encoding="utf-8")
    # The Brand-segment section MUST contain "Other" because we exceeded
    # the top-N cap.
    brand_section = text.split("## Segmented by Brand", 1)[1].split("##", 1)[0]
    assert "Other" in brand_section


# ----------------------------- internal helpers -----------------------------


def test_signed_formatter() -> None:
    assert cmp._signed(0.123) == "+0.123"
    assert cmp._signed(-0.456) == "-0.456"
    assert cmp._signed(0.0) == " 0.000"
    assert cmp._signed(float("nan")) == "n/a"


def test_seifa_quintile_falls_back_to_unknown_for_few_distinct() -> None:
    """One distinct score → not enough variation for 5 bins → 'Unknown'."""
    # All identical → qcut may either return 1 bucket or raise; either way
    # the fallback delivers Unknown.
    out = cmp._seifa_quintile(pd.Series([1000.0] * 10))
    assert (out == "Unknown").all() or (out.nunique() == 1)


def test_bucket_brand_keeps_top_n_intact() -> None:
    """A brand in the top-N must NOT be relabeled 'Other'."""
    rng = np.random.default_rng(0)
    # Make BP the most frequent brand by far.
    brands = ["BP"] * 100 + ["Ampol"] * 50 + ["Shell"] * 25 + ["Costco"] * 5
    rng.shuffle(brands)
    s = pd.Series(brands)
    out = cmp._bucket_brand(s)
    # All 4 brands fit under TOP_N_BRANDS, so nothing should collapse.
    assert "Other" not in out.unique()
    assert set(out.unique()) == {"BP", "Ampol", "Shell", "Costco"}
