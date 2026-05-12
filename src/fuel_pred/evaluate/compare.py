"""Produce ``results/comparison.md`` — the headline result document.

Consumes the prediction parquets that ``train.train_models`` produced
(``models/predictions_test_normal.parquet`` and
``models/predictions_test_crisis.parquet``) plus a slice of
``data/processed/features.parquet`` for segmentation, then renders
spec §8.5 metrics in a single Markdown report.

Sections in the output:

1. **Headline (overall)** — both folds, both models, all five §8.5 metrics
   side-by-side with the B - A delta. The single most important table.
2. **Segmented by metro / regional** — `stn_is_metro`
3. **Segmented by brand** — top 8 brands by row count, plus "Other"
4. **Segmented by fuel type** — U91 only in v1 (DL has no target),
   structure stays present so v2 can drop in DL.
5. **Segmented by SEIFA quintile** — Q1 (most disadvantaged) → Q5
   (most advantaged). The augmentor-story chart's tabular cousin.

The "headline" everyone wants: **Model B's MAE minus Model A's MAE,
overall and per-segment.** Negative = augmentor adds value. Spec
§8.5 wording: "the augmentor's value is the size and direction of
this delta."

Spec: spec.md §8.5, §9.2 step 7, §12 Phase 6.
"""
from __future__ import annotations

import argparse
import logging
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fuel_pred.evaluate.metrics import all_metrics

logger = logging.getLogger(__name__)

# Per-fold parquet filenames produced by train.train_models.
PREDICTION_FILES: dict[str, str] = {
    "test_normal": "predictions_test_normal.parquet",
    "test_crisis": "predictions_test_crisis.parquet",
}

# Segmentation columns we pull from features.parquet. Joined onto the
# prediction frame on (station_id, fuel_code, date).
SEGMENTATION_COLUMNS: tuple[str, ...] = (
    "station_id",
    "fuel_code",
    "date",
    "stn_is_metro",
    "stn_brand_canonical",
    "sa2_seifa_irsd_score",
)

# Number of brands to break out individually before collapsing to "Other".
TOP_N_BRANDS: int = 8


def compare(features_path: Path, models_dir: Path, out_path: Path) -> None:
    """Build the comparison report.

    Args:
        features_path: ``data/processed/features.parquet`` from
            ``build.make_features``. We only read the segmentation
            columns (``SEGMENTATION_COLUMNS``).
        models_dir: directory containing the prediction parquets (and
            the model pickles, though we don't load them — predictions
            are pre-computed for speed).
        out_path: target path, typically ``results/comparison.md``.
            Parent directory is created if missing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load segmentation slice once ------------------------------------
    seg = _load_segmentation_slice(features_path)
    logger.info("loaded segmentation slice: %d rows", len(seg))

    # ---- Build per-fold enriched prediction frames -----------------------
    enriched: dict[str, pd.DataFrame] = {}
    for fold_name, fname in PREDICTION_FILES.items():
        path = models_dir / fname
        if not path.exists():
            logger.warning("missing %s — skipping fold %s", path, fold_name)
            continue
        preds = pd.read_parquet(path)
        merged = preds.merge(
            seg, on=["station_id", "fuel_code", "date"], how="left"
        )
        merged["seifa_quintile"] = _seifa_quintile(merged["sa2_seifa_irsd_score"])
        merged["brand_bucket"] = _bucket_brand(merged["stn_brand_canonical"])
        enriched[fold_name] = merged
        logger.info(
            "fold %s: %d prediction rows after segmentation join",
            fold_name,
            len(merged),
        )

    if not enriched:
        raise RuntimeError(
            f"no prediction parquets found under {models_dir}; expected "
            f"one of: {sorted(PREDICTION_FILES.values())}"
        )

    # ---- Render Markdown -------------------------------------------------
    sections: list[str] = []
    sections.append(_render_header(features_path, models_dir))
    sections.append(_render_headline_table(enriched))
    sections.append(_render_segment_section("Metro / regional", enriched, "stn_is_metro"))
    sections.append(_render_segment_section("Brand (top 8 + Other)", enriched, "brand_bucket"))
    sections.append(_render_segment_section("Fuel type", enriched, "fuel_code"))
    sections.append(_render_segment_section("SEIFA quintile", enriched, "seifa_quintile"))
    sections.append(_render_footer())

    md = "\n\n".join(s for s in sections if s) + "\n"

    # Atomic write.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(md, encoding="utf-8")
    tmp.replace(out_path)
    logger.info("wrote %s (%d bytes)", out_path, len(md))


# ---- internals: loading + enrichment ---------------------------------------


def _load_segmentation_slice(features_path: Path) -> pd.DataFrame:
    """Read just the segmentation columns from features.parquet.

    Loading the whole frame would be wasteful — we only need 6 columns
    out of ~92. PyArrow's column projection skips the rest entirely.
    """
    available = set(pd.read_parquet(features_path, engine="pyarrow").columns)
    cols = [c for c in SEGMENTATION_COLUMNS if c in available]
    missing = [c for c in SEGMENTATION_COLUMNS if c not in available]
    if missing:
        logger.warning(
            "features.parquet missing %d segmentation column(s): %s — "
            "downstream segments may collapse to 'Unknown'",
            len(missing),
            missing,
        )
    return pd.read_parquet(features_path, columns=cols)


def _seifa_quintile(scores: pd.Series) -> pd.Series:
    """Bin SEIFA IRSD score into Q1..Q5 (most → least disadvantaged).

    Uses ``pd.qcut`` for equal-frequency bins so each quintile holds
    ~the same number of rows. Q1 is the lowest (most disadvantaged),
    Q5 is the highest (most advantaged) — same direction as the EDA
    notebook's §6 chart so the tables read consistently.

    Falls back to a single "Unknown" label when there are too few
    distinct scores to form 5 bins (e.g. tiny test corpora).
    """
    try:
        bins = pd.qcut(
            scores,
            q=5,
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
            duplicates="drop",
        )
        return bins.astype(str).fillna("Unknown")
    except (ValueError, TypeError):
        return pd.Series(["Unknown"] * len(scores), index=scores.index)


def _bucket_brand(brand: pd.Series) -> pd.Series:
    """Collapse brand to top-N buckets + 'Other'."""
    counts = brand.value_counts(dropna=False)
    top = counts.head(TOP_N_BRANDS).index
    bucketed = brand.where(brand.isin(top), other="Other")
    return bucketed.fillna("Other")


# ---- internals: metric computation per segment -----------------------------


def _row_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    """Compute the §8.5 metric set for one (segment, model) slice.

    Returns a dict with separate A and B columns plus the B - A delta
    on the headline metrics, ready to drop into a Markdown table row.
    """
    out: dict[str, Any] = {"n": len(rows)}
    if rows.empty:
        for k in ("mae_a", "rmse_a", "mape_a", "median_a", "p90_a",
                 "mae_b", "rmse_b", "mape_b", "median_b", "p90_b",
                 "delta_mae", "delta_mape"):
            out[k] = float("nan")
        return out

    metrics_a = all_metrics(rows["y_true"], rows["y_pred_a"])
    metrics_b = all_metrics(rows["y_true"], rows["y_pred_b"])
    out.update(
        {
            "mae_a": metrics_a["mae"],
            "rmse_a": metrics_a["rmse"],
            "mape_a": metrics_a["mape"],
            "median_a": metrics_a["median_abs_error"],
            "p90_a": metrics_a["p90_abs_error"],
            "mae_b": metrics_b["mae"],
            "rmse_b": metrics_b["rmse"],
            "mape_b": metrics_b["mape"],
            "median_b": metrics_b["median_abs_error"],
            "p90_b": metrics_b["p90_abs_error"],
            "delta_mae": metrics_b["mae"] - metrics_a["mae"],
            "delta_mape": metrics_b["mape"] - metrics_a["mape"],
        }
    )
    return out


# ---- internals: markdown rendering -----------------------------------------


def _render_header(features_path: Path, models_dir: Path) -> str:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return textwrap.dedent(
        f"""
        # Model A vs Model B — comparison report

        Generated: {ts}
        Features: `{features_path}`
        Models:   `{models_dir}/model_a.pkl`, `{models_dir}/model_b.pkl`

        Per spec §8.5: Model A uses lag + upstream + cal + ctx + stn + wx
        feature blocks. Model B adds the SA2 demographic block. **Both
        models train on identical rows** (those where every SA2 column
        is non-null) so the comparison isolates the augmentor's lift.

        - **Negative `Δ MAE` = Model B beats Model A** (augmentor adds value)
        - All metrics in cents/L except MAPE (in %)
        """
    ).strip()


def _render_headline_table(enriched: dict[str, pd.DataFrame]) -> str:
    """Top-of-report 'overall' table — both folds, all metrics, side by side."""
    rows: list[dict[str, Any]] = []
    for fold_name, df in enriched.items():
        m = _row_metrics(df)
        rows.append({"Fold": fold_name, **m})

    if not rows:
        return ""

    body_lines = [
        "## Headline (overall)",
        "",
        "| Fold | n | MAE A | MAE B | Δ MAE | RMSE A | RMSE B | MAPE A | MAPE B | Δ MAPE |",
        "|------|--:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|",
    ]
    for r in rows:
        body_lines.append(
            f"| {r['Fold']} | {r['n']:,} | "
            f"{r['mae_a']:.3f} | {r['mae_b']:.3f} | {_signed(r['delta_mae'])} | "
            f"{r['rmse_a']:.3f} | {r['rmse_b']:.3f} | "
            f"{r['mape_a']:.3f} | {r['mape_b']:.3f} | {_signed(r['delta_mape'])} |"
        )
    return "\n".join(body_lines)


def _render_segment_section(
    title: str, enriched: dict[str, pd.DataFrame], segment_col: str
) -> str:
    """Per-fold segmented table for one segmentation column."""
    chunks: list[str] = [f"## Segmented by {title}"]
    for fold_name, df in enriched.items():
        if segment_col not in df.columns:
            chunks.append(
                f"### {fold_name}\n\n"
                f"_(`{segment_col}` not in fold; skipped)_"
            )
            continue
        rows: list[dict[str, Any]] = []
        for value, group in df.groupby(segment_col, dropna=False, observed=True):
            label = str(value) if pd.notna(value) else "Unknown"
            m = _row_metrics(group)
            rows.append({"Segment": label, **m})
        rows.sort(key=lambda r: -r["n"])  # largest segments first

        if not rows:
            chunks.append(f"### {fold_name}\n\n_(no rows)_")
            continue

        lines = [
            f"### {fold_name}",
            "",
            "| Segment | n | MAE A | MAE B | Δ MAE | MAPE A | MAPE B | Δ MAPE |",
            "|---------|--:|------:|------:|------:|-------:|-------:|-------:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['Segment']} | {r['n']:,} | "
                f"{r['mae_a']:.3f} | {r['mae_b']:.3f} | {_signed(r['delta_mae'])} | "
                f"{r['mape_a']:.3f} | {r['mape_b']:.3f} | {_signed(r['delta_mape'])} |"
            )
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def _render_footer() -> str:
    return textwrap.dedent(
        """
        ---

        _Generated by `python -m fuel_pred.evaluate.compare`. Re-run after
        `make train` to refresh; predictions are read from
        `models/predictions_*.parquet` rather than re-loading the pickles
        for speed._
        """
    ).strip()


def _signed(value: float) -> str:
    """Format a signed delta with explicit + or - prefix."""
    if pd.isna(value):
        return "n/a"
    sign = "+" if value > 0 else ("-" if value < 0 else " ")
    return f"{sign}{abs(value):.3f}"


# ---- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    compare(args.features, args.models, args.out)


if __name__ == "__main__":
    main()
