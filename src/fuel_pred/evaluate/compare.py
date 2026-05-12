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
import json
import logging
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fuel_pred.evaluate.metrics import all_metrics

logger = logging.getLogger(__name__)

# How many top features to show in the importance tables. 20 is enough
# to see both the lag/upstream backbone and the long tail of stn/sa2
# features without making the table unwieldy.
TOP_N_FEATURES_BY_IMPORTANCE: int = 20

# Sample size used for the correlation matrix. The full features.parquet
# is ~15M rows × 90 cols; even on a 50-col subset that's ~7 GB at
# float64. 100k rows is plenty to estimate Pearson r at 3 decimal
# places (SE ~ 1/sqrt(N) so ~0.003 for the absolute scale).
CORRELATION_SAMPLE_SIZE: int = 100_000

# Pearson |r| above this counts as "high correlation" worth flagging
# in the report. 0.5 is a common social-science cutoff for "strong"
# linear relationship; for our purposes (does Model B's SA2 block carry
# information that's already in Model A's features?) it's the right
# order of magnitude — anything below is plausibly independent signal.
HIGH_CORRELATION_THRESHOLD: float = 0.5

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
        # Defensive: segmentation columns may be absent on a stripped-down
        # features.parquet (e.g. test fixtures). Skip the derived columns
        # rather than crashing — the per-fold segment renderer already
        # handles missing columns gracefully.
        if "sa2_seifa_irsd_score" in merged.columns:
            merged["seifa_quintile"] = _seifa_quintile(
                merged["sa2_seifa_irsd_score"]
            )
        if "stn_brand_canonical" in merged.columns:
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

    # ---- Optional: feature_lists.json for importance tables ------------
    importances = _load_feature_lists(models_dir)

    # ---- Render Markdown -------------------------------------------------
    sections: list[str] = []
    sections.append(_render_header(features_path, models_dir))
    sections.append(_render_headline_table(enriched))
    sections.append(_render_segment_section("Metro / regional", enriched, "stn_is_metro"))
    sections.append(_render_segment_section("Brand (top 8 + Other)", enriched, "brand_bucket"))
    sections.append(_render_segment_section("Fuel type", enriched, "fuel_code"))
    sections.append(_render_segment_section("SEIFA quintile", enriched, "seifa_quintile"))
    if importances is not None:
        sections.append(_render_importance_section(importances))
    sections.append(_render_correlation_section(features_path))
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


def _load_feature_lists(models_dir: Path) -> dict[str, Any] | None:
    """Read ``feature_lists.json`` from ``train.train_models`` if present.

    Returns the parsed dict, or None if the file isn't there (we still
    want compare.py to produce a useful report even from a stale models/
    dir that predates feature_lists.json).
    """
    path = models_dir / "feature_lists.json"
    if not path.exists():
        logger.warning(
            "feature_lists.json not found at %s — feature importance "
            "section will be skipped",
            path,
        )
        return None
    try:
        # json.loads is typed as `Any` upstream; tighten to the dict shape
        # we actually expect so mypy accepts the return type.
        parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("feature_lists.json malformed: %s — skipping importance section", e)
        return None


def _render_importance_section(importances: dict[str, Any]) -> str:
    """Top-N features by gain for both models + where SA2 features rank.

    The "where SA2 features rank in Model B" sub-table is the
    project-specific quantitative answer to "did the augmentor's
    columns actually get used by the model?" — separate from whether
    they improved test-fold MAE.
    """
    chunks: list[str] = ["## Feature importance"]

    for model_key in ("A", "B"):
        m = importances.get(model_key, {})
        gain = m.get("importance_gain") or {}
        split = m.get("importance_split") or {}
        if not gain:
            chunks.append(
                f"### Model {model_key}\n\n"
                f"_(no importances recorded — train_models.py predates the "
                f"importance fields)_"
            )
            continue
        ranked = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[:TOP_N_FEATURES_BY_IMPORTANCE]
        total_gain = sum(gain.values()) or 1.0  # avoid /0
        lines = [
            f"### Model {model_key} — top {len(top)} by gain importance",
            "",
            "| Rank | Feature | Block | Gain | Gain % | Splits |",
            "|-----:|---------|-------|-----:|-------:|-------:|",
        ]
        for i, (feat, g) in enumerate(top, start=1):
            block = _block_of(feat)
            s = split.get(feat, 0)
            lines.append(
                f"| {i} | `{feat}` | {block} | {g:,.0f} | "
                f"{100 * g / total_gain:.2f} | {s:,} |"
            )
        chunks.append("\n".join(lines))

    # Cross-model sub-table: where do SA2 features rank in Model B?
    b = importances.get("B", {})
    b_gain = b.get("importance_gain") or {}
    if b_gain:
        b_ranked = sorted(b_gain.items(), key=lambda kv: kv[1], reverse=True)
        b_rank_lookup = {feat: i + 1 for i, (feat, _g) in enumerate(b_ranked)}
        sa2_rows = [
            (feat, gain_val, b_rank_lookup.get(feat, -1))
            for feat, gain_val in b_gain.items()
            if feat.startswith("sa2_")
        ]
        sa2_rows.sort(key=lambda r: r[1], reverse=True)
        if sa2_rows:
            total_b_gain = sum(b_gain.values()) or 1.0
            lines = [
                "### Where SA2 features rank in Model B",
                "",
                "| SA2 feature | Rank in B | Gain | Gain % |",
                "|-------------|----------:|-----:|-------:|",
            ]
            for feat, g, rank in sa2_rows:
                lines.append(
                    f"| `{feat}` | {rank} | {g:,.0f} | "
                    f"{100 * g / total_b_gain:.2f} |"
                )
            chunks.append("\n".join(lines))

    return "\n\n".join(chunks)


def _block_of(feature_name: str) -> str:
    """Map a feature name to its spec §7 block label.

    Lifted from train.feature_blocks but kept inline to avoid an
    extra import — the prefix mapping is stable spec convention.
    """
    if feature_name.startswith("lag_") or feature_name.startswith("roll_") or \
       feature_name.startswith("xfuel_") or feature_name in {
           "days_since_last_price_change", "price_minus_28d_min",
           "price_minus_28d_max",
       }:
        return "lag"
    if feature_name.startswith("upstream_"):
        return "upstream"
    if feature_name.startswith("cal_"):
        return "cal"
    if feature_name.startswith("ctx_"):
        return "ctx"
    if feature_name.startswith("stn_"):
        return "stn"
    if feature_name.startswith("wx_"):
        return "wx"
    if feature_name.startswith("sa2_"):
        return "sa2"
    return "?"


def _render_correlation_section(features_path: Path) -> str:
    """Pearson r between each SA2 feature and each non-SA2 numeric feature.

    The user's concern: if SA2 features are highly correlated with
    features Model A already had, Model B inherits no new information
    and lift will be near-zero regardless of how meaningful the
    underlying SA2 signal is. This section quantifies that.

    Computed on a sample (CORRELATION_SAMPLE_SIZE) for speed —
    Pearson r at 100k rows has SE ~0.003 which is plenty for
    distinguishing "uncorrelated" from "concerning".
    """
    chunks: list[str] = ["## SA2 ↔ non-SA2 feature correlation"]
    chunks.append(
        "_Pearson `r` between each SA2 feature and the most-correlated "
        "non-SA2 numeric feature, computed on a sample of "
        f"{CORRELATION_SAMPLE_SIZE:,} rows. Categoricals are excluded "
        f"(Pearson is numeric-only). High correlation (`|r| > "
        f"{HIGH_CORRELATION_THRESHOLD}`) flags features the model could "
        "already infer from existing inputs._"
    )

    df = _load_correlation_slice(features_path)
    if df is None:
        chunks.append("_(skipped — no SA2 columns found in features.parquet)_")
        return "\n\n".join(chunks)

    sa2_cols = [c for c in df.columns if c.startswith("sa2_")]
    other_cols = [c for c in df.columns if not c.startswith("sa2_")]
    if not sa2_cols or not other_cols:
        chunks.append("_(skipped — not enough columns to correlate)_")
        return "\n\n".join(chunks)

    # Compute the cross-correlation matrix once. .corr() on the union
    # is a single pass through the data; we then slice rows = SA2 cols,
    # cols = non-SA2 cols.
    full = df.corr(method="pearson", numeric_only=True)
    if full.empty:
        chunks.append("_(skipped — no numeric overlap)_")
        return "\n\n".join(chunks)
    cross = full.loc[
        [c for c in sa2_cols if c in full.index],
        [c for c in other_cols if c in full.columns],
    ]
    if cross.empty:
        chunks.append("_(skipped — sa2 / non-sa2 sets are disjoint after numeric filter)_")
        return "\n\n".join(chunks)

    # Per-SA2-feature top correlations (top 3).
    lines_top = [
        "### Top 3 correlated non-SA2 features per SA2 feature",
        "",
        "| SA2 feature | #1 (|r|) | #2 (|r|) | #3 (|r|) |",
        "|-------------|----------|----------|----------|",
    ]
    for sa2 in cross.index:
        row = cross.loc[sa2].dropna()
        if row.empty:
            lines_top.append(f"| `{sa2}` | n/a | n/a | n/a |")
            continue
        top = row.abs().sort_values(ascending=False).head(3).index.tolist()
        cells = []
        for col in top:
            r = row[col]
            flag = " ⚠️" if abs(r) >= HIGH_CORRELATION_THRESHOLD else ""
            cells.append(f"`{col}` ({r:+.3f}){flag}")
        # Pad with em-dashes if fewer than 3 candidates.
        while len(cells) < 3:
            cells.append("—")
        lines_top.append(f"| `{sa2}` | {cells[0]} | {cells[1]} | {cells[2]} |")
    chunks.append("\n".join(lines_top))

    # High-correlation callout: flat list of every (sa2, other, r) above threshold.
    high_pairs: list[tuple[str, str, float]] = []
    for sa2 in cross.index:
        for other in cross.columns:
            r = cross.loc[sa2, other]
            if pd.notna(r) and abs(r) >= HIGH_CORRELATION_THRESHOLD:
                high_pairs.append((sa2, other, float(r)))

    if high_pairs:
        high_pairs.sort(key=lambda t: -abs(t[2]))
        lines_high = [
            f"### High correlations (|r| ≥ {HIGH_CORRELATION_THRESHOLD})",
            "",
            "| SA2 feature | Non-SA2 feature | r | Block |",
            "|-------------|------------------|--:|-------|",
        ]
        for sa2, other, r in high_pairs:
            lines_high.append(f"| `{sa2}` | `{other}` | {r:+.3f} | {_block_of(other)} |")
        chunks.append("\n".join(lines_high))
    else:
        chunks.append(
            f"### High correlations (|r| ≥ {HIGH_CORRELATION_THRESHOLD})\n\n"
            "_(none — every SA2 feature carries information at least "
            "partially independent of the existing feature set)_"
        )

    return "\n\n".join(chunks)


def _load_correlation_slice(features_path: Path) -> pd.DataFrame | None:
    """Read a sample of SA2 + non-SA2 numeric columns for correlation.

    Returns None if there are no SA2 columns to correlate. Drops
    target columns + identifier columns up front so they don't
    pollute the correlation matrix.
    """
    # First pass: peek at column names + dtypes without loading the data.
    schema = pd.read_parquet(features_path, engine="pyarrow").dtypes
    # Identifier / target / today's-price exclusions — same intent as
    # train.feature_blocks.EXCLUDE_FROM_FEATURES, kept in sync manually.
    exclude = {
        "station_id", "fuel_code", "date",
        "price_mean", "price_min", "price_max", "n_obs",
        "y_t1", "y_t1_t7",
        "name", "address", "suburb", "postcode",
        "brand_raw", "brand_canonical", "brand_is_major",
        "first_seen", "last_seen",
        "lat", "lon", "geocoder", "mb_code",
        "sa2_code", "sa2_name", "counter_id",
    }
    numeric_cols = [
        str(c) for c, dt in schema.items()
        if c not in exclude and pd.api.types.is_numeric_dtype(dt)
    ]
    sa2_cols = [c for c in numeric_cols if c.startswith("sa2_")]
    if not sa2_cols:
        return None

    df = pd.read_parquet(features_path, columns=numeric_cols)
    if len(df) > CORRELATION_SAMPLE_SIZE:
        # Stratified random sample is overkill — Pearson r on a
        # well-mixed simple sample is unbiased.
        df = df.sample(CORRELATION_SAMPLE_SIZE, random_state=42)
    # Drop rows where every SA2 col is null — they carry no SA2
    # information and would skew the correlation toward 0.
    df = df.dropna(subset=sa2_cols, how="all")
    return df


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
