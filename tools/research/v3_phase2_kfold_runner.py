"""v3.0 Phase 2 — k-fold re-evaluation of PR A / B / C experiments.

Runs the v3.0 6-fold k-fold harness against the per-experiment features
parquets PR C left behind (see ``results/pr_c_overnight_summary.md``).
Each experiment reuses the existing features parquet — no re-enriching
— and runs ``train.cv.train_kfold`` + ``evaluate.compare.compare_kfold``.

Wall-clock estimate: ~22 min per experiment × 7 experiments ≈ 150 min.
Each experiment writes:

    models_kfold_<exp>/fold_{1..6}/
        model_a.pkl, model_b.pkl, model_b_prime.pkl,
        feature_lists.json, predictions_test.parquet
    models_kfold_<exp>/kfold_audit.json
    results/v3_phase2_<exp>_kfold.md   (comparison report)

Final summary at ``results/v3_phase2_summary.md`` tables every
experiment's mean ± stdev Δ MAE side-by-side so we can see which v2.x
findings hold up vs were fold-specific.

SA2_COLUMNS handling: PR C experiments E4 / E4a / E4b also broadened
``SA2_COLUMNS`` (the model block). The orchestrator monkey-patches
``feature_blocks.SA2_COLUMNS`` + ``make_features.SA2_FEATURE_COLS``
per experiment before invoking ``train_kfold`` so the model fits on
the right column set.

Run via:
    uv run python tools/research/v3_phase2_kfold_runner.py 2>&1 | \\
        tee tools/research/v3_phase2.log
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

LOG_PATH = REPO_ROOT / "tools" / "research" / "v3_phase2.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a"),
    ],
)
logger = logging.getLogger("v3_phase2")


# ---- experiment configs ----


@dataclass(frozen=True)
class Experiment:
    """One Phase 2 re-evaluation.

    - ``name``: short slug used in output dirs + report file names.
    - ``description``: shown in the summary.
    - ``features_parquet``: pre-existing features parquet to load (from
      PR C). Avoids re-enriching.
    - ``sa2_cols_extra``: tuple of column names to APPEND to the
      committed ``feature_blocks.SA2_COLUMNS`` for this experiment.
      Used by E4 / E4a / E4b which monkey-patched the model block.
    """

    name: str
    description: str
    features_parquet: str
    sa2_cols_extra: tuple[str, ...] = field(default_factory=tuple)


# Curation candidates added to SA2_COLUMNS by PR C E4 / E4a / E4b.
CURATION_CANDIDATES = (
    "sa2_erp_population_65_plus",
    "sa2_erp_median_age",
    "sa2_pct_age_pension_recipients",
    "sa2_pct_jobseeker_recipients",
    "sa2_welfare_density_index",
    "sa2_erp_population_density_per_km2",
)

EXPERIMENTS = (
    Experiment(
        name="pr_b_baseline",
        description=(
            "Committed PR B baseline (SEIFA + ERP-total temporal, 15-col SA2 block). "
            "Sets the v3.0 baseline for Phase 2 comparisons; smoke milestone "
            "already ran this once — re-running here for the merged summary."
        ),
        features_parquet="features.parquet",
    ),
    Experiment(
        name="pr_c_e1_dss_temporal",
        description="PR C E1: DSS welfare (9 universal cols) moved to temporal pass.",
        features_parquet="features_e1_dss_temporal.parquet",
    ),
    Experiment(
        name="pr_c_e2_gcp_temporal",
        description="PR C E2: GCP direct + GCP-internal PRESETs (9 vars) moved to temporal.",
        features_parquet="features_e2_gcp_temporal.parquet",
    ),
    Experiment(
        name="pr_c_e3_combined_temporal",
        description="PR C E3: DSS + GCP both moved to temporal (kitchen-sink).",
        features_parquet="features_e3_combined_temporal.parquet",
    ),
    Experiment(
        name="pr_c_e4_density_plus_curation",
        description=(
            "PR C E4: new ERP.population_density_per_km2 added cross-sectional + "
            "SA2_COLUMNS broadened to 21 (PR A's 5 unmodeled + new density)."
        ),
        features_parquet="features_e4_new_erp_density_plus_curation.parquet",
        sa2_cols_extra=CURATION_CANDIDATES,
    ),
    Experiment(
        name="pr_c_e4a_density_only",
        description=(
            "PR C E4a: just the new ERP.population_density_per_km2 column "
            "(no curation broadening; +1 SA2 col)."
        ),
        features_parquet="features_e4a_density_only.parquet",
        sa2_cols_extra=("sa2_erp_population_density_per_km2",),
    ),
    Experiment(
        name="pr_c_e4b_curation_only",
        description=(
            "PR C E4b: 5 PR-A unmodeled candidates added to SA2_COLUMNS, no "
            "new density (+5 SA2 cols)."
        ),
        features_parquet="features_e4b_curation_only.parquet",
        sa2_cols_extra=(
            "sa2_erp_population_65_plus",
            "sa2_erp_median_age",
            "sa2_pct_age_pension_recipients",
            "sa2_pct_jobseeker_recipients",
            "sa2_welfare_density_index",
        ),
    ),
    Experiment(
        name="pr_c_e5_dss_temporal_plus_curation",
        description=(
            "PR C E5: combine E1 (DSS temporal) + E4 (density + 21-col curation). "
            "v2.x single-split flagged this as a destructive interaction "
            "(B regressed both folds); k-fold tests whether that was fold-specific."
        ),
        features_parquet="features_e5_dss_temporal_plus_curation.parquet",
        sa2_cols_extra=CURATION_CANDIDATES,
    ),
)


# ---- helpers ----


def _import_modules():
    """Late import so logging is set up first."""
    from fuel_pred.build import make_features
    from fuel_pred.evaluate.compare import compare_kfold
    from fuel_pred.train import feature_blocks
    from fuel_pred.train.cv import train_kfold
    return feature_blocks, make_features, train_kfold, compare_kfold


def _snapshot_originals(modules):
    feature_blocks, make_features, _train_kfold, _compare = modules
    return {
        "fb.SA2_COLUMNS": feature_blocks.SA2_COLUMNS,
        "fb.BLOCK_COLUMNS_sa2": feature_blocks.BLOCK_COLUMNS["sa2"],
        "mf.SA2_FEATURE_COLS": make_features.SA2_FEATURE_COLS,
    }


def _restore_originals(modules, originals):
    feature_blocks, make_features, _train_kfold, _compare = modules
    feature_blocks.SA2_COLUMNS = originals["fb.SA2_COLUMNS"]
    feature_blocks.BLOCK_COLUMNS["sa2"] = originals["fb.BLOCK_COLUMNS_sa2"]
    make_features.SA2_FEATURE_COLS = originals["mf.SA2_FEATURE_COLS"]


def _apply_experiment(modules, originals, exp: Experiment) -> None:
    feature_blocks, make_features, _train_kfold, _compare = modules
    new_sa2 = originals["fb.SA2_COLUMNS"] + tuple(exp.sa2_cols_extra)
    feature_blocks.SA2_COLUMNS = new_sa2
    feature_blocks.BLOCK_COLUMNS["sa2"] = new_sa2
    make_features.SA2_FEATURE_COLS = new_sa2
    logger.info(
        "experiment %s: SA2_COLUMNS = %d cols (baseline %d + %d extras: %s)",
        exp.name,
        len(new_sa2),
        len(originals["fb.SA2_COLUMNS"]),
        len(exp.sa2_cols_extra),
        list(exp.sa2_cols_extra) or "(none)",
    )


def _extract_aggregate_metrics(comparison_path: Path) -> dict:
    """Pull mean / stdev Δ MAE from a v3.0 k-fold comparison.md.

    Looks for the "Mean" and "Stdev" rows in the A-vs-B headline table.
    Returns {"mean_delta_mae": ..., "stdev_delta_mae": ..., ...}.
    """
    import re

    text = comparison_path.read_text(encoding="utf-8")
    # The A-vs-B headline section
    block_match = re.search(
        r"## Headline — A vs B \(per-fold \+ aggregate\)(.*?)## ",
        text,
        re.DOTALL,
    )
    if not block_match:
        logger.warning("no A-vs-B aggregate block found in %s", comparison_path)
        return {}
    block = block_match.group(1)
    metrics: dict = {}
    for label, key in (
        ("**Mean**", "mean"),
        ("Stdev", "stdev"),
        ("Min", "min"),
        ("Max", "max"),
    ):
        # Row pattern: | <label> | — | n | mae_a | mae_b | δ | rmse_a | rmse_b | mape_a | mape_b | δ mape |
        m = re.search(
            rf"\|\s*{re.escape(label)}\s*\|[^\|]+\|[^\|]+\|"
            rf"([^\|]+)\|([^\|]+)\|([^\|]+)\|",
            block,
        )
        if not m:
            continue
        try:
            metrics[f"{key}_mae_a"] = float(m.group(1).strip())
            metrics[f"{key}_mae_b"] = float(m.group(2).strip())
            # Δ MAE format is e.g. "+0.215" or "-0.135"; convert
            raw = m.group(3).strip()
            metrics[f"{key}_delta_mae"] = float(raw)
        except (IndexError, ValueError):
            pass
    return metrics


def _run_experiment(modules, originals, exp: Experiment) -> dict:
    """Run k-fold + compare for one experiment. Return its aggregate metrics.

    Resume-safe: if ``models_kfold_<exp>/kfold_audit.json`` already
    exists AND the comparison report exists, skip the run and just
    re-extract metrics. Lets the orchestrator survive laptop sleeps
    or kill+restart without re-doing completed experiments.
    """
    _restore_originals(modules, originals)
    _apply_experiment(modules, originals, exp)

    features_path = DATA_PROCESSED / exp.features_parquet
    if not features_path.exists():
        raise RuntimeError(f"features parquet missing: {features_path}")

    out_root = REPO_ROOT / f"models_kfold_{exp.name}"
    report_path = RESULTS_DIR / f"v3_phase2_{exp.name}_kfold.md"
    audit_path = out_root / "kfold_audit.json"

    # ---- Resume short-circuit ------------------------------------------
    if audit_path.exists() and report_path.exists():
        logger.info(
            "[%s] SKIP — kfold_audit.json + report already exist; "
            "extracting metrics from %s",
            exp.name, report_path,
        )
        metrics = _extract_aggregate_metrics(report_path)
        metrics["wall_clock_min"] = 0.0
        metrics["resumed_from_cache"] = True
        logger.info("[%s] DONE (cached) — %s", exp.name, metrics)
        return metrics

    logger.info("=" * 70)
    logger.info("[%s] %s", exp.name, exp.description)
    logger.info("[%s] features: %s -> models: %s -> report: %s",
                exp.name, features_path, out_root, report_path)
    logger.info("=" * 70)

    _feature_blocks, _make_features, train_kfold, compare_kfold = modules

    t0 = time.monotonic()
    train_kfold(features_path, out_root)
    train_min = (time.monotonic() - t0) / 60
    logger.info("[%s] train_kfold complete in %.1f min", exp.name, train_min)

    t0 = time.monotonic()
    compare_kfold(features_path, out_root, report_path)
    eval_min = (time.monotonic() - t0) / 60
    logger.info("[%s] compare_kfold complete in %.1f min", exp.name, eval_min)

    metrics = _extract_aggregate_metrics(report_path)
    metrics["wall_clock_min"] = round(train_min + eval_min, 1)
    logger.info("[%s] DONE — %s", exp.name, metrics)
    return metrics


def _write_summary(all_metrics: dict, originals: dict) -> None:
    """Write v3_phase2_summary.md tabling every experiment's mean ± stdev."""
    path = RESULTS_DIR / "v3_phase2_summary.md"
    lines = [
        "# v3.0 Phase 2 — k-fold re-evaluation summary",
        "",
        "Generated by `tools/research/v3_phase2_kfold_runner.py`. Each "
        "experiment ran the v3.0 6-fold k-fold harness against the "
        "per-experiment features parquet PR C left behind (no re-enriching). "
        "Per-experiment merged reports at `results/v3_phase2_<exp>_kfold.md`.",
        "",
        "## Methodology recap",
        "",
        "- k = 6 folds, 12-month test windows, expanding-window chronological",
        "- gap_days = 1 (prevents the v2.x y_t1 target-shift leak)",
        "- Test windows roll back from panel_end=2026-04-30: every date in "
        "  2017-01..2026-04 appears in exactly one test fold",
        "- No \"crisis\" concept: 2026 is just another time period in "
        "  rotating CV",
        "- **Significance signal: |Mean Δ MAE| vs Stdev Δ MAE.** If "
        "  |Mean| ≫ Stdev → robust; if comparable → fold-specific noise.",
        "",
        "## Headline (A vs B, aggregated across 6 folds)",
        "",
        "Δ MAE = MAE B − MAE A. Negative = augmentor adds value.",
        "",
        "| Experiment | Baseline SA2 cols + extras | Mean Δ MAE | Stdev Δ MAE | Min Δ MAE | Max Δ MAE | Verdict | Wall-clock |",
        "|------------|----------------------------|-----------:|------------:|----------:|----------:|---------|-----------:|",
    ]
    baseline_cols = len(originals["fb.SA2_COLUMNS"])
    for exp in EXPERIMENTS:
        m = all_metrics.get(exp.name, {})
        if not m or "mean_delta_mae" not in m:
            lines.append(
                f"| **{exp.name}** | _failed_ | _—_ | _—_ | _—_ | _—_ | _failed_ | _—_ |"
            )
            continue
        mean = m["mean_delta_mae"]
        stdev = m["stdev_delta_mae"]
        mn = m.get("min_delta_mae", float("nan"))
        mx = m.get("max_delta_mae", float("nan"))
        wall = m.get("wall_clock_min", float("nan"))
        # Verdict heuristic: |Mean| > 2*Stdev = "robust"; > Stdev = "weak"; otherwise "noise"
        if abs(mean) > 2 * stdev:
            verdict = "**robust** ✅" if mean < 0 else "**robust ❌ (B loses)**"
        elif abs(mean) > stdev:
            verdict = "weak"
        else:
            verdict = "noise"
        n_cols = baseline_cols + len(exp.sa2_cols_extra)
        cols_label = f"{baseline_cols}" if not exp.sa2_cols_extra else f"{baseline_cols} + {len(exp.sa2_cols_extra)} = {n_cols}"
        lines.append(
            f"| **{exp.name}** | {cols_label} | "
            f"{mean:+.3f} | {stdev:.3f} | {mn:+.3f} | {mx:+.3f} | "
            f"{verdict} | {wall:.1f} min |"
        )

    lines.extend([
        "",
        "## Experiment descriptions",
        "",
    ])
    for exp in EXPERIMENTS:
        lines.append(f"- **{exp.name}** — {exp.description}")

    lines.extend([
        "",
        "## Per-experiment artefacts",
        "",
        "| Experiment | Comparison report | Models root |",
        "|------------|-------------------|-------------|",
    ])
    for exp in EXPERIMENTS:
        lines.append(
            f"| {exp.name} "
            f"| `results/v3_phase2_{exp.name}_kfold.md` "
            f"| `models_kfold_{exp.name}/` (gitignored) |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote summary to %s", path)

    raw_path = RESULTS_DIR / "v3_phase2_metrics.json"
    raw_path.write_text(
        json.dumps(all_metrics, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote raw metrics to %s", raw_path)


def main() -> None:
    logger.info("v3.0 Phase 2 orchestrator starting (%d experiments)", len(EXPERIMENTS))

    modules = _import_modules()
    originals = _snapshot_originals(modules)
    logger.info(
        "baseline SA2_COLUMNS snapshot: %d cols", len(originals["fb.SA2_COLUMNS"])
    )

    all_metrics: dict = {}
    for exp in EXPERIMENTS:
        try:
            metrics = _run_experiment(modules, originals, exp)
            all_metrics[exp.name] = metrics
        except Exception as exc:
            logger.exception("experiment %s FAILED: %s", exp.name, exc)
            all_metrics[exp.name] = {"error": f"{type(exc).__name__}: {exc}"}
        # Write summary after every experiment so partial runs preserve progress
        _write_summary(all_metrics, originals)

    _restore_originals(modules, originals)
    logger.info("v3.0 Phase 2 orchestrator complete")
    logger.info("Summary: %s", RESULTS_DIR / "v3_phase2_summary.md")


if __name__ == "__main__":
    main()
