"""PR C overnight experiment orchestrator.

Runs 4 experiments sequentially against the current pinned augmentor
(commit ``762a6a0f`` — has all three of our PR-A/B-era issue fixes plus
PR #97's new ``ERP.population_density_per_km2`` column).

For each experiment:

1. Monkey-patches ``config.AUGMENTOR_VARIABLES_{CROSS_SECTIONAL,TEMPORAL}``
   plus the module-level constants in ``enrich_census``,
   ``enrich_panel_temporal``, ``make_features``, and ``feature_blocks``
   that captured the originals at import time.
2. Determines which pipeline stages need to re-run based on what
   changed (e.g. E4 doesn't need to re-temporal-enrich because its
   temporal set matches PR B's baseline).
3. Runs the necessary stages end-to-end.
4. Saves a per-experiment comparison report at
   ``results/pr_c_<exp_name>_comparison.md`` and snapshots the model
   artefacts to ``models_<exp_name>/``.
5. At the end, writes a summary table to
   ``results/pr_c_overnight_summary.md``.

Designed for unattended overnight execution. Runtime estimate per
experiment: ~25-45 min, dominated by augmentor cache fetching for
temporal-DSS/GCP (cold start) + the 14M-row feature build. Total
wall-clock: ~3-4 hours for all four. Each experiment writes its own
artefacts so a partial run still preserves what completed.

Run with:
    uv run python tools/research/pr_c_overnight_runner.py 2>&1 | \\
        tee tools/research/pr_c_overnight.log
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# We monkey-patch a lot. Stay close to the project root for tidy paths.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
MODELS_ROOT = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

STATIONS_PATH = DATA_INTERIM / "stations.parquet"
PANEL_PATH = DATA_INTERIM / "panel.parquet"

# --------------------------- experiment definitions ---------------------------


@dataclass(frozen=True)
class Experiment:
    """One experiment to run. Spec is a delta against the current
    committed PR B baseline.

    - ``move_to_temporal``: friendly names currently in CROSS_SECTIONAL
      to relocate to TEMPORAL.
    - ``add_cross_sectional``: brand-new entries to add to CROSS_SECTIONAL.
    - ``add_to_sa2_model_block``: new ``sa2_*`` column names to append
      to ``feature_blocks.SA2_COLUMNS`` (the 15-col model block).
    """

    name: str
    description: str
    move_to_temporal: tuple[str, ...] = ()
    add_cross_sectional: dict[str, str] = field(default_factory=dict)
    add_to_sa2_model_block: tuple[str, ...] = ()


# GCP direct + GCP-internal PRESETs — the variables that bug #91 Stage 2
# unblocks. All currently in the cross-sectional pass.
GCP_FAMILY = (
    "median_age",
    "median_household_income_weekly",
    "total_population",
    "pct_drive_to_work",
    "motor_vehicles_per_dwelling",
    "pct_renters",
    "pct_employed_full_time",
    "pct_aged_65_plus",
    "pct_one_parent_family",
)

# DSS welfare — 13 vars currently cross-sectional pending #99 (now fixed).
DSS_FAMILY = (
    "dss_age_pension_recipients",
    "dss_jobseeker_payment_recipients",
    "dss_disability_support_pension_recipients",
    "dss_parenting_payment_single_recipients",
    "dss_parenting_payment_partnered_recipients",
    "dss_carer_payment_recipients",
    "dss_carer_allowance_recipients",
    "dss_youth_allowance_other_recipients",
    "dss_youth_allowance_student_and_apprentice_recipients",
    "dss_commonwealth_rent_assistance_recipients",
    "dss_commonwealth_seniors_health_card_recipients",
    "dss_family_tax_benefit_a_recipients",
    "dss_family_tax_benefit_b_recipients",
)

# PR A's 5 unmodeled cross-sectional candidates + the new ERP density
# column. Adds 6 candidates to the 15-col model block.
CURATION_CANDIDATES = (
    "sa2_erp_population_65_plus",
    "sa2_erp_median_age",
    "sa2_pct_age_pension_recipients",
    "sa2_pct_jobseeker_recipients",
    "sa2_welfare_density_index",
    "sa2_erp_population_density_per_km2",  # NEW in upstream PR #97
)

EXPERIMENTS = (
    Experiment(
        name="e1_dss_temporal",
        description="Move 13 DSS variables to temporal pass (orig §7.7.2 motivation, unblocked by #99)",
        move_to_temporal=DSS_FAMILY,
    ),
    Experiment(
        name="e2_gcp_temporal",
        description="Move GCP direct + GCP-internal PRESETs (9 vars) to temporal (unblocked by #91 Stage 2)",
        move_to_temporal=GCP_FAMILY,
    ),
    Experiment(
        name="e3_combined_temporal",
        description="Combine: DSS + GCP both move to temporal (kitchen-sink)",
        move_to_temporal=DSS_FAMILY + GCP_FAMILY,
    ),
    Experiment(
        name="e4_new_erp_density_plus_curation",
        description=(
            "Adopt PR #97's new ERP.population_density_per_km2 + broaden "
            "SA2_COLUMNS with PR A's 5 unmodeled candidates + the new density "
            "(temporal pass = PR B baseline)"
        ),
        add_cross_sectional={
            "erp_population_density_per_km2": "ERP.population_density_per_km2",
        },
        add_to_sa2_model_block=CURATION_CANDIDATES,
    ),
)


# --------------------------- orchestrator ---------------------------


# Logging: write to both stdout and a dedicated log file.
LOG_PATH = REPO_ROOT / "tools" / "research" / "pr_c_overnight.log"
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
logger = logging.getLogger("pr_c_overnight")


def _import_pipeline_modules():
    """Import all pipeline modules. Return (config, eric, ept, mf, fb, train, ev).

    Done lazily inside the orchestrator so logging is set up before the
    augmentor sets up its own loggers.
    """
    from fuel_pred import config
    from fuel_pred.build import enrich_census, enrich_panel_temporal, make_features
    from fuel_pred.evaluate import compare as evaluate
    from fuel_pred.train import feature_blocks, train_models

    return config, enrich_census, enrich_panel_temporal, make_features, feature_blocks, train_models, evaluate


def _snapshot_originals(modules):
    """Capture original module-level constants so we can restore between runs."""
    config, eric, ept, mf, fb, _train, _evaluate = modules
    return {
        "config.CROSS": dict(config.AUGMENTOR_VARIABLES_CROSS_SECTIONAL),
        "config.TEMP": dict(config.AUGMENTOR_VARIABLES_TEMPORAL),
        "config.UNION": dict(config.AUGMENTOR_VARIABLES),
        "eric.DIRECT_VARIABLES": dict(eric.DIRECT_VARIABLES),
        "eric.ENRICHED_COLUMNS": eric.ENRICHED_COLUMNS,
        "ept.TEMPORAL_VARIABLES": dict(ept.TEMPORAL_VARIABLES),
        "ept.OUTPUT_VALUE_COLUMNS": ept.OUTPUT_VALUE_COLUMNS,
        "ept.OUTPUT_COLUMNS": ept.OUTPUT_COLUMNS,
        "fb.SA2_COLUMNS": fb.SA2_COLUMNS,
        "fb.BLOCK_COLUMNS_sa2": fb.BLOCK_COLUMNS["sa2"],
        "mf.SA2_FEATURE_COLS": mf.SA2_FEATURE_COLS,
    }


def _restore_originals(modules, originals):
    """Put the module-level constants back so the next experiment starts clean."""
    config, eric, ept, mf, fb, _train, _evaluate = modules
    config.AUGMENTOR_VARIABLES_CROSS_SECTIONAL = dict(originals["config.CROSS"])
    config.AUGMENTOR_VARIABLES_TEMPORAL = dict(originals["config.TEMP"])
    config.AUGMENTOR_VARIABLES = dict(originals["config.UNION"])
    eric.DIRECT_VARIABLES = dict(originals["eric.DIRECT_VARIABLES"])
    eric.ENRICHED_COLUMNS = originals["eric.ENRICHED_COLUMNS"]
    ept.TEMPORAL_VARIABLES = dict(originals["ept.TEMPORAL_VARIABLES"])
    ept.OUTPUT_VALUE_COLUMNS = originals["ept.OUTPUT_VALUE_COLUMNS"]
    ept.OUTPUT_COLUMNS = originals["ept.OUTPUT_COLUMNS"]
    fb.SA2_COLUMNS = originals["fb.SA2_COLUMNS"]
    fb.BLOCK_COLUMNS["sa2"] = originals["fb.BLOCK_COLUMNS_sa2"]
    mf.SA2_FEATURE_COLS = originals["mf.SA2_FEATURE_COLS"]


def _apply_experiment(modules, originals, exp: Experiment) -> dict:
    """Apply this experiment's deltas; return what changed for logging."""
    config, eric, ept, mf, fb, _train, _evaluate = modules

    new_cs = dict(originals["config.CROSS"])
    new_temp = dict(originals["config.TEMP"])

    # Move from CS → T
    for key in exp.move_to_temporal:
        if key not in new_cs:
            raise KeyError(f"experiment {exp.name}: key {key!r} not in baseline CROSS_SECTIONAL set")
        new_temp[key] = new_cs.pop(key)

    # Add brand-new CS entries
    for key, ref in exp.add_cross_sectional.items():
        if key in new_cs or key in new_temp:
            raise KeyError(f"experiment {exp.name}: key {key!r} already exists")
        new_cs[key] = ref

    # Write back to config + dependent modules
    config.AUGMENTOR_VARIABLES_CROSS_SECTIONAL = new_cs
    config.AUGMENTOR_VARIABLES_TEMPORAL = new_temp
    config.AUGMENTOR_VARIABLES = {**new_cs, **new_temp}

    eric.DIRECT_VARIABLES = dict(new_cs)
    eric.ENRICHED_COLUMNS = ("sa2_code", "sa2_name", *(f"sa2_{k}" for k in new_cs))

    ept.TEMPORAL_VARIABLES = dict(new_temp)
    ept.OUTPUT_VALUE_COLUMNS = tuple(f"sa2_{k}" for k in new_temp)
    ept.OUTPUT_COLUMNS = ("station_id", "date", *ept.OUTPUT_VALUE_COLUMNS)

    # SA2 model block
    sa2_extra = tuple(exp.add_to_sa2_model_block)
    new_sa2 = originals["fb.SA2_COLUMNS"] + sa2_extra
    fb.SA2_COLUMNS = new_sa2
    fb.BLOCK_COLUMNS["sa2"] = new_sa2
    mf.SA2_FEATURE_COLS = new_sa2

    return {
        "n_cross_sectional": len(new_cs),
        "n_temporal": len(new_temp),
        "n_sa2_model": len(new_sa2),
        "moved_to_temporal": list(exp.move_to_temporal),
        "added_cross_sectional": dict(exp.add_cross_sectional),
        "sa2_model_added": list(sa2_extra),
    }


def _run_experiment(modules, originals, exp: Experiment) -> dict:
    """Run one experiment end-to-end. Returns metrics dict."""
    config, eric, ept, mf, fb, train_models, evaluate = modules
    t_start = time.monotonic()

    # Per-experiment output paths
    exp_temporal_path = DATA_INTERIM / f"panel_sa2_temporal_{exp.name}.parquet"
    exp_features_path = DATA_PROCESSED / f"features_{exp.name}.parquet"
    exp_models_dir = REPO_ROOT / f"models_{exp.name}"
    exp_comparison_path = RESULTS_DIR / f"pr_c_{exp.name}_comparison.md"

    logger.info("=" * 70)
    logger.info("EXPERIMENT %s — %s", exp.name, exp.description)
    logger.info("=" * 70)

    # Apply config patches
    delta = _apply_experiment(modules, originals, exp)
    logger.info(
        "config delta: cross=%d temporal=%d sa2_model_cols=%d",
        delta["n_cross_sectional"],
        delta["n_temporal"],
        delta["n_sa2_model"],
    )
    if delta["moved_to_temporal"]:
        logger.info("moved to temporal: %s", delta["moved_to_temporal"])
    if delta["added_cross_sectional"]:
        logger.info("added cross-sectional: %s", list(delta["added_cross_sectional"]))
    if delta["sa2_model_added"]:
        logger.info("added to SA2_COLUMNS: %s", delta["sa2_model_added"])

    # ---- Stage 1: re-enrich cross-sectional (always — schema may have changed) ----
    t0 = time.monotonic()
    logger.info("[%s] stage 1: cross-sectional enrich...", exp.name)
    eric.enrich(STATIONS_PATH, STATIONS_PATH, data_dir=DATA_RAW, force=True)
    logger.info("[%s] stage 1 complete in %.1f min", exp.name, (time.monotonic() - t0) / 60)

    # ---- Stage 2: temporal enrich (skip if empty) ----
    if ept.TEMPORAL_VARIABLES:
        t0 = time.monotonic()
        logger.info(
            "[%s] stage 2: temporal enrich (%d vars)...",
            exp.name,
            len(ept.TEMPORAL_VARIABLES),
        )
        ept.enrich(PANEL_PATH, STATIONS_PATH, exp_temporal_path)
        logger.info(
            "[%s] stage 2 complete in %.1f min",
            exp.name,
            (time.monotonic() - t0) / 60,
        )
    else:
        exp_temporal_path = None
        logger.info("[%s] stage 2 skipped (no temporal variables)", exp.name)

    # ---- Stage 3: rebuild features ----
    t0 = time.monotonic()
    logger.info("[%s] stage 3: build features...", exp.name)
    mf.make_features_from_paths(
        panel_path=PANEL_PATH,
        out_path=exp_features_path,
        panel_sa2_temporal_path=exp_temporal_path,
    )
    logger.info("[%s] stage 3 complete in %.1f min", exp.name, (time.monotonic() - t0) / 60)

    # ---- Stage 4: train ----
    t0 = time.monotonic()
    logger.info("[%s] stage 4: train models...", exp.name)
    exp_models_dir.mkdir(exist_ok=True)
    train_models.train(features_path=exp_features_path, out_dir=exp_models_dir)
    logger.info("[%s] stage 4 complete in %.1f min", exp.name, (time.monotonic() - t0) / 60)

    # ---- Stage 5: evaluate ----
    t0 = time.monotonic()
    logger.info("[%s] stage 5: evaluate...", exp.name)
    evaluate.compare(
        features_path=exp_features_path,
        models_dir=exp_models_dir,
        out_path=exp_comparison_path,
    )
    logger.info("[%s] stage 5 complete in %.1f min", exp.name, (time.monotonic() - t0) / 60)

    # ---- Extract headline metrics from the comparison.md ----
    metrics = _extract_headline_metrics(exp_comparison_path)
    metrics["wall_clock_min"] = round((time.monotonic() - t_start) / 60, 1)
    logger.info("[%s] DONE — %s", exp.name, metrics)
    return metrics


def _extract_headline_metrics(comparison_path: Path) -> dict:
    """Pull the test_normal / test_crisis Δ MAE from a comparison.md.

    The comparison file's headline table follows the layout we control in
    evaluate.compare:

        | Fold | n | MAE A | MAE B | Δ MAE | RMSE A | RMSE B | MAPE A | MAPE B | Δ MAPE |
        | test_normal | ... | 6.373 | 6.134 | -0.239 | ... |
        | test_crisis | ... | 13.616 | 13.295 | -0.321 | ... |
    """
    text = comparison_path.read_text(encoding="utf-8")
    metrics: dict = {}
    for line in text.splitlines():
        for fold in ("test_normal", "test_crisis"):
            if line.startswith(f"| {fold} |"):
                parts = [p.strip() for p in line.split("|")]
                # parts: ["", "test_normal", "n", "MAE A", "MAE B", "Δ MAE", ...]
                try:
                    metrics[f"{fold}_mae_a"] = float(parts[3])
                    metrics[f"{fold}_mae_b"] = float(parts[4])
                    metrics[f"{fold}_delta_mae"] = float(parts[5])
                except (IndexError, ValueError):
                    logger.warning("failed to parse %s line: %r", fold, line)
    return metrics


def _write_summary(all_metrics: dict, baseline: dict) -> None:
    """Write results/pr_c_overnight_summary.md."""
    summary_path = RESULTS_DIR / "pr_c_overnight_summary.md"
    lines = [
        "# PR C overnight experiment summary",
        "",
        "Generated by `tools/research/pr_c_overnight_runner.py`. See "
        "`docs/research/2026-05_abs_census_augmentor_v2.0_review.md` for "
        "context on each experiment.",
        "",
        "## Baseline (committed PR B headline, for comparison)",
        "",
        "| Fold | MAE A | MAE B | Δ MAE |",
        "|------|------:|------:|------:|",
    ]
    for fold in ("test_normal", "test_crisis"):
        a = baseline.get(f"{fold}_mae_a", float("nan"))
        b = baseline.get(f"{fold}_mae_b", float("nan"))
        d = baseline.get(f"{fold}_delta_mae", float("nan"))
        lines.append(f"| {fold} | {a:.3f} | {b:.3f} | **{d:+.3f}** |")
    lines.extend([
        "",
        "## Experiments",
        "",
        "Δ MAE column: more-negative is better. Compare each experiment to "
        "the baseline above.",
        "",
        "| Experiment | test_normal Δ MAE | vs baseline | test_crisis Δ MAE | vs baseline | wall-clock |",
        "|------------|------------------:|------------:|------------------:|------------:|-----------:|",
    ])
    for exp in EXPERIMENTS:
        m = all_metrics.get(exp.name, {})
        if not m:
            lines.append(f"| **{exp.name}** | _(no data)_ | _(no data)_ | _(no data)_ | _(no data)_ | _(failed)_ |")
            continue
        d_normal = m.get("test_normal_delta_mae", float("nan"))
        d_crisis = m.get("test_crisis_delta_mae", float("nan"))
        baseline_normal = baseline.get("test_normal_delta_mae", float("nan"))
        baseline_crisis = baseline.get("test_crisis_delta_mae", float("nan"))
        diff_normal = d_normal - baseline_normal
        diff_crisis = d_crisis - baseline_crisis
        wall = m.get("wall_clock_min", float("nan"))
        lines.append(
            f"| **{exp.name}** "
            f"| {d_normal:+.3f} "
            f"| {diff_normal:+.3f} "
            f"| {d_crisis:+.3f} "
            f"| {diff_crisis:+.3f} "
            f"| {wall:.1f} min |"
        )
    lines.extend([
        "",
        "## Per-experiment artefacts",
        "",
        "| Experiment | Comparison report | Features parquet | Models dir |",
        "|------------|-------------------|------------------|------------|",
    ])
    for exp in EXPERIMENTS:
        lines.append(
            f"| {exp.name} "
            f"| `results/pr_c_{exp.name}_comparison.md` "
            f"| `data/processed/features_{exp.name}.parquet` "
            f"| `models_{exp.name}/` |"
        )
    lines.extend([
        "",
        "## Experiment descriptions",
        "",
    ])
    for exp in EXPERIMENTS:
        lines.append(f"- **{exp.name}** — {exp.description}")
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote summary to %s", summary_path)

    # Also dump the raw metrics as JSON for any follow-up analysis.
    raw_path = RESULTS_DIR / "pr_c_overnight_metrics.json"
    raw_path.write_text(
        json.dumps({"baseline": baseline, "experiments": all_metrics}, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote raw metrics to %s", raw_path)


def main() -> None:
    logger.info("PR C overnight orchestrator starting")
    logger.info("Repo root: %s", REPO_ROOT)
    logger.info("Experiments to run: %s", [e.name for e in EXPERIMENTS])

    modules = _import_pipeline_modules()
    originals = _snapshot_originals(modules)
    logger.info(
        "baseline config snapshot: cross=%d temporal=%d sa2_model=%d",
        len(originals["config.CROSS"]),
        len(originals["config.TEMP"]),
        len(originals["fb.SA2_COLUMNS"]),
    )

    # Capture the committed PR B baseline metrics from the existing
    # comparison.md (the headline file in results/) so the summary table
    # can show "this experiment beat / lost to the existing baseline".
    baseline_path = RESULTS_DIR / "comparison.md"
    baseline = _extract_headline_metrics(baseline_path) if baseline_path.exists() else {}
    logger.info("baseline (PR B headline): %s", baseline)

    all_metrics: dict = {}
    for exp in EXPERIMENTS:
        try:
            # Each experiment starts from a clean baseline config.
            _restore_originals(modules, originals)
            metrics = _run_experiment(modules, originals, exp)
            all_metrics[exp.name] = metrics
            # Write summary after every experiment so a mid-run failure
            # still leaves the user with partial data.
            _write_summary(all_metrics, baseline)
        except Exception as exc:
            logger.exception("experiment %s FAILED: %s", exp.name, exc)
            all_metrics[exp.name] = {"error": f"{type(exc).__name__}: {exc}"}
            _write_summary(all_metrics, baseline)
            # Keep going — the next experiment might succeed.

    # Final restore so the user wakes up to a clean working tree.
    _restore_originals(modules, originals)
    logger.info("PR C overnight orchestrator complete")
    logger.info("Summary: %s", RESULTS_DIR / "pr_c_overnight_summary.md")


if __name__ == "__main__":
    main()
