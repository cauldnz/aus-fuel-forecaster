"""v3.0 Phase 3 #4 follow-up — validate hyperopt winner across 6 seeds.

The Optuna sweep (``v3_phase3_hyperopt_runner.py``) returns ONE best params
combination evaluated at the spec default seed (42). Before locking the
new defaults into spec §8.2, we need to confirm the +0.20 c/L improvement
isn't a single-seed lucky-fit. Same protocol as Phase 3 #2's seed-noise
floor, but with the hyperopt winner's params instead of spec defaults.

Output:

1. Per-seed per-fold MAE_A under the new params.
2. Mean improvement vs the Phase 3 #2 baseline (spec defaults at same 6
   seeds), per fold and overall.
3. Across-seeds stdev of the per-fold improvement (the analogue of the
   published Δ MAE stdev). Validated if |mean improvement| > stdev.

Wall-clock estimate: ~6 min/seed × 6 seeds = ~36 min. Resume-safe via
audit.json check per seed (same pattern as Phase 3 #2).

Spec / discussion:
- `docs/research/2026-06_v3.0_phase2_postmortem_discussion.md` (Reading C1)
- `docs/research/2026-06_v3.0_phase3_closing_summary.md` (sets the
  "validate with 6× seed sanity check" protocol that this script
  implements)
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

SEEDS = (42, 1, 7, 13, 99, 123)
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"
HYPEROPT_JSON = RESULTS_DIR / "v3_phase3_hyperopt.json"
BASELINE_SEED_JSON = RESULTS_DIR / "v3_phase3_seed_noise.json"

LOG_PATH = REPO_ROOT / "tools" / "research" / "v3_phase3_hyperopt_validation.log"
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
logger = logging.getLogger("v3_phase3_hyperopt_val")


def _import_train_kfold() -> Callable[..., object]:
    """Late import so logging is set up first."""
    from fuel_pred.train.cv import train_kfold
    return train_kfold


def _load_best_params() -> dict[str, float]:
    """Pull the hyperopt winner's params from the v3_phase3_hyperopt.json."""
    if not HYPEROPT_JSON.exists():
        raise RuntimeError(
            f"{HYPEROPT_JSON} missing — run v3_phase3_hyperopt_runner.py first"
        )
    data = json.loads(HYPEROPT_JSON.read_text(encoding="utf-8"))
    best = data.get("best_params")
    if not best:
        raise RuntimeError(f"no best_params in {HYPEROPT_JSON}")
    logger.info("loaded hyperopt winner params: %s", best)
    logger.info("hyperopt best_value (seed=42 only): %.4f", data["best_value"])
    return best


def _run_seed(seed: int, best_params: dict[str, float]) -> dict[str, float]:
    """Train Model A only across all 6 folds at the given seed with hyperopt winner params."""
    out_root = REPO_ROOT / f"models_kfold_hyperopt_seed_{seed}"
    audit_path = out_root / "kfold_audit.json"

    if audit_path.exists():
        logger.info("[seed=%d] SKIP — audit already exists at %s", seed, audit_path)
        per_fold = _per_fold_mae_a_from_audit(audit_path)
        per_fold["wall_clock_min"] = 0.0
        per_fold["resumed_from_cache"] = True
        return per_fold

    if not FEATURES_PARQUET.exists():
        raise RuntimeError(f"features parquet missing: {FEATURES_PARQUET}")

    # Monkey-patch config.LGBM_PARAMS so _train_one_fold picks up the
    # hyperopt winner params. Restore in finally — same pattern as the
    # hyperopt runner itself.
    from fuel_pred import config
    snapshot = dict(config.LGBM_PARAMS)
    config.LGBM_PARAMS.update(best_params)
    # Keep n_estimators at spec default 2000 (not the hyperopt cap of
    # 1500) — early stopping handles termination, and the spec config
    # is what would ship.

    logger.info("=" * 70)
    logger.info("[seed=%d] training Model A across 6 folds with hyperopt winner params", seed)
    logger.info("[seed=%d] monkey-patched LGBM_PARAMS: %s", seed, best_params)
    logger.info("=" * 70)

    try:
        train_kfold = _import_train_kfold()
        t0 = time.monotonic()
        train_kfold(
            FEATURES_PARQUET,
            out_root,
            random_state=seed,
            models_to_fit=("A",),
            save_predictions=False,
        )
        wall_min = (time.monotonic() - t0) / 60
    finally:
        config.LGBM_PARAMS.clear()
        config.LGBM_PARAMS.update(snapshot)

    logger.info("[seed=%d] train_kfold complete in %.1f min", seed, wall_min)

    per_fold = _per_fold_mae_a_from_audit(audit_path)
    per_fold["wall_clock_min"] = round(wall_min, 1)
    per_fold["resumed_from_cache"] = False
    return per_fold


def _per_fold_mae_a_from_audit(audit_path: Path) -> dict[str, float]:
    """Same helper as v3_phase3_seed_noise_runner.py — pulls per-fold best_val_mae for Model A."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for entry in audit.get("folds", []):
        fold_n = int(entry["fold"])
        mae = entry["models"]["A"]["best_val_mae"]
        out[f"fold_{fold_n}"] = float(mae)
    if len(out) != 6:
        raise RuntimeError(f"expected 6 folds in {audit_path}, got {len(out)}")
    return out


def _load_baseline_per_seed() -> dict[int, dict[str, float]]:
    """Pull Phase 3 #2's seed-noise per-seed-per-fold MAE_A values (spec defaults)."""
    if not BASELINE_SEED_JSON.exists():
        raise RuntimeError(
            f"{BASELINE_SEED_JSON} missing — run v3_phase3_seed_noise_runner.py first"
        )
    data = json.loads(BASELINE_SEED_JSON.read_text(encoding="utf-8"))
    raw = data["per_seed_per_fold_mae_a"]
    # JSON keys are strings; convert to int
    return {int(k): {fk: float(v) for fk, v in d.items() if fk.startswith("fold_")}
            for k, d in raw.items()}


def _write_summary(
    hyperopt_per_seed: dict[int, dict[str, float]],
    baseline_per_seed: dict[int, dict[str, float]],
    best_params: dict[str, float],
) -> None:
    """Write the validation summary: per-fold seed-stdev under new params, and
    per-fold mean improvement vs the spec-default baseline.
    """
    summary_md = RESULTS_DIR / "v3_phase3_hyperopt_validation.md"
    summary_json = RESULTS_DIR / "v3_phase3_hyperopt_validation.json"

    fold_keys = [f"fold_{i+1}" for i in range(6)]

    # ---- Per-fold seed-stdev of MAE_A under new params ----
    per_fold_new_stdev: dict[str, float] = {}
    per_fold_new_mean: dict[str, float] = {}
    for fk in fold_keys:
        vals = [d[fk] for d in hyperopt_per_seed.values() if fk in d]
        if len(vals) >= 2:
            per_fold_new_mean[fk] = statistics.fmean(vals)
            per_fold_new_stdev[fk] = statistics.pstdev(vals)

    # ---- Per-fold seed-mean improvement: new mean vs baseline mean ----
    per_fold_improvement: dict[str, float] = {}
    per_fold_baseline_mean: dict[str, float] = {}
    for fk in fold_keys:
        base_vals = [d[fk] for d in baseline_per_seed.values() if fk in d]
        if len(base_vals) >= 2 and fk in per_fold_new_mean:
            per_fold_baseline_mean[fk] = statistics.fmean(base_vals)
            per_fold_improvement[fk] = per_fold_new_mean[fk] - per_fold_baseline_mean[fk]

    # ---- Across-folds stdev of per-fold improvement (the analogue of Δ MAE stdev) ----
    improvement_values = list(per_fold_improvement.values())
    mean_improvement = statistics.fmean(improvement_values) if improvement_values else float("nan")
    stdev_improvement = statistics.pstdev(improvement_values) if len(improvement_values) >= 2 else float("nan")

    # ---- Per-seed paired diff: hyperopt(seed_i) − baseline(seed_i) across folds ----
    # For each seed in BOTH dicts, compute per-fold delta. Mean improvement per
    # seed, then mean across seeds.
    paired_seed_improvements: list[dict[str, float]] = []
    for seed in SEEDS:
        if seed not in hyperopt_per_seed or seed not in baseline_per_seed:
            continue
        h = hyperopt_per_seed[seed]
        b = baseline_per_seed[seed]
        deltas = {fk: h[fk] - b[fk] for fk in fold_keys if fk in h and fk in b}
        mean_delta = statistics.fmean(deltas.values()) if deltas else float("nan")
        stdev_delta = statistics.pstdev(deltas.values()) if len(deltas) >= 2 else float("nan")
        paired_seed_improvements.append({
            "seed": seed,
            "per_fold_delta": deltas,
            "mean_delta_across_folds": mean_delta,
            "stdev_delta_across_folds": stdev_delta,
        })

    # ---- Verdict ----
    # Validated if:
    # (a) mean improvement across folds is NEGATIVE (i.e. new params reduce MAE) AND
    # (b) |mean improvement| > stdev improvement (the v3.0 design-doc significance heuristic)
    abs_mean = abs(mean_improvement) if mean_improvement < 0 else 0.0
    validated_significance = mean_improvement < 0 and abs_mean > stdev_improvement
    # Stronger: |mean| > 2 * stdev = robust
    robust = mean_improvement < 0 and abs_mean > 2 * stdev_improvement

    # ---- JSON payload ----
    payload = {
        "seeds": list(hyperopt_per_seed.keys()),
        "hyperopt_winner_params": best_params,
        "per_seed_per_fold_mae_a_NEW": hyperopt_per_seed,
        "per_fold_seed_stats_NEW": {
            fk: {"mean": per_fold_new_mean[fk], "stdev": per_fold_new_stdev[fk]}
            for fk in fold_keys if fk in per_fold_new_stdev
        },
        "per_fold_baseline_mean_SPEC_DEFAULT": per_fold_baseline_mean,
        "per_fold_improvement_NEW_minus_BASELINE": per_fold_improvement,
        "mean_improvement_across_folds": mean_improvement,
        "stdev_improvement_across_folds": stdev_improvement,
        "paired_seed_improvements": paired_seed_improvements,
        "verdict": {
            "validated_significance": validated_significance,
            "robust": robust,
            "criterion": "mean_improvement<0 AND |mean| > stdev across folds",
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_json)

    # ---- Markdown summary ----
    lines: list[str] = []
    lines.append("# v3.0 Phase 3 #4 validation — hyperopt winner across 6 seeds")
    lines.append("")
    lines.append(
        "Re-runs Model A across all 6 k-fold folds, 6 times with different "
        "LightGBM `random_state` values, using the hyperopt winner's "
        "hyperparameters (trial 15 of run 2). Confirms the +0.20 c/L "
        "improvement over spec §8.2 defaults isn't a single-seed lucky-fit."
    )
    lines.append("")
    lines.append("## Hyperopt winner params (under test)")
    lines.append("")
    lines.append("| Param | Value | Spec §8.2 default |")
    lines.append("|-------|------:|------------------:|")
    spec_defaults = {
        "num_leaves": 63, "min_data_in_leaf": 200, "learning_rate": 0.05,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "lambda_l1": 0.0, "lambda_l2": 0.0,
    }
    for k in ("num_leaves", "min_data_in_leaf", "learning_rate",
              "feature_fraction", "bagging_fraction", "bagging_freq",
              "lambda_l1", "lambda_l2"):
        v = best_params.get(k)
        d = spec_defaults.get(k, "—")
        v_s = f"{v:.4g}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {v_s} | {d} |")
    lines.append("")

    lines.append("## Per-seed per-fold MAE_A under new params")
    lines.append("")
    lines.append("| Seed | " + " | ".join(fold_keys) + " | Wall-clock |")
    lines.append("|------|" + "|".join(["---" for _ in fold_keys]) + "|----|")
    for seed in hyperopt_per_seed:
        d = hyperopt_per_seed[seed]
        row = [f"{d.get(fk, float('nan')):.4f}" for fk in fold_keys]
        wall = d.get("wall_clock_min", float("nan"))
        cache = " (cached)" if d.get("resumed_from_cache") else ""
        lines.append(f"| {seed} | " + " | ".join(row) + f" | {wall:.1f} min{cache} |")
    lines.append("")

    lines.append("## Per-fold improvement: new mean − baseline mean (seed-averaged)")
    lines.append("")
    lines.append("Negative = new params are better. Both means are averages across the same 6 seeds.")
    lines.append("")
    lines.append("| Fold | Baseline (spec) mean | New params mean | Δ (improvement) | Seed-stdev (new) |")
    lines.append("|------|---------------------:|----------------:|----------------:|-----------------:|")
    for fk in fold_keys:
        if fk not in per_fold_improvement:
            continue
        base = per_fold_baseline_mean[fk]
        new = per_fold_new_mean[fk]
        d = per_fold_improvement[fk]
        s = per_fold_new_stdev.get(fk, float("nan"))
        lines.append(f"| {fk} | {base:.4f} | {new:.4f} | {d:+.4f} | {s:.4f} |")
    lines.append(f"| **Mean across folds** | — | — | **{mean_improvement:+.4f}** | — |")
    lines.append(f"| **Stdev across folds** | — | — | **{stdev_improvement:.4f}** | — |")
    lines.append("")

    lines.append("## Paired-seed improvements (each seed under new vs same seed under spec defaults)")
    lines.append("")
    lines.append("Stronger test: per-seed, compare the same seed under both param sets. Removes "
                 "seed effect from the comparison entirely.")
    lines.append("")
    lines.append("| Seed | Mean Δ across folds | Stdev Δ across folds |")
    lines.append("|------|--------------------:|---------------------:|")
    for s in paired_seed_improvements:
        lines.append(f"| {s['seed']} | {s['mean_delta_across_folds']:+.4f} | {s['stdev_delta_across_folds']:.4f} |")
    paired_mean_means = [s["mean_delta_across_folds"] for s in paired_seed_improvements]
    if paired_mean_means:
        pm_mean = statistics.fmean(paired_mean_means)
        pm_stdev = statistics.pstdev(paired_mean_means) if len(paired_mean_means) >= 2 else float("nan")
        lines.append(f"| **Mean across seeds** | **{pm_mean:+.4f}** | (stdev of per-seed means: **{pm_stdev:.4f}**) |")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- **Mean improvement across folds: {mean_improvement:+.4f} c/L**")
    lines.append(f"- **Stdev improvement across folds: {stdev_improvement:.4f} c/L**")
    lines.append("")
    if robust:
        lines.append("**ROBUST WIN.** |Mean improvement| > 2 × Stdev. The new hyperparameters "
                     "deliver a real, multi-seed-stable improvement over spec §8.2 defaults. "
                     "**Action: update spec §8.2 with these defaults; ship Model A on new params.**")
    elif validated_significance:
        lines.append("**WEAK WIN.** |Mean improvement| > Stdev but < 2 × Stdev. The improvement "
                     "is consistent in sign across folds + seeds but not overwhelming. **Action: "
                     "update spec §8.2 with the new defaults but caveat that the improvement is "
                     "weak-band; document the per-fold spread.**")
    elif mean_improvement < 0:
        lines.append("**MARGINAL.** Mean improvement is in the right direction but |Mean| < Stdev. "
                     "The original 0.20 c/L improvement at seed=42 was likely a single-seed lucky "
                     "fit. **Action: keep spec §8.2 defaults; document that the hyperopt search "
                     "did not produce robust improvements.**")
    else:
        lines.append("**REGRESSION.** Mean improvement is positive (new params worse) across seeds. "
                     "**Action: keep spec §8.2 defaults absolutely; the hyperopt winner doesn't "
                     "generalise.**")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append("- `tools/research/v3_phase3_hyperopt_validation.py` — this script")
    lines.append("- `results/v3_phase3_hyperopt.json` — hyperopt winner params")
    lines.append("- `results/v3_phase3_seed_noise.json` — baseline per-seed per-fold MAE (spec defaults)")
    lines.append("- `docs/research/2026-06_v3.0_phase3_closing_summary.md` — sets the validation protocol")
    lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", summary_md)

    # Headline to console
    logger.info("=== Hyperopt validation headline ===")
    logger.info("Mean improvement across folds (new − baseline): %+.4f c/L", mean_improvement)
    logger.info("Stdev improvement across folds:                  %.4f c/L", stdev_improvement)
    if robust:
        logger.info("Verdict: ROBUST WIN — update spec §8.2")
    elif validated_significance:
        logger.info("Verdict: WEAK WIN — update spec §8.2 with caveat")
    elif mean_improvement < 0:
        logger.info("Verdict: MARGINAL — keep spec §8.2 defaults")
    else:
        logger.info("Verdict: REGRESSION — keep spec §8.2 defaults")


def main() -> None:
    logger.info("v3.0 Phase 3 #4 validation starting (%d seeds)", len(SEEDS))

    best_params = _load_best_params()
    baseline_per_seed = _load_baseline_per_seed()
    logger.info("loaded baseline (spec defaults) per-seed per-fold MAE for %d seeds",
                len(baseline_per_seed))

    hyperopt_per_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        try:
            hyperopt_per_seed[seed] = _run_seed(seed, best_params)
        except Exception as exc:
            logger.exception("seed=%d FAILED: %s", seed, exc)
            hyperopt_per_seed[seed] = {"error": f"{type(exc).__name__}: {exc}"}  # type: ignore[dict-item]
        # Write partial summary after every seed
        valid = {s: d for s, d in hyperopt_per_seed.items() if "error" not in d}
        if len(valid) >= 2:
            _write_summary(valid, baseline_per_seed, best_params)

    logger.info("v3.0 Phase 3 #4 validation complete")


if __name__ == "__main__":
    main()
