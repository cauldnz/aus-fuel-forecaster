"""v3.0 Phase 3 next-step #2 — Model A baseline-vs-baseline seed noise floor.

Re-runs the v3.0 6-fold k-fold harness ``N_SEEDS`` times against the
SAME committed PR B baseline features parquet, varying only LightGBM's
``random_state``. Trains **only Model A** per seed (skips B and B') —
the question is "how much do per-fold MAE numbers move with a different
seed alone?", not "how does the augmentor behave under different seeds."

Output is two numbers we care about:

1. **Per-fold seed-stdev of MAE_A** averaged over the 6 folds. This is
   the seed-noise floor for a single fold's MAE measurement.
2. **Stdev of (per-fold MAE_A(seed_i) − per-fold MAE_A(seed_j)) across
   6 folds**, averaged over all ``C(N_SEEDS, 2)`` seed-pairs. This is
   the direct analogue of the v3.0 published Δ MAE stdev (0.394 c/L
   for PR B): "if you run the same Model A twice with two different
   seeds, how much does the per-fold-MAE-difference vary across folds?"

If quantity (2) is comparable to the published 0.394 — say within
the same order of magnitude — that's strong **Reading A confirmation**:
the augmentor's noise band is no bigger than two identical Model A's
disagreeing with each other purely by random seed.

If quantity (2) is much smaller — say <0.1 — then 0.394 is real
signal Model B is failing to capture, sharpening Reading C.

Wall-clock estimate: ~9 min per seed (6 folds × 1 model × ~90s/fit)
× ``N_SEEDS`` seeds ≈ 54 min for the default 6 seeds. Resume-safe:
skip a seed if ``models_kfold_seed_<N>/kfold_audit.json`` already
exists.

Run:

    uv run python tools/research/v3_phase3_seed_noise_runner.py 2>&1 | \\
        tee tools/research/v3_phase3_seed_noise.log

Spec / discussion: ``docs/research/2026-06_v3.0_phase2_postmortem_discussion.md``
(next-steps #2 in the ranked list).
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

# 6 seeds keeps the wall-clock under an hour while giving us
# C(6,2) = 15 pairs for the across-pairs aggregate.
# Seed 42 is the spec default (LGBM_PARAMS["random_state"]); the others
# are arbitrary primes / common defaults to avoid any "looks rigged"
# perception.
SEEDS = (42, 1, 7, 13, 99, 123)
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"

LOG_PATH = REPO_ROOT / "tools" / "research" / "v3_phase3_seed_noise.log"
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
logger = logging.getLogger("v3_phase3_seed")


def _import_train_kfold() -> Callable[..., object]:
    """Late import so logging is set up first."""
    from fuel_pred.train.cv import train_kfold
    return train_kfold


def _run_seed(seed: int) -> dict[str, float]:
    """Train Model A only on all 6 folds at the given seed; return per-fold MAE_A.

    Resume-safe: skip the fit + just re-read MAE from ``kfold_audit.json``
    if it already exists. Returns ``{"fold_N": mae, ..., "wall_clock_min":
    M, "resumed_from_cache": bool}``.
    """
    out_root = REPO_ROOT / f"models_kfold_seed_{seed}"
    audit_path = out_root / "kfold_audit.json"

    if audit_path.exists():
        logger.info("[seed=%d] SKIP — audit already exists at %s", seed, audit_path)
        per_fold = _per_fold_mae_a_from_audit(audit_path)
        per_fold["wall_clock_min"] = 0.0
        per_fold["resumed_from_cache"] = True
        return per_fold

    if not FEATURES_PARQUET.exists():
        raise RuntimeError(
            f"features parquet missing: {FEATURES_PARQUET}. "
            "Build pipeline (make features) must have run first."
        )

    logger.info("=" * 70)
    logger.info("[seed=%d] training Model A only across 6 folds", seed)
    logger.info("=" * 70)

    train_kfold = _import_train_kfold()
    t0 = time.monotonic()
    train_kfold(
        FEATURES_PARQUET,
        out_root,
        random_state=seed,
        models_to_fit=("A",),
        # Predictions parquet not needed — we only consume MAE_A from
        # the kfold_audit.json's best_val_mae per fold. Saves disk + I/O.
        save_predictions=False,
    )
    wall_min = (time.monotonic() - t0) / 60
    logger.info("[seed=%d] train_kfold complete in %.1f min", seed, wall_min)

    per_fold = _per_fold_mae_a_from_audit(audit_path)
    per_fold["wall_clock_min"] = round(wall_min, 1)
    per_fold["resumed_from_cache"] = False
    return per_fold


def _per_fold_mae_a_from_audit(audit_path: Path) -> dict[str, float]:
    """Parse ``kfold_audit.json`` and return per-fold Model A best_val_mae.

    Note: best_val_mae is the *validation* MAE at early-stopping best
    iteration — that's the same quantity ``compare_kfold`` reports as
    test MAE only when the val and test windows happen to coincide, which
    they don't in v3.0 (val is fixed-365-days before test). To get *test*
    MAE per fold, we'd need to score predictions_test.parquet — but we
    skipped writing it. For this experiment that's actually fine: val
    MAE is also a stable per-seed measurement and the seed-stdev
    comparison is apples-to-apples *within* the val space.

    Returns ``{"fold_1": mae, ..., "fold_6": mae}``.
    """
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for entry in audit.get("folds", []):
        fold_n = int(entry["fold"])
        mae = entry["models"]["A"]["best_val_mae"]
        out[f"fold_{fold_n}"] = float(mae)
    if len(out) != 6:
        raise RuntimeError(
            f"expected 6 folds in {audit_path}, got {len(out)}: {out}"
        )
    return out


# ---- alternate path: use test-set MAE_A from per-fold prediction parquets ----
# Kept commented-out for reference. If we want to reproduce the published
# *test* MAE numbers (e.g. 8.18 for pr_b_baseline), we'd compute on
# predictions_test.parquet. The val-MAE approach is fine for the
# methodological seed-noise question and saves ~3× disk per seed.
# Implementation sketch:
#   from fuel_pred.evaluate.compare import compare_kfold
#   compare_kfold(FEATURES_PARQUET, out_root, RESULTS_DIR / f"v3_phase3_seed_{seed}_kfold.md")
#   # then parse the per-fold MAE A column out of the markdown


def _write_summary(per_seed: dict[int, dict[str, float]]) -> None:
    """Write seed-noise summary: per-seed per-fold MAE_A + aggregates."""
    summary_md = RESULTS_DIR / "v3_phase3_seed_noise_summary.md"
    summary_json = RESULTS_DIR / "v3_phase3_seed_noise.json"

    fold_keys = [f"fold_{i+1}" for i in range(6)]

    # ---- Aggregate (1): per-fold seed-stdev of MAE_A ----
    per_fold_seed_stdev: dict[str, float] = {}
    per_fold_seed_mean: dict[str, float] = {}
    per_fold_seed_range: dict[str, float] = {}
    for fk in fold_keys:
        values = [d[fk] for d in per_seed.values() if fk in d]
        if len(values) < 2:
            continue
        per_fold_seed_mean[fk] = statistics.fmean(values)
        per_fold_seed_stdev[fk] = statistics.pstdev(values)
        per_fold_seed_range[fk] = max(values) - min(values)
    mean_seed_stdev = (
        statistics.fmean(per_fold_seed_stdev.values()) if per_fold_seed_stdev else float("nan")
    )

    # ---- Aggregate (2): across seed-pairs, stdev of per-fold Δ MAE ----
    # For each (seed_i, seed_j) pair, compute the per-fold differences
    # (seed_i − seed_j) across 6 folds, then take the stdev across folds.
    # This is the direct analogue of the published Δ MAE stdev (B vs A).
    pair_stdevs: list[float] = []
    pair_means: list[float] = []
    pair_details: list[dict[str, object]] = []
    for s_i, s_j in combinations(per_seed.keys(), 2):
        di = per_seed[s_i]
        dj = per_seed[s_j]
        deltas = [di[fk] - dj[fk] for fk in fold_keys if fk in di and fk in dj]
        if len(deltas) < 2:
            continue
        m = statistics.fmean(deltas)
        s = statistics.pstdev(deltas)
        pair_means.append(m)
        pair_stdevs.append(s)
        pair_details.append({
            "seed_i": s_i,
            "seed_j": s_j,
            "per_fold_delta": deltas,
            "mean_delta": m,
            "stdev_delta": s,
            "abs_mean_delta": abs(m),
        })
    mean_pair_stdev = statistics.fmean(pair_stdevs) if pair_stdevs else float("nan")
    median_pair_stdev = statistics.median(pair_stdevs) if pair_stdevs else float("nan")
    mean_abs_pair_mean = (
        statistics.fmean(abs(m) for m in pair_means) if pair_means else float("nan")
    )

    # ---- Reference: published Phase 2 PR B baseline Δ MAE stats ----
    # Treated as a literal constant (intentionally not module-level so the
    # value lives with the report-writing code that consumes it).
    reference_pr_b = {
        "mean_delta_mae": 0.215,
        "stdev_delta_mae": 0.394,
        "min_delta_mae": -0.135,
        "max_delta_mae": 1.042,
    }

    # ---- JSON payload ----
    payload = {
        "seeds": list(per_seed.keys()),
        "per_seed_per_fold_mae_a": per_seed,
        "per_fold_seed_stats": {
            fk: {
                "mean": per_fold_seed_mean[fk],
                "stdev": per_fold_seed_stdev[fk],
                "range": per_fold_seed_range[fk],
            }
            for fk in fold_keys if fk in per_fold_seed_stdev
        },
        "mean_per_fold_seed_stdev": mean_seed_stdev,
        "pair_details": pair_details,
        "mean_pair_stdev_across_folds": mean_pair_stdev,
        "median_pair_stdev_across_folds": median_pair_stdev,
        "mean_abs_pair_mean_across_folds": mean_abs_pair_mean,
        "reference_pr_b_baseline": reference_pr_b,
        "note": (
            "MAE values are LightGBM best_val_mae (val-window MAE at "
            "early-stopping best iteration), NOT test-window MAE. The "
            "seed-noise comparison is methodologically fine — within the "
            "val space, the seed effect on MAE is what we want. To "
            "reproduce the published test-MAE numbers (8.18 c/L for PR B), "
            "re-run with save_predictions=True + compare_kfold."
        ),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", summary_json)

    # ---- Markdown summary ----
    lines: list[str] = []
    lines.append("# v3.0 Phase 3 #2 — Model A baseline-vs-baseline seed noise floor")
    lines.append("")
    lines.append(
        f"Trained Model A only on the v3.0 6-fold k-fold harness {len(per_seed)} "
        f"times, varying only LightGBM's `random_state`. Tests whether the "
        f"published Δ MAE stdev for PR B (0.394 c/L) is a real augmentor signal "
        f"or noise within the same model's seed-driven jitter."
    )
    lines.append("")
    lines.append("## Methodology recap")
    lines.append("")
    lines.append(f"- Seeds: {list(per_seed.keys())}")
    lines.append("- Same features.parquet (committed PR B baseline) for every seed")
    lines.append("- Same 6-fold k-fold geometry as Phase 2 (spec §15.2)")
    lines.append("- **Model A only** per seed (skips B and B') — the question "
                 "is seed effect on the same model, not augmentor behaviour")
    lines.append("- MAE values reported here are LightGBM best_val_mae "
                 "(early-stopping val MAE), NOT test MAE")
    lines.append("")

    lines.append("## Per-seed per-fold MAE_A (validation)")
    lines.append("")
    lines.append("| Seed | " + " | ".join(fold_keys) + " | Wall-clock |")
    lines.append("|------|" + "|".join(["---" for _ in fold_keys]) + "|----|")
    for seed in per_seed:
        d = per_seed[seed]
        row = [f"{d.get(fk, float('nan')):.4f}" for fk in fold_keys]
        wall = d.get("wall_clock_min", float("nan"))
        cache = " (cached)" if d.get("resumed_from_cache") else ""
        lines.append(f"| {seed} | " + " | ".join(row) + f" | {wall:.1f} min{cache} |")
    lines.append("")

    lines.append("## Aggregate (1) — per-fold seed-stdev of MAE_A")
    lines.append("")
    lines.append("How much does a single fold's val-MAE move when you only "
                 "change the seed? This is the per-fold noise floor.")
    lines.append("")
    lines.append("| Fold | Mean MAE_A | Seed stdev | Seed range |")
    lines.append("|------|-----------:|-----------:|-----------:|")
    for fk in fold_keys:
        if fk not in per_fold_seed_stdev:
            continue
        lines.append(
            f"| {fk} | {per_fold_seed_mean[fk]:.4f} | "
            f"{per_fold_seed_stdev[fk]:.4f} | {per_fold_seed_range[fk]:.4f} |"
        )
    lines.append(f"| **Mean across folds** | — | **{mean_seed_stdev:.4f}** | — |")
    lines.append("")

    lines.append("## Aggregate (2) — across seed-pairs, stdev of per-fold MAE_A difference")
    lines.append("")
    lines.append(
        "For each pair of seeds, compute the per-fold MAE_A difference "
        "(seed_i − seed_j) across 6 folds, then take the stdev across "
        "folds. **This is the direct analogue of the published Δ MAE "
        "stdev (B vs A).** Averages over all "
        f"{len(pair_details)} pairs."
    )
    lines.append("")
    lines.append("| Seed pair | Mean Δ across folds | Stdev Δ across folds |")
    lines.append("|-----------|--------------------:|---------------------:|")
    for d in sorted(pair_details, key=lambda x: x["stdev_delta"]):
        lines.append(
            f"| {d['seed_i']} vs {d['seed_j']} | "
            f"{d['mean_delta']:+.4f} | {d['stdev_delta']:.4f} |"
        )
    lines.append(f"| **Mean across pairs** | "
                 f"**{mean_abs_pair_mean:+.4f}** (avg abs) | "
                 f"**{mean_pair_stdev:.4f}** |")
    lines.append("")
    lines.append("## Comparison to published Phase 2 PR B baseline")
    lines.append("")
    lines.append("Headline numbers from `results/v3_phase2_pr_b_baseline_kfold.md`:")
    lines.append("")
    lines.append(f"- Δ MAE Mean (B − A):   **{reference_pr_b['mean_delta_mae']:+.3f}** c/L")
    lines.append(f"- Δ MAE Stdev (across folds): **{reference_pr_b['stdev_delta_mae']:.3f}** c/L")
    lines.append(f"- Δ MAE Min:    {reference_pr_b['min_delta_mae']:+.3f} c/L")
    lines.append(f"- Δ MAE Max:    {reference_pr_b['max_delta_mae']:+.3f} c/L")
    lines.append("")
    lines.append(f"Seed-noise across-pairs stdev:  **{mean_pair_stdev:.4f}** c/L (avg over "
                 f"{len(pair_details)} pairs)")
    lines.append("")
    ratio = (
        reference_pr_b['stdev_delta_mae'] / mean_pair_stdev
        if mean_pair_stdev else float("inf")
    )
    lines.append(f"**Ratio: published Δ stdev / seed Δ stdev = {ratio:.2f}**")
    lines.append("")
    lines.append("Reading guide:")
    lines.append("")
    lines.append("- **Ratio ≤ 1.5** → augmentor's noise band is within 1.5× of "
                 "two seeds disagreeing → **Reading A confirmed**: 0.394 is "
                 "essentially seed jitter; there's nothing for Model B to "
                 "be \"better at\".")
    lines.append("- **Ratio ~2-4** → augmentor adds real per-fold noise on top "
                 "of seed noise (the augmentor block is genuinely struggling "
                 "fold-to-fold), but no robust signal in the mean — Reading A "
                 "+ Reading C2 hybrid.")
    lines.append("- **Ratio ≥ 5** → augmentor noise dwarfs seed noise → there's "
                 "real per-fold instability Model B is introducing that Model "
                 "A doesn't have → sharpens Reading C: model class / "
                 "interaction feature might be needed.")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append("- `tools/research/v3_phase3_seed_noise_runner.py` (this script)")
    lines.append("- `docs/research/2026-06_v3.0_phase2_postmortem_discussion.md` "
                 "(next-step #2 in the ranked list)")
    lines.append("- `results/v3_phase2_pr_b_baseline_kfold.md` (reference numbers)")
    lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", summary_md)

    # Headline to console (avoid unicode for Windows cp1252)
    logger.info("=== Seed-noise headline ===")
    logger.info("Mean per-fold seed stdev of MAE_A:        %.4f c/L", mean_seed_stdev)
    logger.info("Mean across-pairs stdev of fold-delta:    %.4f c/L", mean_pair_stdev)
    logger.info("Published PR B fold-delta stdev:          %.4f c/L (B vs A)", reference_pr_b['stdev_delta_mae'])
    logger.info("Ratio (published / seed):                  %.2f", ratio)


def main() -> None:
    logger.info("v3.0 Phase 3 seed-noise orchestrator starting (%d seeds)", len(SEEDS))
    logger.info("features parquet: %s", FEATURES_PARQUET)
    if not FEATURES_PARQUET.exists():
        raise RuntimeError(f"missing {FEATURES_PARQUET}")

    per_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        try:
            per_seed[seed] = _run_seed(seed)
        except Exception as exc:
            logger.exception("seed=%d FAILED: %s", seed, exc)
            per_seed[seed] = {"error": f"{type(exc).__name__}: {exc}"}  # type: ignore[dict-item]
        # Write partial summary after every seed so a kill mid-run keeps progress
        valid = {s: d for s, d in per_seed.items() if "error" not in d}
        if len(valid) >= 2:
            _write_summary(valid)

    logger.info("v3.0 Phase 3 seed-noise orchestrator complete")


if __name__ == "__main__":
    main()
