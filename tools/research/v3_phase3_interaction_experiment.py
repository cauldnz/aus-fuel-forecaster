"""v3.0 Phase 3 next-step #3 — explicit SEIFA × day-of-fortnight interaction.

Direct test of the Reading-C2 hypothesis from
``2026-06_v3.0_phase2_postmortem_discussion.md``:

> GBM with num_leaves=63 trees CAN find depth-3 interactions but won't
> unless both parents are individually high-gain (SEIFA isn't). Test:
> add ``sa2_seifa_irsd_score * cal_day_of_fortnight``, re-evaluate
> under k-fold.

This is the cleanest single-feature test of the Centrelink-fortnightly-
cycle × SEIFA-disadvantage interaction the v2.x SHAP-interaction probe
flagged as "null result". Spec §13 #3 hypothesised it; PR C tried
broader curation instead. This script tests it explicitly.

Pipeline:

1. Load committed ``features.parquet``.
2. Compute new column ``sa2_seifa_x_dof =
   sa2_seifa_irsd_score * cal_day_of_fortnight``. Raw product — no
   centering. LightGBM is scale-invariant.
3. Persist to ``data/processed/features_e6_seifa_dof_interaction.parquet``.
4. Monkey-patch ``feature_blocks.SA2_COLUMNS`` to add the new col so
   Model B (but not Model A) consumes it.
5. Run ``train_kfold`` (A + B only; skip B' — venue block isn't the
   experiment) at the spec default seed (42).
6. Run ``compare_kfold`` → write
   ``results/v3_phase3_e6_seifa_dof_interaction_kfold.md``.
7. Report headline: per-fold Δ MAE + aggregate stats + the new
   column's gain importance rank in Model B's feature list.

What to look for:

- **Robust win** (|Mean Δ| > 2×Stdev, Mean negative): the explicit
  interaction column IS the missing structural feature. Reading C2
  confirmed; revisit explicit interaction features systematically.
- **Same noise band as PR B** (mean near +0.215, stdev near 0.394):
  interaction adds nothing. Reading C2 falsified; lag features already
  capture this implicitly via per-station effects.
- **High gain on the new col but no Δ MAE improvement**: the model
  USES the new feature but it doesn't generalise — bias-variance shift
  rather than capacity unlock.

Wall-clock: ~9 min for train (A + B only, 6 folds × 2 models × ~45s/fit)
+ ~2 min for compare = ~11 min total. Faster than a Phase 2
experiment because B' is skipped.

Run:

    uv run python tools/research/v3_phase3_interaction_experiment.py 2>&1 | \\
        tee tools/research/v3_phase3_interaction.log

Spec / discussion: ``docs/research/2026-06_v3.0_phase2_postmortem_discussion.md``
(experiment #3 in the ranked next-steps list).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

EXPERIMENT_NAME = "e6_seifa_dof_interaction"
INTERACTION_COL = "sa2_seifa_x_dof"

INPUT_FEATURES = DATA_PROCESSED / "features.parquet"
OUTPUT_FEATURES = DATA_PROCESSED / f"features_{EXPERIMENT_NAME}.parquet"
OUTPUT_MODELS = REPO_ROOT / f"models_kfold_{EXPERIMENT_NAME}"
OUTPUT_REPORT = RESULTS_DIR / f"v3_phase3_{EXPERIMENT_NAME}_kfold.md"
OUTPUT_HEADLINE = RESULTS_DIR / f"v3_phase3_{EXPERIMENT_NAME}_headline.md"

LOG_PATH = REPO_ROOT / "tools" / "research" / "v3_phase3_interaction.log"
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
logger = logging.getLogger("v3_phase3_interaction")


def _build_features() -> None:
    """Load features.parquet, compute interaction col, write to output path.

    Idempotent: skip the rebuild if the output already exists AND contains
    the new column. Lets re-runs after a partial train_kfold reuse the
    cached features parquet.
    """
    if OUTPUT_FEATURES.exists():
        # Verify the cached file has the new column (defensive — caller
        # might have a stale parquet from an aborted earlier attempt).
        import pyarrow.parquet as pq
        schema = pq.read_schema(OUTPUT_FEATURES)
        if INTERACTION_COL in schema.names:
            logger.info("SKIP feature build — %s already exists with new col",
                        OUTPUT_FEATURES)
            return
        logger.warning("cached %s missing %r — rebuilding",
                       OUTPUT_FEATURES, INTERACTION_COL)

    import pandas as pd

    logger.info("loading %s", INPUT_FEATURES)
    t0 = time.monotonic()
    df = pd.read_parquet(INPUT_FEATURES)
    logger.info("loaded %d rows x %d cols in %.1fs",
                len(df), len(df.columns), time.monotonic() - t0)

    if "sa2_seifa_irsd_score" not in df.columns:
        raise RuntimeError("sa2_seifa_irsd_score missing from features.parquet")
    if "cal_day_of_fortnight" not in df.columns:
        raise RuntimeError("cal_day_of_fortnight missing from features.parquet")

    # Raw product. LightGBM is scale-invariant; mean-centering SEIFA would
    # add no information the tree can use. The null pattern: rows where
    # SEIFA is null (~1.4M / ~15M, ~9%) — the SA2 identical-rows guard in
    # train_models._train_one_fold already filters those out for Model B
    # via the SA2_COLUMNS notna() mask, so we don't need to fillna here.
    df[INTERACTION_COL] = (
        df["sa2_seifa_irsd_score"].astype("float64")
        * df["cal_day_of_fortnight"].astype("float64")
    )
    logger.info(
        "new column %s: dtype=%s min=%.1f max=%.1f null=%d (%.1f%%)",
        INTERACTION_COL,
        df[INTERACTION_COL].dtype,
        df[INTERACTION_COL].min(),
        df[INTERACTION_COL].max(),
        df[INTERACTION_COL].isna().sum(),
        100 * df[INTERACTION_COL].isna().sum() / len(df),
    )

    t0 = time.monotonic()
    df.to_parquet(OUTPUT_FEATURES, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %s in %.1fs", OUTPUT_FEATURES, time.monotonic() - t0)


def _run_kfold() -> None:
    """Monkey-patch SA2_COLUMNS to include the new col; run train_kfold + compare_kfold."""
    from fuel_pred.build import make_features
    from fuel_pred.evaluate.compare import compare_kfold
    from fuel_pred.train import feature_blocks
    from fuel_pred.train.cv import train_kfold

    # Snapshot originals so the monkey-patch can be reverted at the end
    # even if the run fails.
    orig_sa2 = feature_blocks.SA2_COLUMNS
    orig_block = feature_blocks.BLOCK_COLUMNS["sa2"]
    orig_mf = make_features.SA2_FEATURE_COLS

    new_sa2 = (*orig_sa2, INTERACTION_COL)
    feature_blocks.SA2_COLUMNS = new_sa2
    feature_blocks.BLOCK_COLUMNS["sa2"] = new_sa2
    make_features.SA2_FEATURE_COLS = new_sa2
    logger.info("SA2_COLUMNS monkey-patched: %d -> %d cols (added %r)",
                len(orig_sa2), len(new_sa2), INTERACTION_COL)

    try:
        audit_path = OUTPUT_MODELS / "kfold_audit.json"
        report_exists = OUTPUT_REPORT.exists()
        if audit_path.exists() and report_exists:
            logger.info("SKIP train_kfold — %s + %s already exist",
                        audit_path, OUTPUT_REPORT)
        else:
            logger.info("=" * 70)
            logger.info("training Model A + B across 6 folds (B includes %r)",
                        INTERACTION_COL)
            logger.info("=" * 70)
            t0 = time.monotonic()
            # Skip B' — venue block isn't the question here, and it makes
            # the run ~33% faster.
            train_kfold(
                OUTPUT_FEATURES,
                OUTPUT_MODELS,
                models_to_fit=("A", "B"),
                # Use the spec default seed (42) so this is directly
                # comparable to the published PR B baseline (which also
                # used seed 42 via config.LGBM_PARAMS).
                random_state=None,
            )
            logger.info("train_kfold complete in %.1f min",
                        (time.monotonic() - t0) / 60)

            t0 = time.monotonic()
            compare_kfold(OUTPUT_FEATURES, OUTPUT_MODELS, OUTPUT_REPORT)
            logger.info("compare_kfold complete in %.1f min",
                        (time.monotonic() - t0) / 60)
    finally:
        feature_blocks.SA2_COLUMNS = orig_sa2
        feature_blocks.BLOCK_COLUMNS["sa2"] = orig_block
        make_features.SA2_FEATURE_COLS = orig_mf
        logger.info("SA2_COLUMNS reverted to %d cols", len(orig_sa2))


def _write_headline() -> None:
    """Pull aggregate stats from the merged report + the new col's gain
    importance from feature_lists.json; write a short headline doc.
    """
    import re

    if not OUTPUT_REPORT.exists():
        logger.warning("no report at %s — skipping headline", OUTPUT_REPORT)
        return

    text = OUTPUT_REPORT.read_text(encoding="utf-8")

    # ---- Extract the per-fold Δ MAE values + aggregate row ----
    # Same regex pattern used in v3_phase3_rank_consistency.py — stable
    # across the merged-report format.
    fold_row = re.compile(
        r"^\|\s*fold_(\d+)\s*\|\s*[\d-]+\s*→\s*[\d-]+\s*\|"
        r"\s*[\d,]+\s*\|\s*(-?\d+\.\d+)\s*\|\s*(-?\d+\.\d+)\s*\|"
        r"\s*([+-]?\d+\.\d+)\s*\|"
    )
    per_fold = []
    for line in text.splitlines():
        if len(per_fold) >= 6:
            break
        m = fold_row.match(line)
        if m:
            per_fold.append({
                "fold": int(m.group(1)),
                "mae_a": float(m.group(2)),
                "mae_b": float(m.group(3)),
                "delta_mae": float(m.group(4)),
            })

    # Aggregate (Mean / Stdev) row
    agg = {}
    for label, key in (("**Mean**", "mean"), ("Stdev", "stdev"),
                       ("Min", "min"), ("Max", "max")):
        m = re.search(
            rf"\|\s*{re.escape(label)}\s*\|[^\|]+\|[^\|]+\|"
            rf"([^\|]+)\|([^\|]+)\|([^\|]+)\|",
            text,
        )
        if m:
            agg[f"{key}_mae_a"] = float(m.group(1).strip())
            agg[f"{key}_mae_b"] = float(m.group(2).strip())
            agg[f"{key}_delta_mae"] = float(m.group(3).strip())

    # ---- Extract the new col's gain importance from fold 1's audit ----
    interaction_gains: list[tuple[int, float, int]] = []  # (fold, gain, rank)
    for fold_n in range(1, 7):
        fl_path = OUTPUT_MODELS / f"fold_{fold_n}" / "feature_lists.json"
        if not fl_path.exists():
            continue
        fl = json.loads(fl_path.read_text(encoding="utf-8"))
        if "B" not in fl:
            continue
        gains = fl["B"]["importance_gain"]
        if INTERACTION_COL not in gains:
            continue
        # Compute rank (1 = highest gain)
        sorted_gains = sorted(gains.items(), key=lambda kv: kv[1], reverse=True)
        rank = next(
            (i + 1 for i, (name, _) in enumerate(sorted_gains)
             if name == INTERACTION_COL),
            None,
        )
        interaction_gains.append((fold_n, gains[INTERACTION_COL], rank or -1))

    # ---- Reference: published PR B baseline ----
    ref = {"mean": 0.215, "stdev": 0.394, "min": -0.135, "max": 1.042}

    # ---- Verdict ----
    mean_delta = agg.get("mean_delta_mae", float("nan"))
    stdev_delta = agg.get("stdev_delta_mae", float("nan"))
    if abs(mean_delta) > 2 * stdev_delta:
        verdict = "**robust** ✅" if mean_delta < 0 else "**robust ❌ (B loses)**"
    elif abs(mean_delta) > stdev_delta:
        verdict = "weak"
    else:
        verdict = "noise"

    # ---- Compose markdown ----
    lines = [
        "# v3.0 Phase 3 #3 — SEIFA × day-of-fortnight interaction",
        "",
        f"Single-experiment headline for `{EXPERIMENT_NAME}`. Tests the "
        f"Reading-C2 hypothesis from "
        f"`docs/research/2026-06_v3.0_phase2_postmortem_discussion.md`: "
        f"does an explicit `{INTERACTION_COL}` feature unlock the "
        f"Centrelink-fortnightly-cycle × SEIFA-disadvantage interaction "
        f"that v2.x's SHAP-interaction probe missed?",
        "",
        f"Full merged report: `{OUTPUT_REPORT.relative_to(REPO_ROOT)}`",
        "",
        "## Headline",
        "",
        f"- **Mean Δ MAE: {mean_delta:+.3f} c/L** (negative = Model B with "
        f"interaction beats Model A)",
        f"- Stdev across 6 folds: {stdev_delta:.3f} c/L",
        f"- Min: {agg.get('min_delta_mae', float('nan')):+.3f}",
        f"- Max: {agg.get('max_delta_mae', float('nan')):+.3f}",
        f"- **Verdict** (|Mean|>2×Stdev = robust): {verdict}",
        "",
        "## Compared to published PR B baseline (no interaction)",
        "",
        "| Metric | This run (with interaction) | PR B baseline |",
        "|--------|-----------------------------:|--------------:|",
        f"| Mean Δ MAE | {mean_delta:+.3f} | {ref['mean']:+.3f} |",
        f"| Stdev | {stdev_delta:.3f} | {ref['stdev']:.3f} |",
        f"| Min | {agg.get('min_delta_mae', float('nan')):+.3f} | {ref['min']:+.3f} |",
        f"| Max | {agg.get('max_delta_mae', float('nan')):+.3f} | {ref['max']:+.3f} |",
        "",
    ]

    if per_fold:
        lines += [
            "## Per-fold Δ MAE (with interaction)",
            "",
            "| Fold | MAE A | MAE B | Δ MAE |",
            "|------|------:|------:|------:|",
        ]
        for f in per_fold:
            lines.append(
                f"| fold_{f['fold']} | {f['mae_a']:.3f} | "
                f"{f['mae_b']:.3f} | {f['delta_mae']:+.3f} |"
            )
        lines.append("")

    if interaction_gains:
        lines += [
            f"## Where does `{INTERACTION_COL}` rank in Model B's gain importance?",
            "",
            "Per-fold LightGBM `gain` importance for the new column + its rank "
            "among all Model B features (rank 1 = highest gain). High rank "
            "(top 20) means the model is actively using the interaction; low "
            "rank (bottom 25) means LightGBM found nothing.",
            "",
            "| Fold | Gain | Rank (of all Model B features) |",
            "|------|-----:|-------------------------------:|",
        ]
        for fold_n, gain, rank in interaction_gains:
            lines.append(f"| fold_{fold_n} | {gain:,.0f} | {rank} |")
        lines.append("")

    lines += [
        "## Reading",
        "",
    ]
    if mean_delta < -2 * stdev_delta:
        lines.append(
            "**Reading C2 CONFIRMED.** Adding the explicit "
            f"`{INTERACTION_COL}` feature produces a robust improvement. "
            "LightGBM's depth-3 tree path wasn't finding this interaction "
            "implicitly; the explicit feature was needed. **Next step:** "
            "systematic interaction-feature exploration (other × Centrelink "
            "cycle, SEIFA × public holiday, etc.) under k-fold.")
    elif abs(mean_delta - ref['mean']) < 0.1 and abs(stdev_delta - ref['stdev']) < 0.1:
        lines.append(
            "**Reading C2 FALSIFIED.** The explicit interaction column "
            "produces essentially the same headline as the PR B baseline. "
            "The SHAP-interaction probe was right — LightGBM (or the lag-"
            "derived features) already captures whatever Centrelink-cycle "
            "× SEIFA structure exists in the data. **Next step:** lean "
            "Reading-A and ship Model A; or escalate to Reading C3 "
            "(different model class — FT-Transformer / SAINT).")
    else:
        lines.append(
            "**Inconclusive — partial effect.** The interaction moves the "
            "headline but not robustly. Check the per-fold table above for "
            "where it helps vs hurts; if fold_3 or fold_6 specifically "
            "benefits, that's a Reading-C-leaning signal worth chasing.")
    lines.append("")

    lines += [
        "## Sources",
        "",
        f"- `{OUTPUT_REPORT.relative_to(REPO_ROOT)}` — full per-fold report",
        "- `tools/research/v3_phase3_interaction_experiment.py` — this script",
        "- `docs/research/2026-06_v3.0_phase2_postmortem_discussion.md` — "
        "Reading C2 hypothesis + ranked next-steps",
        "- `results/v3_phase2_pr_b_baseline_kfold.md` — reference numbers",
        "",
    ]

    OUTPUT_HEADLINE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", OUTPUT_HEADLINE)

    # Console echo
    logger.info("=== Interaction-experiment headline ===")
    logger.info("Mean Δ MAE:  %+.3f c/L (PR B baseline: %+.3f)",
                mean_delta, ref['mean'])
    logger.info("Stdev Δ MAE: %.3f c/L (PR B baseline:  %.3f)",
                stdev_delta, ref['stdev'])
    logger.info("Verdict:     %s", verdict)
    if interaction_gains:
        avg_rank = sum(r for _, _, r in interaction_gains) / len(interaction_gains)
        logger.info("New col mean rank across folds: %.1f", avg_rank)


def main() -> None:
    logger.info("v3.0 Phase 3 interaction experiment starting")
    logger.info("interaction column: %s = sa2_seifa_irsd_score * cal_day_of_fortnight",
                INTERACTION_COL)

    _build_features()
    _run_kfold()
    _write_headline()

    logger.info("v3.0 Phase 3 interaction experiment complete")
    logger.info("Headline:        %s", OUTPUT_HEADLINE)
    logger.info("Full report:     %s", OUTPUT_REPORT)


if __name__ == "__main__":
    main()
