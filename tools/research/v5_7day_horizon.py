"""v5 — does the SA2 augmentor help at the 7-day horizon (y_t1_t7)?

The v3.0 conclusion (ship Model A; augmentor adds no robust lift) is
bounded to the **1-day** target (y_t1). The natural challenge: at a
longer horizon the lag features carry less information — tomorrow's
price is ~today's price, but the t+1..t+7 mean is harder to pin from
recent lags alone. If the augmentor's static demographic context is
ever going to win, it's where the lag advantage decays.

This experiment re-runs the v3.0 Phase 2 PR-B-baseline A-vs-B comparison
at the 7-day horizon:

- target = ``y_t1_t7`` (mean of price[t+1..t+7])
- KFoldConfig horizon_days=7 (the folds.py fix makes this widen the
  train/test gap to 8 days so the 7-day-ahead target can't leak)
- Model A (no SA2) vs Model B (15-col SA2 block), v3.0 tuned defaults
- 6-fold k-fold, per-fold + aggregate Δ MAE

It is the cheapest decisive test: if the augmentor's per-fold mean Δ MAE
goes robustly negative at t+7 (B beats A, |Mean| > Stdev), that's a
genuine sign-flip vs t+1 and a real finding — "augmentor helps at longer
horizons where lags weaken." If it stays in the noise band like t+1,
the v3.0 conclusion generalises across horizons and we stop.

Two reference points to compare against:
- t+1 PR B baseline (v3.0 Phase 2): Mean Δ MAE +0.215, Stdev 0.394 (noise)
- This run at t+7

Wall-clock ~15-20 min (A + B only, 6 folds, ~40% of rows have the
7-day target so folds are a bit smaller than t+1).

Run:
    uv run python tools/research/v5_7day_horizon.py 2>&1 | \\
        tee tools/research/v5_7day_horizon.tee.log
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

FEATURES = DATA_PROCESSED / "features.parquet"
OUT_MODELS = REPO_ROOT / "models_kfold_v5_7day"
OUT_REPORT = RESULTS_DIR / "v5_7day_horizon_kfold.md"
OUT_HEADLINE = RESULTS_DIR / "v5_7day_horizon_headline.md"

TARGET = "y_t1_t7"
HORIZON_DAYS = 7

# Reference: t+1 PR B baseline from v3.0 Phase 2 (results/v3_phase2_metrics.json).
T1_REF = {"mean_delta_mae": 0.215, "stdev_delta_mae": 0.394,
          "min_delta_mae": -0.135, "max_delta_mae": 1.042}

LOG_PATH = REPO_ROOT / "tools" / "research" / "v5_7day_horizon.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, mode="a")],
)
logger = logging.getLogger("v5_7day")


def _run() -> None:
    from fuel_pred.evaluate.compare import compare_kfold
    from fuel_pred.train.cv import train_kfold
    from fuel_pred.train.folds import KFoldConfig

    cfg = KFoldConfig(horizon_days=HORIZON_DAYS)
    logger.info(
        "v5 7-day horizon: target=%s horizon_days=%d (gap widens to %d days)",
        TARGET, HORIZON_DAYS, 1 + cfg.gap_days + (HORIZON_DAYS - 1),
    )

    audit = OUT_MODELS / "kfold_audit.json"
    if audit.exists() and OUT_REPORT.exists():
        logger.info("SKIP train+compare — audit + report already exist")
    else:
        t0 = time.monotonic()
        # A + B only (no B' — venue block isn't the question). v3.0 tuned
        # defaults come from config.LGBM_PARAMS automatically.
        train_kfold(
            FEATURES, OUT_MODELS,
            kfold_config=cfg,
            target=TARGET,
            models_to_fit=("A", "B"),
        )
        logger.info("train_kfold done in %.1f min", (time.monotonic() - t0) / 60)

        t0 = time.monotonic()
        compare_kfold(FEATURES, OUT_MODELS, OUT_REPORT)
        logger.info("compare_kfold done in %.1f min", (time.monotonic() - t0) / 60)

    _write_headline()


def _write_headline() -> None:
    import re

    if not OUT_REPORT.exists():
        logger.warning("no report at %s", OUT_REPORT)
        return
    text = OUT_REPORT.read_text(encoding="utf-8")

    fold_row = re.compile(
        r"^\|\s*fold_(\d+)\s*\|\s*[\d-]+\s*→\s*[\d-]+\s*\|"
        r"\s*[\d,]+\s*\|\s*(-?\d+\.\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*([+-]?\d+\.\d+)\s*\|"
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

    agg = {}
    for label, key in (("**Mean**", "mean"), ("Stdev", "stdev"),
                       ("Min", "min"), ("Max", "max")):
        m = re.search(
            rf"\|\s*{re.escape(label)}\s*\|[^\|]+\|[^\|]+\|"
            rf"([^\|]+)\|([^\|]+)\|([^\|]+)\|", text)
        if m:
            agg[f"{key}_delta_mae"] = float(m.group(3).strip())

    mean_d = agg.get("mean_delta_mae", float("nan"))
    stdev_d = agg.get("stdev_delta_mae", float("nan"))
    if abs(mean_d) > 2 * stdev_d:
        verdict = "**ROBUST WIN** (B beats A)" if mean_d < 0 else "**ROBUST (B loses)**"
    elif abs(mean_d) > stdev_d:
        verdict = "weak (B beats A)" if mean_d < 0 else "weak (B loses)"
    else:
        verdict = "noise"

    payload = {
        "target": TARGET, "horizon_days": HORIZON_DAYS,
        "per_fold": per_fold, "aggregate": agg, "verdict": verdict,
        "t1_reference": T1_REF,
    }
    (RESULTS_DIR / "v5_7day_horizon.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# v5 — SA2 augmentor at the 7-day horizon (y_t1_t7)",
        "",
        "Re-runs the v3.0 Phase 2 PR-B-baseline A-vs-B comparison at the "
        "7-day target. Tests whether the augmentor helps where the lag "
        "features carry less information.",
        "",
        f"Full report: `{OUT_REPORT.relative_to(REPO_ROOT)}`",
        "",
        "## Headline — A vs B at t+7",
        "",
        f"- **Mean Δ MAE: {mean_d:+.3f} c/L** (negative = Model B with SA2 beats Model A)",
        f"- Stdev across 6 folds: {stdev_d:.3f}",
        f"- **Verdict: {verdict}**",
        "",
        "## t+7 vs t+1 (the key comparison)",
        "",
        "| Metric | t+1 (v3.0 Phase 2) | t+7 (this run) |",
        "|--------|-------------------:|---------------:|",
        f"| Mean Δ MAE (B−A) | {T1_REF['mean_delta_mae']:+.3f} | {mean_d:+.3f} |",
        f"| Stdev | {T1_REF['stdev_delta_mae']:.3f} | {stdev_d:.3f} |",
        "",
    ]
    if per_fold:
        lines += ["## Per-fold Δ MAE at t+7", "",
                  "| Fold | MAE A | MAE B | Δ MAE |",
                  "|------|------:|------:|------:|"]
        for f in per_fold:
            lines.append(f"| fold_{f['fold']} | {f['mae_a']:.3f} | "
                         f"{f['mae_b']:.3f} | {f['delta_mae']:+.3f} |")
        lines.append("")

    lines += ["## Reading", ""]
    if mean_d < -stdev_d:
        lines.append("**Augmentor helps at t+7.** Sign-flipped vs the t+1 null — "
                     "Model B robustly beats Model A at the longer horizon where "
                     "lag features weaken. This is a genuine new finding: the v3.0 "
                     "ship-Model-A conclusion is horizon-specific. Next: run the "
                     "full Phase 2 (8 variants) + seed-noise floor at t+7.")
    elif abs(mean_d) <= stdev_d:
        lines.append("**Augmentor still null at t+7.** Same noise-band outcome as "
                     "t+1 — the v3.0 conclusion generalises across horizons. The "
                     "lag features weakening did NOT open a gap the augmentor fills; "
                     "whatever predicts the 7-day mean, it isn't SA2 demographics. "
                     "Strengthens the methodology story (null holds at 2 horizons).")
    else:
        lines.append("**Augmentor HURTS more at t+7** (Model B worse, |Mean| > Stdev). "
                     "The longer horizon amplifies the augmentor's overfitting rather "
                     "than helping. Firmly confirms ship-Model-A across horizons.")
    lines.append("")
    lines += ["## Sources", "",
              f"- `{OUT_REPORT.relative_to(REPO_ROOT)}` — full per-fold report",
              "- `tools/research/v5_7day_horizon.py` — this script",
              "- `results/v3_phase2_pr_b_baseline_kfold.md` — t+1 reference",
              ""]
    OUT_HEADLINE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", OUT_HEADLINE)

    logger.info("=== v5 7-day horizon headline ===")
    logger.info("t+7 Mean Delta MAE: %+.3f (t+1 ref: %+.3f)", mean_d, T1_REF["mean_delta_mae"])
    logger.info("t+7 Stdev:          %.3f (t+1 ref: %.3f)", stdev_d, T1_REF["stdev_delta_mae"])
    logger.info("Verdict:            %s", verdict)


def main() -> None:
    logger.info("v5 7-day horizon experiment starting")
    if not FEATURES.exists():
        raise RuntimeError(f"missing {FEATURES}")
    _run()
    logger.info("v5 7-day horizon experiment complete")


if __name__ == "__main__":
    main()
