"""v4 hypothesis test — does adding the fuel excise rate stabilise fold 3?

Phase 3 #2 found that **folds 3 (2022-23) and 6 (2025-26) have 3-5×
higher per-fold seed-stdev** than folds 1/2/4/5 (Mean MAE_A seed-stdev
0.163 + 0.178 vs ~0.03-0.07 elsewhere). The Phase 3 closing summary
documented the finding but didn't ask why.

**Hypothesis**: fold_3's instability is the September 28, 2022 fuel
excise restoration. The Australian government **halved the federal
fuel excise from 44.2 c/L to 22.1 c/L on March 30, 2022**, then
restored it (with CPI catch-up indexation, to ~46.0 c/L) on
September 29, 2022. Fold 3's test window is 2022-05 → 2023-04 — it
spans both the halved-excise period (May 1 - Sept 28) AND the
post-restoration period.

The model has no feature for this. The lag features track recent
prices, the Brent block tracks the crude side, but nothing tells the
model the *tax* changed by ~24 c/L overnight. Without that feature,
fold_3 training data ≤ 2022-04 never saw the change, and the model
extrapolates over a structural break — which is exactly the kind of
thing that makes per-fold MAE seed-sensitive.

**Test**: add ``cal_fuel_excise_cents_per_litre`` to the calendar
block, re-run the Phase 3 #2 seed-noise protocol (6 seeds × Model A
only across all 6 folds), and compare per-fold seed-stdev to the
original baseline.

Predicted outcomes:
- **fold_3 seed-stdev drops from 0.163 to ~0.07 or lower** → hypothesis
  confirmed; the missing structural feature was the explanation.
- **fold_6 also drops** → bonus; the 2026 spike maybe also has a policy
  component (unlikely from just the excise feature).
- **Neither drops** → falsified; the instability is something deeper
  than the policy break. Refines the next-experiment question.

Wall-clock estimate: ~45 min (6 seeds × ~7-8 min per Model A k-fold
run, similar to Phase 3 #2).
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

SEEDS = (42, 1, 7, 13, 99, 123)
INPUT_FEATURES = DATA_PROCESSED / "features.parquet"
OUTPUT_FEATURES = DATA_PROCESSED / "features_v4_excise.parquet"
# Baseline: the hyperopt validation per-seed per-fold MAE under the NEW
# tuned defaults but WITHOUT the excise feature. This is the only fair
# comparison — same hyperparameters on both sides, the only experimental
# difference is the new feature. Using v3_phase3_seed_noise.json (the OLD
# spec-defaults baseline) instead conflates two effects (the hyperparameter
# retune AND the new feature). The hyperopt validation file gives us per-
# seed per-fold MAE under the new defaults without excise — that's the
# clean baseline.
BASELINE_SEED_JSON = RESULTS_DIR / "v3_phase3_hyperopt_validation.json"
# Kept around for explicit-comparison reporting.
OLD_DEFAULTS_BASELINE_JSON = RESULTS_DIR / "v3_phase3_seed_noise.json"

LOG_PATH = REPO_ROOT / "tools" / "research" / "v4_excise_fold_instability.log"
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
logger = logging.getLogger("v4_excise")


# Australian federal fuel excise on U91/Diesel, cents per litre.
# Sources:
# - Pre-2022: indexed semi-annually (Feb 1, Aug 1) per CPI; rates from
#   ABS / ATO published tables.
# - 2022-03-30 to 2022-09-28: halved as a temporary cost-of-living
#   measure (Coalition government's March 2022 budget).
# - 2022-09-29 onwards: restored with CPI catch-up indexation; semi-
#   annual indexation resumes.
#
# Rates approximate to 1 decimal place — sufficient for the model to
# see the discontinuity at Mar 30 + Sept 29 2022. Where exact rates
# weren't found I interpolated linearly between known indexation dates.
EXCISE_SCHEDULE: tuple[tuple[str, float], ...] = (
    ("2016-08-01", 39.5),  # pre-panel anchor
    ("2017-02-01", 39.9),
    ("2017-08-01", 40.1),
    ("2018-02-01", 40.5),
    ("2018-08-01", 41.0),
    ("2019-02-01", 41.3),
    ("2019-08-01", 41.6),
    ("2020-02-01", 42.3),
    ("2020-08-01", 42.7),
    ("2021-02-01", 42.7),
    ("2021-08-01", 43.2),
    ("2022-02-01", 44.2),
    ("2022-03-30", 22.1),  # *** HALVED — temporary 6-month cut ***
    ("2022-09-29", 46.0),  # *** RESTORED with CPI catch-up ***
    ("2023-02-01", 47.7),
    ("2023-08-01", 48.8),
    ("2024-02-01", 49.6),
    ("2024-08-01", 50.6),
    ("2025-02-01", 50.8),
    ("2025-08-01", 51.6),
    ("2026-02-01", 51.6),  # estimated — would need Feb 2026 ABS pub to verify
)


def _build_excise_series(dates: pd.Series) -> pd.Series:
    """Map each panel date to the federal fuel excise rate in c/L.

    Step function: for each date, the rate is the most recent schedule
    entry on or before that date. Implemented via ``np.searchsorted``
    on int64-normalized datetimes — avoids the ``pd.merge_asof`` dtype-
    mismatch trap when the input has datetime64[s] and the hand-built
    schedule defaults to datetime64[us].
    """
    # Sort schedule by date + extract parallel rate array
    sorted_schedule = sorted(EXCISE_SCHEDULE, key=lambda x: x[0])
    schedule_dates = np.array(
        [pd.Timestamp(d).to_datetime64() for d, _ in sorted_schedule],
        dtype="datetime64[s]",
    )
    schedule_rates = np.array([c for _, c in sorted_schedule], dtype="float64")

    # Normalize input dates to the same dtype so np.searchsorted compares
    # like-for-like.
    panel_dates = pd.to_datetime(dates).to_numpy().astype("datetime64[s]")

    # For each date, find the most recent schedule entry ≤ date.
    # searchsorted(..., side='right') gives the insertion point AFTER any
    # equal entries; subtract 1 to get the "≤" semantics.
    idx = np.searchsorted(schedule_dates, panel_dates, side="right") - 1
    idx = np.clip(idx, 0, len(schedule_rates) - 1)

    return pd.Series(schedule_rates[idx], index=pd.RangeIndex(len(panel_dates)))


def _build_features() -> None:
    """Load features.parquet, compute excise rate per date, save new parquet.

    Idempotent: skip the rebuild if the output already exists AND contains
    the new column.
    """
    if OUTPUT_FEATURES.exists():
        import pyarrow.parquet as pq
        schema = pq.read_schema(OUTPUT_FEATURES)
        if "cal_fuel_excise_cents_per_litre" in schema.names:
            logger.info("SKIP feature build — %s already exists with new col",
                        OUTPUT_FEATURES)
            return
        logger.warning("cached %s missing new col — rebuilding", OUTPUT_FEATURES)

    logger.info("loading %s", INPUT_FEATURES)
    t0 = time.monotonic()
    df = pd.read_parquet(INPUT_FEATURES)
    logger.info("loaded %d rows x %d cols in %.1fs",
                len(df), len(df.columns), time.monotonic() - t0)

    excise = _build_excise_series(df["date"])
    df["cal_fuel_excise_cents_per_litre"] = excise.astype("float64")

    # Diagnostic row counters — compare on a Timestamp series to dodge the
    # ``datetime.date >= str`` TypeError when the source column is object-
    # dtype (it is: pandas stores datetime.date as object). Doing this
    # vectorized once is cheap on a 15M-row frame.
    date_ts = pd.to_datetime(df["date"])
    cut_start = pd.Timestamp("2022-03-30")
    cut_end = pd.Timestamp("2022-09-29")
    n_holiday = int(((date_ts >= cut_start) & (date_ts < cut_end)).sum())
    n_full = int((date_ts >= cut_end).sum())
    n_pre = int((date_ts < cut_start).sum())
    logger.info(
        "new column cal_fuel_excise_cents_per_litre: dtype=%s "
        "min=%.2f max=%.2f mean=%.2f null=%d (%.1f%%)",
        df["cal_fuel_excise_cents_per_litre"].dtype,
        df["cal_fuel_excise_cents_per_litre"].min(),
        df["cal_fuel_excise_cents_per_litre"].max(),
        df["cal_fuel_excise_cents_per_litre"].mean(),
        df["cal_fuel_excise_cents_per_litre"].isna().sum(),
        100 * df["cal_fuel_excise_cents_per_litre"].isna().sum() / len(df),
    )
    logger.info(
        "row breakdown - pre-cut: %d (%.1f%%), holiday (22.1 c/L): %d (%.1f%%), "
        "post-restoration (>=46 c/L): %d (%.1f%%)",
        n_pre, 100 * n_pre / len(df),
        n_holiday, 100 * n_holiday / len(df),
        n_full, 100 * n_full / len(df),
    )

    t0 = time.monotonic()
    df.to_parquet(OUTPUT_FEATURES, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %s in %.1fs", OUTPUT_FEATURES, time.monotonic() - t0)


def _import_train_kfold() -> Callable[..., object]:
    """Late import so logging is set up first."""
    from fuel_pred.train.cv import train_kfold
    return train_kfold


def _patch_calendar_block() -> None:
    """Monkey-patch ``feature_blocks.CALENDAR_COLUMNS`` to include the new col.

    Same pattern as Phase 2 / Phase 3 monkey-patching of SA2_COLUMNS.
    """
    from fuel_pred.train import feature_blocks
    orig = feature_blocks.CALENDAR_COLUMNS
    if "cal_fuel_excise_cents_per_litre" in orig:
        return  # already patched
    new = (*orig, "cal_fuel_excise_cents_per_litre")
    feature_blocks.CALENDAR_COLUMNS = new
    feature_blocks.BLOCK_COLUMNS["cal"] = new
    logger.info("CALENDAR_COLUMNS monkey-patched: %d -> %d cols (added cal_fuel_excise)",
                len(orig), len(new))


def _run_seed(seed: int) -> dict[str, float]:
    """Train Model A only on all 6 folds at this seed. Resume-safe."""
    out_root = REPO_ROOT / f"models_kfold_v4_excise_seed_{seed}"
    audit_path = out_root / "kfold_audit.json"

    if audit_path.exists():
        logger.info("[seed=%d] SKIP — audit already exists at %s", seed, audit_path)
        per_fold = _per_fold_mae_a_from_audit(audit_path)
        per_fold["wall_clock_min"] = 0.0
        per_fold["resumed_from_cache"] = True
        return per_fold

    if not OUTPUT_FEATURES.exists():
        raise RuntimeError(f"features parquet missing: {OUTPUT_FEATURES}")

    logger.info("=" * 70)
    logger.info("[seed=%d] training Model A only across 6 folds (with excise feature)",
                seed)
    logger.info("=" * 70)

    train_kfold = _import_train_kfold()
    t0 = time.monotonic()
    train_kfold(
        OUTPUT_FEATURES,
        out_root,
        random_state=seed,
        models_to_fit=("A",),
        save_predictions=False,
    )
    wall_min = (time.monotonic() - t0) / 60
    logger.info("[seed=%d] train_kfold complete in %.1f min", seed, wall_min)

    per_fold = _per_fold_mae_a_from_audit(audit_path)
    per_fold["wall_clock_min"] = round(wall_min, 1)
    per_fold["resumed_from_cache"] = False
    return per_fold


def _per_fold_mae_a_from_audit(audit_path: Path) -> dict[str, float]:
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
    """Hyperopt-validation baseline — per-seed per-fold MAE_A under the NEW
    tuned defaults WITHOUT the excise feature. Same hyperparameters as this
    experiment minus the new feature.
    """
    if not BASELINE_SEED_JSON.exists():
        raise RuntimeError(
            f"{BASELINE_SEED_JSON} missing — run "
            f"tools/research/v3_phase3_hyperopt_validation.py first"
        )
    data = json.loads(BASELINE_SEED_JSON.read_text(encoding="utf-8"))
    # The hyperopt validation JSON uses key 'per_seed_per_fold_mae_a_NEW'
    # to reflect "new tuned defaults without excise".
    raw = data["per_seed_per_fold_mae_a_NEW"]
    return {int(k): {fk: float(v) for fk, v in d.items() if fk.startswith("fold_")}
            for k, d in raw.items()}


def _load_old_defaults_baseline() -> dict[int, dict[str, float]]:
    """Phase 3 #2 OLD spec-defaults baseline — kept for explicit comparison.

    Used in the report to show the *total* improvement (retune + feature)
    against the original baseline, separately from the clean
    feature-only comparison.
    """
    if not OLD_DEFAULTS_BASELINE_JSON.exists():
        return {}
    data = json.loads(OLD_DEFAULTS_BASELINE_JSON.read_text(encoding="utf-8"))
    raw = data["per_seed_per_fold_mae_a"]
    return {int(k): {fk: float(v) for fk, v in d.items() if fk.startswith("fold_")}
            for k, d in raw.items()}


def _write_summary(
    new_per_seed: dict[int, dict[str, float]],
    baseline_per_seed: dict[int, dict[str, float]],
    old_defaults_per_seed: dict[int, dict[str, float]] | None = None,
) -> None:
    """Compute per-fold seed-stdev under new feature, compare to baseline."""
    summary_md = RESULTS_DIR / "v4_excise_fold_instability_summary.md"
    summary_json = RESULTS_DIR / "v4_excise_fold_instability.json"

    fold_keys = [f"fold_{i+1}" for i in range(6)]

    def _per_fold_seed_stats(per_seed: dict[int, dict[str, float]]) -> dict:
        stats = {}
        for fk in fold_keys:
            vals = [d[fk] for d in per_seed.values() if fk in d]
            if len(vals) >= 2:
                stats[fk] = {
                    "mean": statistics.fmean(vals),
                    "stdev": statistics.pstdev(vals),
                    "range": max(vals) - min(vals),
                }
        return stats

    new_stats = _per_fold_seed_stats(new_per_seed)
    base_stats = _per_fold_seed_stats(baseline_per_seed)
    old_stats = (
        _per_fold_seed_stats(old_defaults_per_seed)
        if old_defaults_per_seed else {}
    )

    # Per-fold stdev change: new - baseline. Negative = improved (lower
    # stdev = more stable).
    stdev_deltas: dict[str, float] = {}
    for fk in fold_keys:
        if fk in new_stats and fk in base_stats:
            stdev_deltas[fk] = new_stats[fk]["stdev"] - base_stats[fk]["stdev"]

    # Per-fold mean MAE change: new - baseline. Negative = better
    # (lower MAE).
    mae_deltas: dict[str, float] = {}
    for fk in fold_keys:
        if fk in new_stats and fk in base_stats:
            mae_deltas[fk] = new_stats[fk]["mean"] - base_stats[fk]["mean"]

    # Verdict per fold
    def _verdict_stdev(d_stdev: float, baseline_stdev: float) -> str:
        # "stabilised" if stdev dropped by more than half
        if d_stdev < -0.5 * baseline_stdev:
            return "STABILISED (>50% drop)"
        if d_stdev < -0.2 * baseline_stdev:
            return "improved"
        if d_stdev > 0.2 * baseline_stdev:
            return "worsened"
        return "unchanged"

    fold_verdicts = {
        fk: _verdict_stdev(stdev_deltas[fk], base_stats[fk]["stdev"])
        for fk in fold_keys if fk in stdev_deltas and fk in base_stats
    }

    # Hypothesis verdict — based on fold_3 specifically
    if fold_verdicts.get("fold_3") == "STABILISED (>50% drop)":
        hypothesis_verdict = "CONFIRMED"
    elif fold_verdicts.get("fold_3") == "improved":
        hypothesis_verdict = "PARTIAL"
    elif fold_verdicts.get("fold_3") == "worsened":
        hypothesis_verdict = "REJECTED"
    else:
        hypothesis_verdict = "FALSIFIED"

    payload = {
        "seeds": list(new_per_seed.keys()),
        "new_feature": "cal_fuel_excise_cents_per_litre",
        "per_seed_per_fold_mae_a_NEW_with_excise": new_per_seed,
        "per_fold_seed_stats_NEW_with_excise": new_stats,
        "per_fold_seed_stats_BASELINE_hyperopt_no_excise": base_stats,
        "per_fold_seed_stats_OLD_DEFAULTS_no_excise": old_stats,
        "stdev_deltas_FEATURE_ONLY": stdev_deltas,  # NEW - HYPEROPT_BASELINE
        "mae_deltas_FEATURE_ONLY": mae_deltas,
        "fold_verdicts": fold_verdicts,
        "hypothesis_verdict": hypothesis_verdict,
        "comparison_note": (
            "stdev_deltas_FEATURE_ONLY uses the hyperopt-validation baseline "
            "(new tuned defaults, no excise feature) — this isolates the "
            "feature's effect from the retune's effect. The OLD_DEFAULTS "
            "stats are kept for the cumulative comparison (retune + feature)."
        ),
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_json)

    lines: list[str] = []
    lines.append("# v4 hypothesis test — fuel excise feature → fold-3 stability")
    lines.append("")
    lines.append(
        "Adds ``cal_fuel_excise_cents_per_litre`` to the calendar block and "
        "re-runs the Phase 3 #2 seed-noise protocol. Tests whether fold_3's "
        "seed-instability (Phase 3 #2: 0.163 c/L stdev, 3-5× higher than "
        "the other folds) is the model failing to extrapolate over the "
        "Sept 28 2022 fuel excise restoration."
    )
    lines.append("")
    lines.append(
        "**Methodology note (corrected 2026-06-10):** the clean comparison "
        "is **against the hyperopt-validation baseline** (new tuned defaults "
        "WITHOUT the excise feature) — same hyperparameters on both sides, "
        "feature is the only experimental difference. The initial version "
        "of this script compared to the Phase 3 #2 OLD-defaults baseline, "
        "which conflated the retune's effect with the feature's effect. "
        "The OLD-defaults numbers are kept below as the **cumulative** "
        "(retune + feature) view for context."
    )
    lines.append("")

    lines.append("## Clean comparison — excise feature ONLY (new tuned defaults both sides)")
    lines.append("")
    lines.append("Per-fold seed-stdev under the new tuned defaults, with and without the excise feature:")
    lines.append("")
    lines.append("| Fold | WITHOUT excise | WITH excise | Δ stdev | Δ % | Verdict |")
    lines.append("|------|---------------:|------------:|--------:|----:|---------|")
    for fk in fold_keys:
        if fk not in new_stats or fk not in base_stats:
            continue
        b = base_stats[fk]["stdev"]
        n = new_stats[fk]["stdev"]
        d = stdev_deltas[fk]
        pct = 100 * d / b if b else 0
        v = fold_verdicts[fk]
        lines.append(f"| {fk} | {b:.4f} | {n:.4f} | {d:+.4f} | {pct:+5.0f}% | {v} |")
    lines.append("")

    if old_stats:
        lines.append("## Cumulative comparison — retune + excise feature (vs Phase 3 #2 OLD defaults)")
        lines.append("")
        lines.append("Shows the *total* improvement when both the hyperparameter retune "
                     "AND the excise feature are added, vs the original v1/v2 spec defaults. "
                     "This is the picture against the **original** Phase 3 #2 baseline.")
        lines.append("")
        lines.append("| Fold | OLD defaults stdev | NEW (retune + excise) stdev | Δ stdev |")
        lines.append("|------|-------------------:|----------------------------:|--------:|")
        for fk in fold_keys:
            if fk not in new_stats or fk not in old_stats:
                continue
            o = old_stats[fk]["stdev"]
            n = new_stats[fk]["stdev"]
            lines.append(f"| {fk} | {o:.4f} | {n:.4f} | {n-o:+.4f} |")
        lines.append("")

    lines.append("## Per-fold seed-mean MAE — new vs baseline")
    lines.append("")
    lines.append("Negative Δ = new feature reduced MAE. The headline test is "
                 "the **stdev** column above; the mean is a secondary check that "
                 "the feature isn't degrading overall MAE.")
    lines.append("")
    lines.append("| Fold | Baseline mean | New mean | Δ mean |")
    lines.append("|------|--------------:|---------:|-------:|")
    for fk in fold_keys:
        if fk not in new_stats or fk not in base_stats:
            continue
        b = base_stats[fk]["mean"]
        n = new_stats[fk]["mean"]
        d = mae_deltas[fk]
        lines.append(f"| {fk} | {b:.4f} | {n:.4f} | {d:+.4f} |")
    lines.append("")

    lines.append("## Per-seed per-fold MAE_A (new feature)")
    lines.append("")
    lines.append("| Seed | " + " | ".join(fold_keys) + " | Wall-clock |")
    lines.append("|------|" + "|".join(["---" for _ in fold_keys]) + "|----|")
    for seed in new_per_seed:
        d = new_per_seed[seed]
        row = [f"{d.get(fk, float('nan')):.4f}" for fk in fold_keys]
        wall = d.get("wall_clock_min", float("nan"))
        cache = " (cached)" if d.get("resumed_from_cache") else ""
        lines.append(f"| {seed} | " + " | ".join(row) + f" | {wall:.1f} min{cache} |")
    lines.append("")

    lines.append("## Hypothesis verdict")
    lines.append("")
    if hypothesis_verdict == "CONFIRMED":
        lines.append(
            "**Hypothesis CONFIRMED.** Fold_3's seed-stdev dropped by >50% "
            "with the excise feature added. The instability was the model "
            "failing to extrapolate over the September 28, 2022 fuel excise "
            "restoration. **Action:** roll the excise feature into the "
            "production calendar block (`feature_blocks.CALENDAR_COLUMNS`); "
            "update spec §7.3."
        )
    elif hypothesis_verdict == "PARTIAL":
        lines.append(
            "**Hypothesis PARTIAL.** Fold_3's seed-stdev improved but not "
            "dramatically. The excise feature helps but isn't the full "
            "explanation. **Action:** add the feature to production but "
            "continue investigating the residual instability."
        )
    elif hypothesis_verdict == "REJECTED":
        lines.append(
            "**Hypothesis REJECTED.** Fold_3's seed-stdev got WORSE with the "
            "excise feature added. The feature is either wrong (incorrect "
            "rates? wrong dates?) or it's introducing more variance than it "
            "removes. **Action:** verify excise dates against ATO official "
            "schedule; if rates are correct, the instability is not the "
            "policy break."
        )
    else:
        lines.append(
            "**Hypothesis FALSIFIED.** Fold_3's seed-stdev is essentially "
            "unchanged. The fuel excise feature doesn't explain the "
            "instability. **Action:** investigate alternative hypotheses — "
            "maybe fold_3's variance is the 2022-23 LNG/oil shock, or "
            "training-data composition (fold_3 trains on 2017-2022-04 — "
            "the last 4 months are pre-cut, then test on cut+restore)."
        )
    lines.append("")

    # Bonus check on fold_6
    if "fold_6" in fold_verdicts:
        v6 = fold_verdicts["fold_6"]
        lines.append(f"**Bonus — fold_6 verdict:** {v6}. "
                     "The 2026 spike-period instability has a different cause "
                     "than the 2022 excise cut, but if it also shows movement, "
                     "the excise feature might be capturing some broader "
                     "regime-change signal worth investigating further.")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append("- `tools/research/v4_excise_fold_instability.py` — this script")
    lines.append("- `results/v3_phase3_hyperopt_validation.json` — CLEAN baseline "
                 "(new tuned defaults, no excise) — the fair feature-only comparison")
    lines.append("- `results/v3_phase3_seed_noise.json` — OLD-defaults baseline "
                 "(cumulative retune+feature comparison)")
    lines.append("- `docs/research/2026-06_v3.0_phase3_closing_summary.md` — Phase 3 #2 + #4")
    lines.append("- `docs/research/2026-06_v4_fold_instability_excise_outcome.md` — narrative")
    lines.append("")
    lines.append("**Excise schedule sources:** Australian Government Federal "
                 "Treasury, March 2022 budget papers; ATO indexation tables.")
    lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", summary_md)

    logger.info("=== v4 excise hypothesis headline ===")
    logger.info("fold_3 baseline stdev: %.4f c/L", base_stats["fold_3"]["stdev"])
    logger.info("fold_3 new stdev:      %.4f c/L", new_stats["fold_3"]["stdev"])
    logger.info("Delta:                 %+.4f c/L", stdev_deltas["fold_3"])
    logger.info("fold_3 verdict:        %s", fold_verdicts["fold_3"])
    logger.info("Hypothesis:            %s", hypothesis_verdict)


def main() -> None:
    logger.info("v4 fold instability experiment starting")

    _build_features()
    _patch_calendar_block()
    baseline_per_seed = _load_baseline_per_seed()
    logger.info(
        "loaded CLEAN baseline (hyperopt val: new tuned defaults WITHOUT excise) "
        "for %d seeds", len(baseline_per_seed),
    )
    old_defaults_per_seed = _load_old_defaults_baseline()
    if old_defaults_per_seed:
        logger.info(
            "loaded OLD-defaults baseline (Phase 3 #2: original v1/v2 spec defaults) "
            "for %d seeds — kept for cumulative comparison",
            len(old_defaults_per_seed),
        )

    new_per_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        try:
            new_per_seed[seed] = _run_seed(seed)
        except Exception as exc:
            logger.exception("seed=%d FAILED: %s", seed, exc)
            new_per_seed[seed] = {"error": f"{type(exc).__name__}: {exc}"}  # type: ignore[dict-item]
        valid = {s: d for s, d in new_per_seed.items() if "error" not in d}
        if len(valid) >= 2:
            _write_summary(valid, baseline_per_seed, old_defaults_per_seed)

    logger.info("v4 fold instability experiment complete")


if __name__ == "__main__":
    main()
