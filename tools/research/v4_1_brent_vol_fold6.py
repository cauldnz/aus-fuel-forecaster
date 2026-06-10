"""v4.1 hypothesis test — do Brent realized-volatility features stabilise fold_6?

v4 (the excise experiment) established:
- The Phase 3 #4 hyperparameter retune already fixed fold_3's instability.
- fold_6 (2025-26, the oil-shock / 2026 price-spike period) remains the
  MOST unstable fold even after the retune (seed-stdev 0.100 vs ~0.02-0.045
  elsewhere). It is NOT a domestic-policy artifact — the excise feature
  made it worse.

**Hypothesis**: fold_6's residual instability is a *crude-volatility regime*
the model can't see. The upstream block has directional change features
(``upstream_brent_change_7d/14d``) but NO volatility features. Realized
volatility — the dispersion of daily returns, direction-agnostic — tells
the model "the crude market is in a turbulent regime" independent of which
way prices moved. During the 2026 spike the daily Brent swings are large
in both directions; a volatility feature would flag that regime where a
directional-change feature averages out.

Unlike the excise feature (which was near-constant within fold_6's test
window — that's why it failed there), realized volatility is *genuinely
elevated and variable* in the 2026 oil-shock window. So it avoids the
v4 "near-constant-within-fold" trap by construction.

**Test**: add three Brent realized-volatility features to the upstream
block —
  - ``upstream_brent_realized_vol_14d``  (stdev of trailing-14d log returns)
  - ``upstream_brent_realized_vol_30d``  (stdev of trailing-30d log returns)
  - ``upstream_brent_vol_ratio_14_90``   (14d vol / 90d vol — regime proxy:
        >1 means recent turbulence above the medium-term baseline)
— then re-run the 6-seed seed-noise protocol and compare per-fold
seed-stdev to the **clean** baseline (new tuned defaults, no extra
feature, from the hyperopt validation run).

All windows are TRAILING (use only data ≤ the row's date), so no
lookahead leakage for the y_t1 target — same convention as the existing
``upstream_brent_change_*`` features.

Predicted outcomes:
- **fold_6 seed-stdev drops materially** (say 0.100 -> ~0.06 or lower) →
  hypothesis confirmed; crude-volatility regime was the missing signal.
- **fold_6 unchanged or worse** → falsified; fold_6's instability is
  something other than crude volatility (candidate next hypotheses:
  the 2026 structural-break magnitude itself, or train-set composition).

Wall-clock ~50 min (6 seeds x Model A only across 6 folds).
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
OUTPUT_FEATURES = DATA_PROCESSED / "features_v4_1_brentvol.parquet"

# CLEAN baseline: new tuned defaults WITHOUT the extra features. Same
# rationale as the corrected v4 comparison — hold hyperparameters fixed,
# vary only the feature set.
BASELINE_SEED_JSON = RESULTS_DIR / "v3_phase3_hyperopt_validation.json"

NEW_COLS = (
    "upstream_brent_realized_vol_14d",
    "upstream_brent_realized_vol_30d",
    "upstream_brent_vol_ratio_14_90",
)

LOG_PATH = REPO_ROOT / "tools" / "research" / "v4_1_brent_vol_fold6.log"
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
logger = logging.getLogger("v4_1_brentvol")


def _build_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 3 Brent realized-vol columns on the per-date series, then
    broadcast back onto every station-day row by date.

    All windows are trailing (min_periods set so early dates produce NaN,
    which LightGBM handles natively). No lookahead: realized vol at date t
    uses Brent levels up to and including t — available when predicting t+1.
    """
    # Per-date Brent level series
    per_date = (
        df.dropna(subset=["upstream_brent_lag_0"])
        .groupby("date")["upstream_brent_lag_0"]
        .first()
        .sort_index()
    )
    brent = per_date.to_numpy(dtype="float64")

    # Daily log returns
    log_ret = np.full(len(brent), np.nan)
    log_ret[1:] = np.log(brent[1:] / brent[:-1])
    ret_s = pd.Series(log_ret, index=per_date.index)

    # Trailing rolling stdev of returns. min_periods = window // 2 so we
    # don't demand a full window before emitting (keeps early-panel coverage
    # reasonable) but still require enough points for a meaningful stdev.
    vol_14 = ret_s.rolling(14, min_periods=7).std()
    vol_30 = ret_s.rolling(30, min_periods=15).std()
    vol_90 = ret_s.rolling(90, min_periods=45).std()
    # Regime ratio: recent (14d) vs medium-term (90d) volatility.
    # >1 => recent turbulence above baseline. Guard div-by-zero.
    vol_ratio = vol_14 / vol_90.replace(0.0, np.nan)

    vol_df = pd.DataFrame({
        "date": per_date.index,
        "upstream_brent_realized_vol_14d": vol_14.to_numpy(),
        "upstream_brent_realized_vol_30d": vol_30.to_numpy(),
        "upstream_brent_vol_ratio_14_90": vol_ratio.to_numpy(),
    })

    # Broadcast back onto all rows by date (left join preserves row order).
    out = df.merge(vol_df, on="date", how="left")
    return out


def _build_features() -> None:
    """Load features.parquet, add the 3 vol columns, write the new parquet.

    Idempotent: skip if output already has all 3 new columns.
    """
    if OUTPUT_FEATURES.exists():
        import pyarrow.parquet as pq
        schema = pq.read_schema(OUTPUT_FEATURES)
        if all(c in schema.names for c in NEW_COLS):
            logger.info("SKIP feature build — %s already has all vol cols",
                        OUTPUT_FEATURES)
            return
        logger.warning("cached %s missing some vol cols — rebuilding", OUTPUT_FEATURES)

    logger.info("loading %s", INPUT_FEATURES)
    t0 = time.monotonic()
    df = pd.read_parquet(INPUT_FEATURES)
    logger.info("loaded %d rows x %d cols in %.1fs",
                len(df), len(df.columns), time.monotonic() - t0)

    df = _build_vol_features(df)

    for col in NEW_COLS:
        s = df[col]
        logger.info(
            "new column %s: min=%.5f max=%.5f mean=%.5f null=%d (%.1f%%)",
            col, s.min(), s.max(), s.mean(), s.isna().sum(),
            100 * s.isna().sum() / len(df),
        )

    # Diagnostic — confirm the vol features are NOT near-constant in fold_6's
    # test window (2025-05-01 -> 2026-04-30). That was the v4 failure mode.
    date_ts = pd.to_datetime(df["date"])
    f6_mask = (date_ts >= "2025-05-01") & (date_ts <= "2026-04-30")
    for col in NEW_COLS:
        f6 = df.loc[f6_mask, col]
        full = df[col]
        logger.info(
            "fold_6-window %s: mean=%.5f stdev=%.5f (full-panel stdev=%.5f) "
            "-> fold_6 within-window variation %s",
            col, f6.mean(), f6.std(), full.std(),
            "OK (variable)" if f6.std() > 0.2 * full.std() else "WARN near-constant",
        )

    t0 = time.monotonic()
    df.to_parquet(OUTPUT_FEATURES, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %s in %.1fs", OUTPUT_FEATURES, time.monotonic() - t0)


def _patch_upstream_block() -> None:
    """Monkey-patch ``feature_blocks.UPSTREAM_COLUMNS`` to add the vol cols."""
    from fuel_pred.train import feature_blocks
    orig = feature_blocks.UPSTREAM_COLUMNS
    if all(c in orig for c in NEW_COLS):
        return
    new = (*orig, *NEW_COLS)
    feature_blocks.UPSTREAM_COLUMNS = new
    feature_blocks.BLOCK_COLUMNS["upstream"] = new
    logger.info("UPSTREAM_COLUMNS monkey-patched: %d -> %d cols (added %s)",
                len(orig), len(new), list(NEW_COLS))


def _import_train_kfold() -> Callable[..., object]:
    from fuel_pred.train.cv import train_kfold
    return train_kfold


def _run_seed(seed: int) -> dict[str, float]:
    out_root = REPO_ROOT / f"models_kfold_v4_1_brentvol_seed_{seed}"
    audit_path = out_root / "kfold_audit.json"

    if audit_path.exists():
        logger.info("[seed=%d] SKIP — audit already exists", seed)
        per_fold = _per_fold_mae_a_from_audit(audit_path)
        per_fold["wall_clock_min"] = 0.0
        per_fold["resumed_from_cache"] = True
        return per_fold

    if not OUTPUT_FEATURES.exists():
        raise RuntimeError(f"features parquet missing: {OUTPUT_FEATURES}")

    logger.info("=" * 70)
    logger.info("[seed=%d] training Model A across 6 folds (with brent-vol features)", seed)
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
    """Clean baseline: new tuned defaults WITHOUT extra features (hyperopt val)."""
    if not BASELINE_SEED_JSON.exists():
        raise RuntimeError(f"{BASELINE_SEED_JSON} missing")
    data = json.loads(BASELINE_SEED_JSON.read_text(encoding="utf-8"))
    raw = data["per_seed_per_fold_mae_a_NEW"]
    return {int(k): {fk: float(v) for fk, v in d.items() if fk.startswith("fold_")}
            for k, d in raw.items()}


def _write_summary(
    new_per_seed: dict[int, dict[str, float]],
    baseline_per_seed: dict[int, dict[str, float]],
) -> None:
    summary_md = RESULTS_DIR / "v4_1_brent_vol_fold6_summary.md"
    summary_json = RESULTS_DIR / "v4_1_brent_vol_fold6.json"
    fold_keys = [f"fold_{i+1}" for i in range(6)]

    def _stats(per_seed: dict[int, dict[str, float]]) -> dict:
        out = {}
        for fk in fold_keys:
            vals = [d[fk] for d in per_seed.values() if fk in d]
            if len(vals) >= 2:
                out[fk] = {
                    "mean": statistics.fmean(vals),
                    "stdev": statistics.pstdev(vals),
                }
        return out

    new_stats = _stats(new_per_seed)
    base_stats = _stats(baseline_per_seed)

    stdev_deltas = {
        fk: new_stats[fk]["stdev"] - base_stats[fk]["stdev"]
        for fk in fold_keys if fk in new_stats and fk in base_stats
    }
    mae_deltas = {
        fk: new_stats[fk]["mean"] - base_stats[fk]["mean"]
        for fk in fold_keys if fk in new_stats and fk in base_stats
    }

    def _verdict(d: float, base: float) -> str:
        if d < -0.5 * base:
            return "STABILISED (>50% drop)"
        if d < -0.2 * base:
            return "improved"
        if d > 0.2 * base:
            return "worsened"
        return "unchanged"

    fold_verdicts = {
        fk: _verdict(stdev_deltas[fk], base_stats[fk]["stdev"])
        for fk in fold_keys if fk in stdev_deltas
    }

    f6 = fold_verdicts.get("fold_6", "n/a")
    if f6 == "STABILISED (>50% drop)":
        hypothesis = "CONFIRMED"
    elif f6 == "improved":
        hypothesis = "PARTIAL"
    elif f6 == "worsened":
        hypothesis = "REJECTED"
    else:
        hypothesis = "FALSIFIED"

    payload = {
        "seeds": list(new_per_seed.keys()),
        "new_features": list(NEW_COLS),
        "per_seed_per_fold_mae_a_NEW": new_per_seed,
        "per_fold_seed_stats_NEW": new_stats,
        "per_fold_seed_stats_BASELINE_clean": base_stats,
        "stdev_deltas_FEATURE_ONLY": stdev_deltas,
        "mae_deltas_FEATURE_ONLY": mae_deltas,
        "fold_verdicts": fold_verdicts,
        "hypothesis_verdict": hypothesis,
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_json)

    lines: list[str] = []
    lines.append("# v4.1 hypothesis test — Brent realized-volatility features -> fold_6 stability")
    lines.append("")
    lines.append(
        "Adds 3 Brent realized-volatility features to the upstream block and "
        "re-runs the seed-noise protocol. Tests whether fold_6's residual "
        "instability (clean-baseline seed-stdev 0.100, still the worst fold "
        "after the Phase 3 #4 retune) is a crude-volatility regime the model "
        "couldn't see. Comparison is against the **clean** baseline (new tuned "
        "defaults, no extra features) — same hyperparameters both sides."
    )
    lines.append("")
    lines.append("New features: " + ", ".join(f"`{c}`" for c in NEW_COLS))
    lines.append("")

    lines.append("## Per-fold seed-stdev — feature-only effect")
    lines.append("")
    lines.append("| Fold | WITHOUT vol | WITH vol | Δ stdev | Δ % | Verdict |")
    lines.append("|------|------------:|---------:|--------:|----:|---------|")
    for fk in fold_keys:
        if fk not in new_stats or fk not in base_stats:
            continue
        b = base_stats[fk]["stdev"]
        n = new_stats[fk]["stdev"]
        d = stdev_deltas[fk]
        pct = 100 * d / b if b else 0
        lines.append(f"| {fk} | {b:.4f} | {n:.4f} | {d:+.4f} | {pct:+5.0f}% | {fold_verdicts[fk]} |")
    lines.append("")

    lines.append("## Per-fold seed-mean MAE — feature-only effect")
    lines.append("")
    lines.append("Negative Δ = vol features reduced MAE.")
    lines.append("")
    lines.append("| Fold | WITHOUT vol | WITH vol | Δ mean |")
    lines.append("|------|------------:|---------:|-------:|")
    for fk in fold_keys:
        if fk not in new_stats or fk not in base_stats:
            continue
        lines.append(f"| {fk} | {base_stats[fk]['mean']:.4f} | "
                     f"{new_stats[fk]['mean']:.4f} | {mae_deltas[fk]:+.4f} |")
    lines.append("")

    lines.append("## Per-seed per-fold MAE_A (with vol features)")
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
    f6_b = base_stats.get("fold_6", {}).get("stdev", float("nan"))
    f6_n = new_stats.get("fold_6", {}).get("stdev", float("nan"))
    lines.append(f"- fold_6 seed-stdev: {f6_b:.4f} (without) -> {f6_n:.4f} (with vol)")
    lines.append(f"- **Hypothesis: {hypothesis}**")
    lines.append("")
    if hypothesis == "CONFIRMED":
        lines.append("**CONFIRMED.** Brent realized-volatility was the missing signal for "
                     "fold_6's instability. Roll the vol features into the production "
                     "upstream block; update spec §7.2. Validate mean-MAE isn't degraded "
                     "elsewhere before locking in.")
    elif hypothesis == "PARTIAL":
        lines.append("**PARTIAL.** Vol features help fold_6 but don't fully stabilise it. "
                     "Worth adding if mean MAE holds elsewhere, but fold_6 has residual "
                     "instability from another source.")
    elif hypothesis == "REJECTED":
        lines.append("**REJECTED.** Vol features made fold_6 worse — same near-constant / "
                     "added-capacity failure mode as the v4 excise feature, OR the vol "
                     "signal is genuinely unhelpful. Check the fold_6-window variation "
                     "diagnostic in the build log.")
    else:
        lines.append("**FALSIFIED.** fold_6 seed-stdev essentially unchanged. Its "
                     "instability is not a crude-volatility regime. Next candidates: the "
                     "2026 structural-break *magnitude* (the spike level itself, not its "
                     "volatility), or train-set composition (fold_6 trains on 2017-2025-04 "
                     "and tests on the 2026 spike — the model has never seen a shock of "
                     "that size). The latter may be irreducible without 2026-like training "
                     "data.")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append("- `tools/research/v4_1_brent_vol_fold6.py` — this script")
    lines.append("- `results/v3_phase3_hyperopt_validation.json` — clean baseline")
    lines.append("- `docs/research/2026-06_v4_fold_instability_excise_outcome.md` — v4 (excise) precursor")
    lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", summary_md)

    logger.info("=== v4.1 brent-vol headline ===")
    logger.info("fold_6 stdev WITHOUT vol: %.4f c/L", f6_b)
    logger.info("fold_6 stdev WITH vol:    %.4f c/L", f6_n)
    logger.info("fold_6 verdict:           %s", f6)
    logger.info("Hypothesis:               %s", hypothesis)


def main() -> None:
    logger.info("v4.1 brent-vol fold_6 experiment starting")
    _build_features()
    _patch_upstream_block()
    baseline_per_seed = _load_baseline_per_seed()
    logger.info("loaded clean baseline (new tuned defaults, no extra feats) for %d seeds",
                len(baseline_per_seed))

    new_per_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        try:
            new_per_seed[seed] = _run_seed(seed)
        except Exception as exc:
            logger.exception("seed=%d FAILED: %s", seed, exc)
            new_per_seed[seed] = {"error": f"{type(exc).__name__}: {exc}"}  # type: ignore[dict-item]
        valid = {s: d for s, d in new_per_seed.items() if "error" not in d}
        if len(valid) >= 2:
            _write_summary(valid, baseline_per_seed)

    logger.info("v4.1 brent-vol fold_6 experiment complete")


if __name__ == "__main__":
    main()
