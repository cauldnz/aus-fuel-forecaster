"""v4.2 hypothesis test — do higher return moments (kurtosis, skew) add to v4.1?

User domain pointer (2026-06-10): "Kurtosis of returns often an interesting
feature too." Realized vol (v4.1) is the 2nd moment of the return
distribution. Kurtosis (4th moment) captures fat-tailedness / jump risk —
two windows can share the same vol but differ sharply in jumpiness. A high-
kurtosis regime (calm punctuated by jumps) is plausibly what the 2026 oil
shock looks like, and plain vol averages that out.

This is a **nested** test on top of v4.1:
- v4.1 feature set = base + {vol_14d, vol_30d, vol_ratio_14_90}
- v4.2 feature set = v4.1 + {skew_30d, kurt_30d, kurt_60d}

So three comparisons are available:
- v4.2 vs clean baseline → total effect of all 6 moment features
- v4.2 vs v4.1 → **marginal** effect of skew + kurtosis beyond vol

Higher moments use LONGER windows than vol: a 4th moment estimated on 14
points is garbage. Skew on 30d, kurtosis on 30d and 60d. pandas rolling
skew/kurt use Fisher's definitions (normal skew=0, excess kurt=0).

All windows TRAILING (no lookahead for y_t1). Builds on the v4.1 parquet
so the vol features are already present.

Predicted outcomes (keyed on fold_6 seed-stdev vs clean baseline):
- fold_6 drops materially beyond what v4.1 achieved → kurtosis is the
  missing jump-risk signal; the user's intuition pays off.
- fold_6 unchanged vs v4.1 → higher moments add nothing beyond vol; vol
  was already capturing whatever regime signal exists.
- fold_6 worse → over-capacity (6 added features); the higher moments are
  noise for this fold.

Wall-clock ~50 min. Resume-safe.
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
# Build on the v4.1 parquet (already has the 3 vol features). If it's
# missing, fall back to the base features parquet + recompute vol too.
V41_FEATURES = DATA_PROCESSED / "features_v4_1_brentvol.parquet"
BASE_FEATURES = DATA_PROCESSED / "features.parquet"
OUTPUT_FEATURES = DATA_PROCESSED / "features_v4_2_brentkurt.parquet"

BASELINE_SEED_JSON = RESULTS_DIR / "v3_phase3_hyperopt_validation.json"
V41_SEED_JSON = RESULTS_DIR / "v4_1_brent_vol_fold6.json"

# v4.1 vol features (recomputed here only if building from base parquet).
VOL_COLS = (
    "upstream_brent_realized_vol_14d",
    "upstream_brent_realized_vol_30d",
    "upstream_brent_vol_ratio_14_90",
)
# v4.2 NEW higher-moment features.
MOMENT_COLS = (
    "upstream_brent_return_skew_30d",
    "upstream_brent_return_kurt_30d",
    "upstream_brent_return_kurt_60d",
)
# Full added set in v4.2 (vol + higher moments) relative to the base.
ALL_NEW_COLS = (*VOL_COLS, *MOMENT_COLS)

LOG_PATH = REPO_ROOT / "tools" / "research" / "v4_2_brent_kurtosis_fold6.log"
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
logger = logging.getLogger("v4_2_brentkurt")


def _per_date_returns(df: pd.DataFrame) -> pd.Series:
    """Daily Brent log returns indexed by date."""
    per_date = (
        df.dropna(subset=["upstream_brent_lag_0"])
        .groupby("date")["upstream_brent_lag_0"]
        .first()
        .sort_index()
    )
    brent = per_date.to_numpy(dtype="float64")
    log_ret = np.full(len(brent), np.nan)
    log_ret[1:] = np.log(brent[1:] / brent[:-1])
    return pd.Series(log_ret, index=per_date.index)


def _build_features() -> None:
    """Add skew_30d, kurt_30d, kurt_60d (and vol cols if not inherited)."""
    if OUTPUT_FEATURES.exists():
        import pyarrow.parquet as pq
        schema = pq.read_schema(OUTPUT_FEATURES)
        if all(c in schema.names for c in MOMENT_COLS):
            logger.info("SKIP feature build — %s already has moment cols", OUTPUT_FEATURES)
            return
        logger.warning("cached %s missing moment cols — rebuilding", OUTPUT_FEATURES)

    # Prefer the v4.1 parquet (vol features already present); else base.
    if V41_FEATURES.exists():
        src = V41_FEATURES
        have_vol = True
    else:
        src = BASE_FEATURES
        have_vol = False
    logger.info("loading %s (vol features %s)", src,
                "inherited" if have_vol else "will recompute")
    t0 = time.monotonic()
    df = pd.read_parquet(src)
    logger.info("loaded %d rows x %d cols in %.1fs",
                len(df), len(df.columns), time.monotonic() - t0)

    ret_s = _per_date_returns(df)

    # Higher moments — longer windows for the 4th moment's stability.
    skew_30 = ret_s.rolling(30, min_periods=20).skew()
    kurt_30 = ret_s.rolling(30, min_periods=20).kurt()
    kurt_60 = ret_s.rolling(60, min_periods=40).kurt()
    moment_df = pd.DataFrame({
        "date": ret_s.index,
        "upstream_brent_return_skew_30d": skew_30.to_numpy(),
        "upstream_brent_return_kurt_30d": kurt_30.to_numpy(),
        "upstream_brent_return_kurt_60d": kurt_60.to_numpy(),
    })

    # If vol features weren't inherited, recompute them too (keeps v4.2 a
    # strict superset of v4.1's feature set).
    if not have_vol:
        vol_14 = ret_s.rolling(14, min_periods=7).std()
        vol_30 = ret_s.rolling(30, min_periods=15).std()
        vol_90 = ret_s.rolling(90, min_periods=45).std()
        vol_ratio = vol_14 / vol_90.replace(0.0, np.nan)
        vol_df = pd.DataFrame({
            "date": ret_s.index,
            "upstream_brent_realized_vol_14d": vol_14.to_numpy(),
            "upstream_brent_realized_vol_30d": vol_30.to_numpy(),
            "upstream_brent_vol_ratio_14_90": vol_ratio.to_numpy(),
        })
        df = df.merge(vol_df, on="date", how="left")

    df = df.merge(moment_df, on="date", how="left")

    for col in MOMENT_COLS:
        s = df[col]
        logger.info("new column %s: min=%.4f max=%.4f mean=%.4f null=%d (%.1f%%)",
                    col, s.min(), s.max(), s.mean(), s.isna().sum(),
                    100 * s.isna().sum() / len(df))

    # fold_6-window variation diagnostic (the v4 near-constant guard).
    date_ts = pd.to_datetime(df["date"])
    f6 = (date_ts >= "2025-05-01") & (date_ts <= "2026-04-30")
    for col in MOMENT_COLS:
        fw = df.loc[f6, col]
        full = df[col]
        logger.info("fold_6-window %s: mean=%.4f stdev=%.4f (full stdev=%.4f) -> %s",
                    col, fw.mean(), fw.std(), full.std(),
                    "OK (variable)" if fw.std() > 0.2 * full.std() else "WARN near-constant")

    t0 = time.monotonic()
    df.to_parquet(OUTPUT_FEATURES, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %s in %.1fs", OUTPUT_FEATURES, time.monotonic() - t0)


def _patch_upstream_block() -> None:
    from fuel_pred.train import feature_blocks
    orig = feature_blocks.UPSTREAM_COLUMNS
    if all(c in orig for c in ALL_NEW_COLS):
        return
    # Add only the columns not already present (vol may be there if a prior
    # v4.1 patch ran in the same process — unlikely, but defensive).
    additions = tuple(c for c in ALL_NEW_COLS if c not in orig)
    new = (*orig, *additions)
    feature_blocks.UPSTREAM_COLUMNS = new
    feature_blocks.BLOCK_COLUMNS["upstream"] = new
    logger.info("UPSTREAM_COLUMNS monkey-patched: %d -> %d cols (added %s)",
                len(orig), len(new), list(additions))


def _import_train_kfold() -> Callable[..., object]:
    from fuel_pred.train.cv import train_kfold
    return train_kfold


def _run_seed(seed: int) -> dict[str, float]:
    out_root = REPO_ROOT / f"models_kfold_v4_2_brentkurt_seed_{seed}"
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
    logger.info("[seed=%d] training Model A across 6 folds (vol + higher moments)", seed)
    logger.info("=" * 70)
    train_kfold = _import_train_kfold()
    t0 = time.monotonic()
    train_kfold(OUTPUT_FEATURES, out_root, random_state=seed,
                models_to_fit=("A",), save_predictions=False)
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
        out[f"fold_{int(entry['fold'])}"] = float(entry["models"]["A"]["best_val_mae"])
    if len(out) != 6:
        raise RuntimeError(f"expected 6 folds in {audit_path}, got {len(out)}")
    return out


def _load_seed_json(path: Path, key: str) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data[key]
    return {int(k): {fk: float(v) for fk, v in d.items() if fk.startswith("fold_")}
            for k, d in raw.items()}


def _stats(per_seed: dict[int, dict[str, float]]) -> dict:
    fold_keys = [f"fold_{i+1}" for i in range(6)]
    out = {}
    for fk in fold_keys:
        vals = [d[fk] for d in per_seed.values() if fk in d]
        if len(vals) >= 2:
            out[fk] = {"mean": statistics.fmean(vals), "stdev": statistics.pstdev(vals)}
    return out


def _write_summary(
    new_per_seed: dict[int, dict[str, float]],
    clean_base: dict[int, dict[str, float]],
    v41: dict[int, dict[str, float]],
) -> None:
    summary_md = RESULTS_DIR / "v4_2_brent_kurtosis_fold6_summary.md"
    summary_json = RESULTS_DIR / "v4_2_brent_kurtosis_fold6.json"
    fold_keys = [f"fold_{i+1}" for i in range(6)]

    new_stats = _stats(new_per_seed)
    base_stats = _stats(clean_base)
    v41_stats = _stats(v41)

    # Marginal effect (v4.2 vs v4.1) keyed on fold_6 stdev.
    f6_base = base_stats.get("fold_6", {}).get("stdev", float("nan"))
    f6_v41 = v41_stats.get("fold_6", {}).get("stdev", float("nan"))
    f6_new = new_stats.get("fold_6", {}).get("stdev", float("nan"))

    marginal = f6_new - f6_v41 if v41_stats else float("nan")
    if v41_stats:
        if marginal < -0.2 * f6_v41:
            verdict = "HIGHER MOMENTS HELP (beyond vol)"
        elif marginal > 0.2 * f6_v41:
            verdict = "HIGHER MOMENTS HURT (beyond vol)"
        else:
            verdict = "HIGHER MOMENTS NEUTRAL (vol already captured it)"
    else:
        verdict = "v4.1 results unavailable — only total-vs-baseline comparison"

    payload = {
        "seeds": list(new_per_seed.keys()),
        "new_features_total": list(ALL_NEW_COLS),
        "new_features_vs_v41": list(MOMENT_COLS),
        "per_seed_per_fold_mae_a_NEW": new_per_seed,
        "per_fold_seed_stats_NEW_vol_plus_moments": new_stats,
        "per_fold_seed_stats_CLEAN_baseline": base_stats,
        "per_fold_seed_stats_V41_vol_only": v41_stats,
        "fold6_stdev_clean_baseline": f6_base,
        "fold6_stdev_v41_vol_only": f6_v41,
        "fold6_stdev_v42_vol_plus_moments": f6_new,
        "marginal_verdict_vs_v41": verdict,
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_json)

    lines: list[str] = []
    lines.append("# v4.2 — Brent higher-moment (kurtosis + skew) features for fold_6")
    lines.append("")
    lines.append("Nested test on top of v4.1 (realized vol). Adds rolling skew (30d) "
                 "and kurtosis (30d, 60d) of Brent returns. Three-way comparison "
                 "isolates the **marginal** value of higher moments beyond vol.")
    lines.append("")
    lines.append("New vs v4.1: " + ", ".join(f"`{c}`" for c in MOMENT_COLS))
    lines.append("")

    lines.append("## fold_6 seed-stdev — the three-way comparison")
    lines.append("")
    lines.append("| Config | fold_6 seed-stdev |")
    lines.append("|--------|------------------:|")
    lines.append(f"| Clean baseline (no extra feats) | {f6_base:.4f} |")
    lines.append(f"| v4.1 (+ vol) | {f6_v41:.4f} |")
    lines.append(f"| v4.2 (+ vol + skew + kurtosis) | {f6_new:.4f} |")
    lines.append("")
    lines.append(f"**Marginal verdict (v4.2 vs v4.1): {verdict}**")
    lines.append("")

    lines.append("## Per-fold seed-stdev — all configs")
    lines.append("")
    lines.append("| Fold | Clean baseline | v4.1 vol-only | v4.2 vol+moments |")
    lines.append("|------|---------------:|--------------:|-----------------:|")
    for fk in fold_keys:
        b = base_stats.get(fk, {}).get("stdev", float("nan"))
        v = v41_stats.get(fk, {}).get("stdev", float("nan"))
        n = new_stats.get(fk, {}).get("stdev", float("nan"))
        lines.append(f"| {fk} | {b:.4f} | {v:.4f} | {n:.4f} |")
    lines.append("")

    lines.append("## Per-fold seed-mean MAE — v4.2 vs clean baseline")
    lines.append("")
    lines.append("| Fold | Clean baseline | v4.2 | Δ mean |")
    lines.append("|------|---------------:|-----:|-------:|")
    for fk in fold_keys:
        b = base_stats.get(fk, {}).get("mean", float("nan"))
        n = new_stats.get(fk, {}).get("mean", float("nan"))
        lines.append(f"| {fk} | {b:.4f} | {n:.4f} | {n-b:+.4f} |")
    lines.append("")

    lines.append("## Per-seed per-fold MAE_A (v4.2)")
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

    lines.append("## Sources")
    lines.append("")
    lines.append("- `tools/research/v4_2_brent_kurtosis_fold6.py` — this script")
    lines.append("- `results/v4_1_brent_vol_fold6.json` — v4.1 (vol-only) baseline for the marginal comparison")
    lines.append("- `results/v3_phase3_hyperopt_validation.json` — clean baseline")
    lines.append("- User domain pointer: kurtosis of returns as a regime feature")
    lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", summary_md)

    logger.info("=== v4.2 brent-kurtosis headline ===")
    logger.info("fold_6 stdev: clean %.4f | v4.1 vol %.4f | v4.2 vol+moments %.4f",
                f6_base, f6_v41, f6_new)
    logger.info("Marginal verdict (vs v4.1): %s", verdict)


def main() -> None:
    logger.info("v4.2 brent-kurtosis fold_6 experiment starting")
    _build_features()
    _patch_upstream_block()
    clean_base = _load_seed_json(BASELINE_SEED_JSON, "per_seed_per_fold_mae_a_NEW")
    v41 = _load_seed_json(V41_SEED_JSON, "per_seed_per_fold_mae_a_NEW")
    logger.info("loaded clean baseline (%d seeds) + v4.1 vol-only (%d seeds)",
                len(clean_base), len(v41))

    new_per_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        try:
            new_per_seed[seed] = _run_seed(seed)
        except Exception as exc:
            logger.exception("seed=%d FAILED: %s", seed, exc)
            new_per_seed[seed] = {"error": f"{type(exc).__name__}: {exc}"}  # type: ignore[dict-item]
        valid = {s: d for s, d in new_per_seed.items() if "error" not in d}
        if len(valid) >= 2:
            _write_summary(valid, clean_base, v41)

    logger.info("v4.2 brent-kurtosis fold_6 experiment complete")


if __name__ == "__main__":
    main()
