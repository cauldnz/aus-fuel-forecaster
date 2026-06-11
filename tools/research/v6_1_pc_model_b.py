"""v6.1 — does a PC-compressed augmentor block beat Model A?

v6 showed the 37-column augmentor surface is ~4-dimensional (80% variance
in 4 PCs; 26/37 columns near-perfectly reconstructable from the rest).
Every prior augmentor experiment fed the model RAW columns — which carry
the overfitting tax that the v1.x 31-col broadening hit. This tests the
one representation we never tried: **distil the augmentor to its top 4
principal components and feed those as the SA2 block.**

It's the augmentor's best possible shot. An orthogonal, low-rank,
overfit-resistant encoding of the same socioeconomic gradients. If
*anything* augmentor-shaped can beat Model A, this is it. A null here is
the final nail: even the augmentor's information distilled to its essence
adds nothing over the lag-rich baseline.

Design:
- Fit StandardScaler + PCA(4) on the 580 unique complete SA2 profiles
  (full 37-col surface from stations.parquet).
- Project every station's 37-col vector -> 4 PC values; join onto the
  panel by station_id as columns sa2_pc1..sa2_pc4.
- Monkey-patch the SA2 block to those 4 PC columns.
- train_kfold (Model A vs Model B-PC, A+B only, v3.0 tuned defaults),
  target y_t1, default 6-fold geometry; then compare_kfold.

Leakage note: PCA is **unsupervised** and fit on **exogenous static**
census data (it never sees y_t1, and SA2 demographics don't carry future
price information). Fitting the rotation on all SA2 profiles is therefore
not target leakage — it's equivalent to using a fixed published census
embedding. (Per-fold refitting would barely move a rotation estimated
from 580 SA2s and is unnecessary for an unsupervised exogenous transform.)

Compares against the v3.0 t+1 PR-B baseline (raw 15-col block):
Mean Δ MAE +0.215, Stdev 0.394 (noise).

Wall-clock ~10-15 min (A+B only, 6 folds). Writes
results/v6_1_pc_model_b_kfold.md + headline + json.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
RESULTS_DIR = REPO_ROOT / "results"

FEATURES = DATA_PROCESSED / "features.parquet"
STATIONS = DATA_INTERIM / "stations.parquet"
OUT_FEATURES = DATA_PROCESSED / "features_v6_1_pc.parquet"
OUT_MODELS = REPO_ROOT / "models_kfold_v6_1_pc"
OUT_REPORT = RESULTS_DIR / "v6_1_pc_model_b_kfold.md"
OUT_HEADLINE = RESULTS_DIR / "v6_1_pc_model_b_headline.md"

N_PC = 4  # v6 intrinsic dimensionality (80% variance)
PC_COLS = tuple(f"sa2_pc{i+1}" for i in range(N_PC))

T1_RAW_REF = {"mean_delta_mae": 0.215, "stdev_delta_mae": 0.394}

LOG_PATH = REPO_ROOT / "tools" / "research" / "v6_1_pc_model_b.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, mode="a")],
)
logger = logging.getLogger("v6_1_pc")


def _build_features() -> None:
    """Fit PCA on the full augmentor surface, project to N_PC, join into panel."""
    if OUT_FEATURES.exists():
        import pyarrow.parquet as pq
        if all(c in pq.read_schema(OUT_FEATURES).names for c in PC_COLS):
            logger.info("SKIP feature build — %s already has PC cols", OUT_FEATURES)
            return

    import pyarrow.parquet as pq
    num_cols = [
        n for n in pq.read_schema(STATIONS).names
        if n.startswith("sa2_") and n not in ("sa2_code", "sa2_name")
    ]
    st = pd.read_parquet(STATIONS, columns=["station_id", "sa2_code", *num_cols])

    # Fit on unique complete SA2 profiles (unsupervised, exogenous).
    uniq = st.drop_duplicates("sa2_code").dropna(subset=num_cols)
    logger.info("fitting PCA(%d) on %d unique complete SA2 profiles (%d cols)",
                N_PC, len(uniq), len(num_cols))
    scaler = StandardScaler().fit(uniq[num_cols].to_numpy(float))
    pca = PCA(n_components=N_PC, random_state=42).fit(
        scaler.transform(uniq[num_cols].to_numpy(float)))
    logger.info("PCA explained variance (cum): %.3f  per-PC: %s",
                pca.explained_variance_ratio_.sum(),
                ", ".join(f"{v:.3f}" for v in pca.explained_variance_ratio_))

    # Project every station (NaN where its SA2 profile is incomplete).
    st_complete = st.dropna(subset=num_cols)
    proj = pca.transform(scaler.transform(st_complete[num_cols].to_numpy(float)))
    pc_df = pd.DataFrame(proj, columns=list(PC_COLS))
    pc_df["station_id"] = st_complete["station_id"].to_numpy()
    # One row per station_id (stations.parquet is already one row per station,
    # but guard against dupes).
    pc_df = pc_df.drop_duplicates("station_id")

    logger.info("loading %s", FEATURES)
    t0 = time.monotonic()
    df = pd.read_parquet(FEATURES)
    logger.info("loaded %d rows in %.1fs", len(df), time.monotonic() - t0)
    df = df.merge(pc_df, on="station_id", how="left")
    cov = df[list(PC_COLS)].notna().all(axis=1).mean()
    logger.info("PC columns joined; non-null coverage on panel: %.1f%%", 100 * cov)

    df.to_parquet(OUT_FEATURES, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %s", OUT_FEATURES)


def _run_kfold() -> None:
    from fuel_pred.build import make_features
    from fuel_pred.evaluate.compare import compare_kfold
    from fuel_pred.train import feature_blocks
    from fuel_pred.train.cv import train_kfold
    from fuel_pred.train.folds import KFoldConfig

    orig_sa2 = feature_blocks.SA2_COLUMNS
    orig_block = feature_blocks.BLOCK_COLUMNS["sa2"]
    orig_mf = make_features.SA2_FEATURE_COLS
    feature_blocks.SA2_COLUMNS = PC_COLS
    feature_blocks.BLOCK_COLUMNS["sa2"] = PC_COLS
    make_features.SA2_FEATURE_COLS = PC_COLS
    logger.info("SA2 block monkey-patched to %d PC columns: %s", N_PC, list(PC_COLS))

    try:
        audit = OUT_MODELS / "kfold_audit.json"
        if audit.exists() and OUT_REPORT.exists():
            logger.info("SKIP train+compare — audit + report already exist")
        else:
            t0 = time.monotonic()
            train_kfold(OUT_FEATURES, OUT_MODELS, kfold_config=KFoldConfig(),
                        models_to_fit=("A", "B"))
            logger.info("train_kfold done in %.1f min", (time.monotonic() - t0) / 60)
            t0 = time.monotonic()
            compare_kfold(OUT_FEATURES, OUT_MODELS, OUT_REPORT)
            logger.info("compare_kfold done in %.1f min", (time.monotonic() - t0) / 60)
    finally:
        feature_blocks.SA2_COLUMNS = orig_sa2
        feature_blocks.BLOCK_COLUMNS["sa2"] = orig_block
        make_features.SA2_FEATURE_COLS = orig_mf


def _write_headline() -> None:
    import re
    if not OUT_REPORT.exists():
        logger.warning("no report at %s", OUT_REPORT)
        return
    text = OUT_REPORT.read_text(encoding="utf-8")
    fold_row = re.compile(
        r"^\|\s*fold_(\d+)\s*\|\s*[\d-]+\s*→\s*[\d-]+\s*\|"
        r"\s*[\d,]+\s*\|\s*(-?\d+\.\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*([+-]?\d+\.\d+)\s*\|")
    per_fold = []
    for line in text.splitlines():
        if len(per_fold) >= 6:
            break
        m = fold_row.match(line)
        if m:
            per_fold.append({"fold": int(m.group(1)), "mae_a": float(m.group(2)),
                             "mae_b": float(m.group(3)), "delta_mae": float(m.group(4))})
    agg = {}
    for label, key in (("**Mean**", "mean"), ("Stdev", "stdev")):
        m = re.search(rf"\|\s*{re.escape(label)}\s*\|[^\|]+\|[^\|]+\|"
                      rf"([^\|]+)\|([^\|]+)\|([^\|]+)\|", text)
        if m:
            agg[f"{key}_delta_mae"] = float(m.group(3).strip())
    mean_d = agg.get("mean_delta_mae", float("nan"))
    stdev_d = agg.get("stdev_delta_mae", float("nan"))
    if abs(mean_d) > 2 * stdev_d:
        verdict = "ROBUST WIN (B-PC beats A)" if mean_d < 0 else "ROBUST (B-PC loses)"
    elif abs(mean_d) > stdev_d:
        verdict = "weak (B-PC beats A)" if mean_d < 0 else "weak (B-PC loses)"
    else:
        verdict = "noise"

    payload = {"n_pc": N_PC, "pc_cols": list(PC_COLS), "per_fold": per_fold,
               "aggregate": agg, "verdict": verdict, "t1_raw_reference": T1_RAW_REF}
    (RESULTS_DIR / "v6_1_pc_model_b.json").write_text(json.dumps(payload, indent=2),
                                                      encoding="utf-8")
    lines = [
        f"# v6.1 — PC-compressed augmentor block ({N_PC} PCs) vs Model A",
        "",
        f"Distils the full 37-col augmentor surface to its top {N_PC} principal "
        f"components (the v6 intrinsic dimensionality, ~80% variance) and tests "
        "Model A vs Model B-PC under 6-fold k-fold. The augmentor's best possible "
        "representation — orthogonal, low-rank, overfit-resistant.",
        "",
        f"Full report: `{OUT_REPORT.relative_to(REPO_ROOT)}`",
        "",
        "## Headline",
        "",
        f"- **Mean Δ MAE: {mean_d:+.3f} c/L** (negative = Model B-PC beats Model A)",
        f"- Stdev across 6 folds: {stdev_d:.3f}",
        f"- **Verdict: {verdict}**",
        "",
        "## vs the raw-column augmentor (v3.0 t+1 PR B baseline)",
        "",
        "| Config | Mean Δ MAE | Stdev |",
        "|--------|-----------:|------:|",
        f"| Raw 15-col SA2 block | {T1_RAW_REF['mean_delta_mae']:+.3f} | {T1_RAW_REF['stdev_delta_mae']:.3f} |",
        f"| {N_PC}-PC compressed block | {mean_d:+.3f} | {stdev_d:.3f} |",
        "",
    ]
    if per_fold:
        lines += ["## Per-fold Δ MAE", "", "| Fold | MAE A | MAE B-PC | Δ MAE |",
                  "|------|------:|---------:|------:|"]
        for f in per_fold:
            lines.append(f"| fold_{f['fold']} | {f['mae_a']:.3f} | {f['mae_b']:.3f} | {f['delta_mae']:+.3f} |")
        lines.append("")
    lines += ["## Reading", ""]
    if mean_d < -stdev_d:
        lines.append("**The PC representation beats Model A** where raw columns "
                     "couldn't — distilling the augmentor to its orthogonal essence "
                     "unlocked signal the redundant raw block buried. A genuine, "
                     "surprising result worth a full follow-up.")
    elif abs(mean_d) <= stdev_d:
        lines.append("**Null — the PC block is no better than the raw block.** Even "
                     "the augmentor's information distilled to its 4 orthogonal "
                     "essence-dimensions adds nothing over Model A. This is the final "
                     "nail: the null isn't a representation artifact (raw vs PC), it's "
                     "the information content. The ~4 socioeconomic gradients are "
                     "redundant with what the lag features already encode.")
    else:
        lines.append("**The PC block makes things worse.** The compression didn't "
                     "help; consistent with the augmentor adding noise, not signal.")
    lines.append("")
    lines += ["## Sources", "",
              f"- `{OUT_REPORT.relative_to(REPO_ROOT)}` — full per-fold report",
              "- `tools/research/v6_1_pc_model_b.py` — this script",
              "- `docs/research/2026-06_v6_augmentor_intrinsic_dimensionality.md` — the PCA analysis",
              "- `results/v3_phase2_pr_b_baseline_kfold.md` — raw-column reference", ""]
    OUT_HEADLINE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", OUT_HEADLINE)
    logger.info("=== v6.1 PC Model B headline ===")
    logger.info("Mean Delta MAE: %+.3f (raw ref %+.3f) | Stdev %.3f | Verdict: %s",
                mean_d, T1_RAW_REF["mean_delta_mae"], stdev_d, verdict)


def main() -> None:
    logger.info("v6.1 PC-compressed Model B experiment starting (N_PC=%d)", N_PC)
    _build_features()
    _run_kfold()
    _write_headline()
    logger.info("v6.1 experiment complete")


if __name__ == "__main__":
    main()
