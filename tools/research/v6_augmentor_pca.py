"""v6 — intrinsic dimensionality of the FULL augmentor surface.

A complementary lens on the augmentor-utility question. The v2.x/v3.0
fuel experiments asked "does the augmentor help predict prices?" (no,
robustly, across 8 variants / 6 folds / 6 seeds / 2 horizons). This asks
a property of the augmentor data *itself*, independent of the fuel
problem: how internally redundant is it / how low is its intrinsic
dimensionality?

If any one SA2 column is highly predictable from the others, the surface
isn't N independent signals — it's a few socioeconomic gradients wearing
N hats. That mechanistically *explains* the fuel null: a handful of broad,
slow-moving cross-sectional gradients is exactly what per-station price
history already encodes implicitly, so the augmentor is redundant with
what the lag features already carry.

This v6 run uses the FULL 37-numeric-column augmentor surface from
``data/interim/stations.parquet`` — 2.5x the curated 15-col model block
that the earlier scratch probe used. The full surface is the honest unit
for the dimensionality question (the 15-col block was already curated,
which understates redundancy).

GOTCHA handled: SA2 columns are broadcast across stations / station-days.
We dedupe to UNIQUE SA2 profiles (one row per ``sa2_code``) — otherwise
duplication trivially inflates every R^2.

Three analyses:
  1. PCA / SVD -> intrinsic dimensionality (PCs for 80/90/95/99% variance)
  2. Leave-one-column-out predictability (Ridge linear + GBM nonlinear),
     5-fold CV R^2 per column
  3. Correlation structure + PC loadings (names the latent axes)

Writes results/v6_augmentor_pca_summary.md + .json. No model training,
no network, runs in seconds.
"""
# Uppercase X for the 2-D feature matrix is sklearn convention (matches
# fuel_pred.train._fit). Suppress the pep8-naming rule for this module.
# ruff: noqa: N806
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIONS = REPO_ROOT / "data" / "interim" / "stations.parquet"
RESULTS_DIR = REPO_ROOT / "results"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("v6_aug_pca")

# Short labels for readable loadings/output.
def _short(c: str) -> str:
    return c.replace("sa2_", "").replace("_recipients", "")


def main() -> None:
    logger.info("loading full augmentor surface from %s", STATIONS)
    import pyarrow.parquet as pq
    schema = pq.read_schema(STATIONS)
    num_cols = [
        n for n in schema.names
        if n.startswith("sa2_") and n not in ("sa2_code", "sa2_name")
    ]
    df = pd.read_parquet(STATIONS, columns=["sa2_code", *num_cols])

    # Dedupe to unique SA2 profiles, drop any-null rows.
    uniq = df.drop_duplicates("sa2_code").dropna(subset=num_cols).reset_index(drop=True)
    logger.info("unique complete SA2 profiles: %d  | numeric columns: %d",
                len(uniq), len(num_cols))

    X = uniq[num_cols].to_numpy(dtype="float64")
    Xz = StandardScaler().fit_transform(X)

    # ---- 1. PCA / intrinsic dimensionality ------------------------------
    _u, s, vt = np.linalg.svd(Xz, full_matrices=False)
    evr = (s**2) / (s**2).sum()
    cum = np.cumsum(evr)
    n_for = {p: int(np.searchsorted(cum, p) + 1) for p in (0.80, 0.90, 0.95, 0.99)}
    logger.info("=== 1. Intrinsic dimensionality (%d columns) ===", len(num_cols))
    for p, n in n_for.items():
        logger.info("  %d%% variance: %d components", int(p * 100), n)
    logger.info("  top-8 explained-variance ratio: %s",
                ", ".join(f"{e:.3f}" for e in evr[:8]))

    # ---- 2. Leave-one-column-out predictability -------------------------
    logger.info("=== 2. Leave-one-column-out CV R^2 ===")
    rows = []
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    for j, col in enumerate(num_cols):
        y = Xz[:, j]
        Xo = np.delete(Xz, j, axis=1)
        r2_ridge = float(cross_val_score(ridge, Xo, y, cv=5, scoring="r2").mean())
        gbm = HistGradientBoostingRegressor(max_iter=200, max_depth=4,
                                            learning_rate=0.05, random_state=42)
        r2_gbm = float(cross_val_score(gbm, Xo, y, cv=5, scoring="r2").mean())
        rows.append({"column": col, "r2_ridge": r2_ridge, "r2_gbm": r2_gbm})
    r2_df = pd.DataFrame(rows)
    mean_ridge, mean_gbm = r2_df["r2_ridge"].mean(), r2_df["r2_gbm"].mean()
    med_gbm = r2_df["r2_gbm"].median()
    n_high = int((r2_df["r2_gbm"] > 0.5).sum())
    n_vhigh = int((r2_df["r2_gbm"] > 0.9).sum())
    logger.info("  MEAN R^2 ridge=%.3f gbm=%.3f | MEDIAN gbm=%.3f", mean_ridge, mean_gbm, med_gbm)
    logger.info("  GBM R^2 > 0.5: %d/%d  | > 0.9: %d/%d",
                n_high, len(num_cols), n_vhigh, len(num_cols))

    # ---- 3. Correlation structure + PC loadings -------------------------
    corr = np.corrcoef(Xz, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    max_abs = np.abs(corr).max(axis=1)
    logger.info("=== 3. Correlation ===")
    logger.info("  mean max|r|=%.3f  | cols with a >0.7 partner: %d/%d  | >0.9: %d/%d",
                max_abs.mean(), int((max_abs > 0.7).sum()), len(num_cols),
                int((max_abs > 0.9).sum()), len(num_cols))

    loadings = []
    logger.info("=== PC loadings (name the axes) ===")
    for pc in range(4):
        load = vt[pc]
        order = np.argsort(-np.abs(load))
        top = [(num_cols[i], float(load[i])) for i in order[:6]]
        loadings.append({"pc": pc + 1, "evr": float(evr[pc]), "top": top})
        logger.info("  PC%d (%.0f%% var): %s", pc + 1, evr[pc] * 100,
                    ", ".join(f"{v:+.2f} {_short(c)}" for c, v in top))

    # ---- write artefacts -------------------------------------------------
    payload = {
        "source": "data/interim/stations.parquet (full augmentor surface)",
        "n_unique_sa2": len(uniq),
        "n_columns": len(num_cols),
        "columns": num_cols,
        "pca_components_for_variance": n_for,
        "explained_variance_ratio": [float(e) for e in evr],
        "cumulative_variance": [float(c) for c in cum],
        "leave_one_out_r2": rows,
        "mean_r2_ridge": float(mean_ridge),
        "mean_r2_gbm": float(mean_gbm),
        "median_r2_gbm": float(med_gbm),
        "n_gbm_r2_over_0.5": n_high,
        "n_gbm_r2_over_0.9": n_vhigh,
        "max_abs_corr_per_col": {num_cols[i]: float(max_abs[i]) for i in range(len(num_cols))},
        "mean_max_abs_corr": float(max_abs.mean()),
        "pc_loadings_top": loadings,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "v6_augmentor_pca.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    # markdown summary
    lines: list[str] = []
    lines.append("# v6 — intrinsic dimensionality of the full augmentor surface")
    lines.append("")
    lines.append(f"Self-prediction / PCA analysis of the **{len(num_cols)}-column** "
                 f"full augmentor surface (`data/interim/stations.parquet`), "
                 f"{len(uniq)} unique SA2 profiles. A property of the augmentor "
                 "data itself, independent of the fuel problem.")
    lines.append("")
    lines.append("## 1. Intrinsic dimensionality")
    lines.append("")
    lines.append(f"| Variance explained | Components (of {len(num_cols)}) |")
    lines.append("|---|---:|")
    for p, n in n_for.items():
        lines.append(f"| {int(p*100)}% | {n} |")
    lines.append("")
    lines.append("Top-8 explained-variance ratio: " +
                 ", ".join(f"{e:.3f}" for e in evr[:8]) + ".")
    lines.append("")
    lines.append("## 2. Leave-one-column-out predictability")
    lines.append("")
    lines.append(f"- **Mean R² = {mean_gbm:.3f}** (GBM), median {med_gbm:.3f}; Ridge mean {mean_ridge:.3f}")
    lines.append(f"- GBM R² > 0.5 (majority-predictable): **{n_high}/{len(num_cols)}**")
    lines.append(f"- GBM R² > 0.9 (near-perfectly reconstructable): **{n_vhigh}/{len(num_cols)}**")
    lines.append("")
    lines.append("| Column | Ridge R² | GBM R² |")
    lines.append("|--------|---------:|-------:|")
    for r in sorted(rows, key=lambda x: -x["r2_gbm"]):
        lines.append(f"| {r['column']} | {r['r2_ridge']:.3f} | {r['r2_gbm']:.3f} |")
    lines.append("")
    lines.append("## 3. Correlation + latent axes")
    lines.append("")
    lines.append(f"- Mean max |r| to any other column: **{max_abs.mean():.3f}**")
    lines.append(f"- Columns with a >0.7 correlated partner: {int((max_abs>0.7).sum())}/{len(num_cols)}; "
                 f">0.9: {int((max_abs>0.9).sum())}/{len(num_cols)}")
    lines.append("")
    lines.append("**Top principal-component loadings (the latent axes):**")
    lines.append("")
    for ld in loadings:
        top_str = ", ".join(f"{v:+.2f} {_short(c)}" for c, v in ld["top"])
        lines.append(f"- **PC{ld['pc']} ({ld['evr']*100:.0f}% var):** {top_str}")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("- `tools/research/v6_augmentor_pca.py` — this script")
    lines.append("- `data/interim/stations.parquet` — full materialized augmentor surface")
    lines.append("- Complements the fuel-utility null in "
                 "`docs/research/2026-06_v3.0_phase3_closing_summary.md`")
    lines.append("")
    (RESULTS_DIR / "v6_augmentor_pca_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote results/v6_augmentor_pca_summary.md + .json")


if __name__ == "__main__":
    main()
