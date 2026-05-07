"""Generate the three notebook templates with sections per spec.md §9.

Run once from the repo root:

    python tools/make_notebooks.py

This is a one-shot tool — once notebooks are populated by the EDA / modeling /
explainability work, they are owned by their respective phases and this
generator should not overwrite them. Re-running this script will fail unless
--force is passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str = "") -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True) if text else [],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (fuel-pred)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_CODE = """\
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fuel_pred import config

features = pd.read_parquet(config.DATA_PROCESSED / "features.parquet")
print(f"Rows: {len(features):,}")
print(f"Stations: {features['station_id'].nunique():,}")
print(f"Date range: {features['date'].min()} → {features['date'].max()}")
features.head()
"""


def build_eda() -> dict:
    return notebook(
        [
            md("# 01 — Exploratory Data Analysis\n\nReads `data/processed/features.parquet`. Per spec.md §9.1."),
            md("## Setup"),
            code(SETUP_CODE),
            md("## 1. Dataset overview\n\nStation count over time, fuel-code coverage, observation density."),
            code("# TODO: station count over time, fuel mix, missingness heatmap"),
            md("## 2. Geographic distribution\n\nMap of stations coloured by SA2 SEIFA, brand mix by region."),
            code("# TODO: map (matplotlib or plotly), brand mix by SA2 SEIFA quintile"),
            md("## 3. Price level and dispersion\n\nBy fuel, by brand, over time."),
            code("# TODO: price distributions per fuel; ECDFs; brand boxplots"),
            md("## 4. The petrol cycle\n\nAutocorrelation by station; FFT on a sample station to demonstrate the ~3-week period.\n\n**Sanity check (per spec.md §13 question 4):** verify the cycle is endogenously captured by lag features by training a tiny LightGBM on a single station with lag features only and inspecting predictions."),
            code("# TODO: ACF plot, FFT, sanity-check tiny model"),
            md("## 5. The 2026 crisis\n\nVisible regime change in Brent + retail prices. Overlay `data/static/crisis_events.csv` markers."),
            code("# TODO: dual-axis plot of Brent and average retail U91; vertical lines for crisis events"),
            md("## 6. Centrelink-day check ⭐\n\n**This is the augmentor-story chart and must be in the notebook.** Average price residual (vs 28-day rolling mean) by `cal_day_of_fortnight`, segmented by SEIFA quintile.\n\nIf the augmentor adds value, this chart should show meaningfully different fortnight-pattern shapes between SEIFA quintiles."),
            code("# TODO: residual = price_mean - roll_price_mean_28\n# Group by (cal_day_of_fortnight, sa2_seifa_quintile), plot mean residual"),
            md("## 7. Cross-correlations\n\nBrent (lagged) vs retail at Sydney metro vs regional. Motivates the lag block."),
            code("# TODO: cross-correlation plots by metro/regional split"),
            md("## 8. Missingness map\n\nFraction of rows that lack each `sa2_*` variable — sanity check on the augmentor coverage."),
            code("# TODO: per-column missingness for sa2_* features"),
        ]
    )


def build_modeling() -> dict:
    return notebook(
        [
            md("# 02 — Modeling\n\nFit Models A and B per spec.md §8 and §9.2."),
            md("## Setup"),
            code(SETUP_CODE),
            md("## 1. Define folds\n\nTime-based, no shuffling. See `fuel_pred.config` for the exact dates."),
            code("# TODO: split features into train / val / test_normal / test_crisis using config dates"),
            md("## 2. Define feature columns\n\nModel A: lag, upstream, calendar, ctx, stn, wx (no `sa2_*`).\nModel B: same plus `sa2_*`.\n\n**Identical training rows** for both — only rows where every Model B column is non-null."),
            code("# TODO: column lists for A and B; row mask = all sa2_* non-null"),
            md("## 3. Fit Model A"),
            code("# TODO: lgb.LGBMRegressor with config.LGBM_PARAMS, early stopping on val"),
            md("## 4. Fit Model B"),
            code("# TODO: same hyperparameters, with sa2_* columns added"),
            md("## 5. Headline metrics\n\nMAE / RMSE / MAPE / median / p90 absolute error on each test fold."),
            code("# TODO: side-by-side metrics table"),
            md("## 6. Segmented metrics\n\nBy metro/regional, brand, fuel type, SEIFA quintile."),
            code("# TODO: segmented tables — this is the headline result of the project"),
            md("## 7. Residual diagnostics\n\nResiduals over time, check for crisis-period blowup."),
            code("# TODO: residuals over time, by station type"),
            md("## 8. Save\n\nWrite `models/model_a.pkl`, `models/model_b.pkl`, `results/comparison.md`."),
            code("# TODO: pickle models, write comparison.md via fuel_pred.evaluate.compare"),
        ]
    )


def build_explainability() -> dict:
    return notebook(
        [
            md("# 03 — Explainability\n\nSHAP analysis per spec.md §9.3. Reads saved Model B from `models/`."),
            md("## Setup"),
            code(SETUP_CODE),
            code("# TODO: load model_b.pkl, prepare a test-fold sample for SHAP"),
            md("## 1. SHAP summary plot — Model B (top 30 features)\n\nWrite output to `results/shap/summary_b.png`."),
            code("# TODO: shap.TreeExplainer, summary plot"),
            md("## 2. SHAP dependence plots for top SA2 features\n\nOne plot per top-ranking `sa2_*` feature. Saves to `results/shap/dependence_<feature>.png`."),
            code("# TODO: dependence plots for top sa2_* features"),
            md("## 3. SHAP interaction plot ⭐\n\n`cal_day_of_fortnight × sa2_seifa_irsd_score` — the demonstration of the augmentor's interaction value.\n\nWrite output to `results/shap/interaction_dof_seifa.png`."),
            code("# TODO: shap.dependence_plot with interaction_index=sa2_seifa_irsd_score on cal_day_of_fortnight"),
            md("## 4. Comparison of top-20 feature importances: Model A vs Model B\n\nIf the augmentor is doing its job, Model B's top-20 should include `sa2_*` features."),
            code("# TODO: side-by-side bar charts"),
            md("## 5. Per-station case studies\n\n3 stations across the SEIFA spectrum: predictions vs actuals over the test fold, plus a SHAP waterfall for one prediction each."),
            code("# TODO: pick low/mid/high SEIFA stations; predictions vs actuals; waterfall plots"),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing notebooks")
    args = parser.parse_args()

    targets = {
        "01_eda.ipynb": build_eda(),
        "02_modeling.ipynb": build_modeling(),
        "03_explainability.ipynb": build_explainability(),
    }

    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    for name, nb in targets.items():
        path = NOTEBOOKS_DIR / name
        if path.exists() and not args.force:
            print(f"skip {path} (exists; use --force to overwrite)")
            continue
        path.write_text(json.dumps(nb, indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
