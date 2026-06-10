"""One-off — refresh notebook 02_modeling for v3.0 framing.

Adds v3.0 callouts at the top + Model A retirement-of-B context. Inserts a
new section after the headline showing the k-fold results from the v3.0
methodology (loads the per-fold table from
``results/v3_phase2_pr_b_baseline_kfold.md`` via direct parquet read from
the kfold model dirs if they exist; falls back to a static summary table
from the published metrics JSON).

Doesn't refit anything — preserves the spec §9 "notebooks read, don't refit"
rule. Adds new cells for v3.0 reporting; rewrites the existing v2.x
single-split sections to be "historical context" rather than the headline.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/02_modeling.ipynb")


def _src(text: str) -> list[str]:
    return text.splitlines(keepends=True)


CELL_0_NEW = """# 02 — Modeling

Reports the v3.0 ship-Model-A outcome per spec §8 + §9.2 + §15.6.

> **v3.0 update (2026-06-09):** the project's original framing ("Model B should beat Model A by adding the SA2 augmentor block") returned a null result under proper 6-fold k-fold validation. The 8 augmentor variants tested in Phase 2 produced 0 robust wins. Phase 2.5 postmortem (rank consistency + seed-noise floor + explicit interaction probe + Optuna hyperparameter sweep) confirmed: ship Model A on tuned hyperparameters. Full evidence: `docs/research/2026-06_v3.0_phase3_closing_summary.md`.
>
> This notebook leads with Model A on the tuned defaults and reports v3.0 k-fold metrics as the headline. Model B is loaded for historical comparison only (the artifacts still exist in `models/` because `make train` builds them for reproducibility of the v2.x story).

**Design note.** Spec §9 states *"none of [the notebooks] refit data or re-call APIs."* The actual fitting lives in `fuel_pred.train.train_models` (run via `make train`) and `fuel_pred.train.cv` (run via `make train-kfold`). This notebook **loads** those artifacts and presents the modeling analysis — it does not refit.

> Run `make train && make evaluate && make train-kfold && make evaluate-kfold` first if any artifacts aren't present.
"""

CELL_3_NEW = """## 1. Folds — v3.0 k-fold methodology

v3.0 replaced the v2.x single-split (train ≤ 2022, val 2023, test_normal 2024-25, test_crisis 2026) with 6-fold expanding-window k-fold CV across the full panel (spec §15.2). The v2.x single-split fold boundaries are loaded below for the historical comparison; the v3.0 k-fold geometry is documented in `results/v3_phase2_pr_b_baseline_kfold.md`.
"""

CELL_5_NEW = """## 2. Feature columns

Model A: lag, upstream, calendar, ctx, stn, wx (no `sa2_*`). **Production model.**
Model B: same plus `sa2_*`. **Retired in v3.0** — kept as a buildable artifact for reproducibility.

The `sa2_*` block is still ingested into `features.parquet` (spec §7.7 — research surface), but the production Model A does not consume it. Under v3.0 k-fold validation no augmentor variant produces a robust lift over Model A; see Phase 3 closing summary doc for the full evidence.

**Identical training rows** for both A and B — only rows where every `sa2_*` column is non-null. This makes the historical A vs B comparison apples-to-apples (preserves spec §8.4's original framing for the v2.x record).
"""

CELL_7_NEW = """## 3. Model A — production (loaded)

**Production model.** Lag, upstream, calendar, ctx, stn, wx blocks; no `sa2_*`. Hyperparameters per spec §8.2 (v3.0-tuned via Optuna TPE in Phase 3 #4; validated across 6 seeds with a mean improvement of 0.170 c/L over the original v1/v2 defaults).
"""

CELL_9_NEW = """## 4. Model B — historical / retired (loaded)

**Retired in v3.0.** Same hyperparameters and training rows as Model A; the only addition was the SA2 augmentor block. Under v3.0 6-fold k-fold validation, Model B did *not* robustly outperform Model A across 8 augmentor variants tested. The artifacts persist for reproducibility of the v2.x experimental record.

Top gain-importances side by side below — useful for understanding which SA2 features the model *would* split on if it were in production. The v3.0 verdict isn't "the model ignored these features" but "the splits don't generalise across folds" (see `docs/research/2026-06_v3.0_phase3_closing_summary.md`, §3 interaction-probe section).
"""

CELL_11_NEW = """## 5. Headline metrics

**v3.0 headline: k-fold (6 folds, mean ± stdev) — see new cell below.**

The v2.x single-split numbers (test_normal + test_crisis) are still computed below for historical continuity, but they no longer drive the ship decision — v3.0 §15.2 deprecated the single-split scheme. MAE / RMSE / MAPE / median / p90 absolute error.
"""

CELL_15_NEW = """## 7. Residual diagnostics

Residuals over time across the v2.x folds. The "crisis-period blowup" the original notebook was set up to flag is real — Model A's residuals on test_crisis (now folder 6 in v3.0) are wider — but v3.0 absorbs this into the rotating k-fold rather than treating it as a separate concept.
"""

CELL_17_NEW = """## 8. Artifact provenance

Where the loaded artifacts come from (the notebook reads, it does not write — spec §9).
"""

CELL_18_NEW = """# Artifacts are produced by the pipeline, not this notebook:
#
#   make train          -> models/model_a.pkl, models/model_b.pkl,
#                          models/feature_lists.json,
#                          models/predictions_test_{normal,crisis}.parquet
#   make evaluate       -> results/comparison.md  (v2.x single-split headline)
#   make train-kfold    -> models_kfold/fold_{1..6}/ + kfold_audit.json
#   make evaluate-kfold -> results/v3_phase1_smoke_kfold.md (or similar)
#
# v3.0 canonical artifacts under results/:
#   v3_phase2_summary.md                            -- 8-variant summary
#   v3_phase2_pr_b_baseline_kfold.md                -- PR B baseline k-fold report
#   v3_phase2_pr_c_*_kfold.md                       -- per-variant reports
#   v3_phase3_rank_consistency.md                   -- postmortem #1
#   v3_phase3_seed_noise_summary.md                 -- postmortem #2
#   v3_phase3_e6_seifa_dof_interaction_headline.md  -- postmortem #3
#   v3_phase3_hyperopt_summary.md                   -- postmortem #4 (sweep)
#   v3_phase3_hyperopt_validation.md                -- postmortem #4 (6-seed validation)
#
# Closing summary doc:
#   docs/research/2026-06_v3.0_phase3_closing_summary.md
#
print("Loaded artifacts:")
for p in sorted([
    Path("models/model_a.pkl"),
    Path("models/model_b.pkl"),
    Path("models/feature_lists.json"),
    Path("models/predictions_test_normal.parquet"),
    Path("models/predictions_test_crisis.parquet"),
]):
    if p.exists():
        size_kb = p.stat().st_size / 1024
        print(f"  {p}  ({size_kb:,.0f} KB)")
    else:
        print(f"  {p}  (missing — run `make train`)")
print()
print("v3.0 canonical headline lives at:")
print("  results/v3_phase2_pr_b_baseline_kfold.md")
print("  docs/research/2026-06_v3.0_phase3_closing_summary.md")
"""


# New cell added AFTER cell 12 (the v2.x headline metrics table) to add the
# v3.0 k-fold headline. We insert it as a markdown + code pair.
V3_HEADLINE_MARKDOWN = """## 5b. v3.0 headline — k-fold metrics (canonical)

The v3.0 canonical headline. PR B baseline (the same model config that v2.x shipped) under 6-fold time-series k-fold. Loaded from the published metrics JSON. **Mean Δ MAE is well inside the per-fold Stdev → Model B has no robust win.**
"""

V3_HEADLINE_CODE = """# Load v3.0 metrics from the published JSON.
phase2_metrics_path = Path("results/v3_phase2_metrics.json")
if not phase2_metrics_path.exists():
    print(f"missing {phase2_metrics_path} — run tools/research/v3_phase2_kfold_runner.py")
else:
    phase2 = json.loads(phase2_metrics_path.read_text(encoding="utf-8"))
    summary_rows = []
    for exp_name, m in phase2.items():
        if "mean_delta_mae" not in m:
            continue
        mean_d = m["mean_delta_mae"]
        std_d = m["stdev_delta_mae"]
        if abs(mean_d) > 2 * std_d:
            verdict = "robust" if mean_d < 0 else "robust (B loses)"
        elif abs(mean_d) > std_d:
            verdict = "weak"
        else:
            verdict = "noise"
        summary_rows.append({
            "experiment": exp_name,
            "Mean Δ MAE (B−A)": round(mean_d, 3),
            "Stdev Δ MAE": round(std_d, 3),
            "Min Δ MAE": round(m["min_delta_mae"], 3),
            "Max Δ MAE": round(m["max_delta_mae"], 3),
            "verdict": verdict,
        })
    v3_summary_df = pd.DataFrame(summary_rows).set_index("experiment")
    display(v3_summary_df)

# v3.0 Phase 3 #4 (hyperopt) — Model A retune validation
hyperopt_val_path = Path("results/v3_phase3_hyperopt_validation.json")
if hyperopt_val_path.exists():
    val = json.loads(hyperopt_val_path.read_text(encoding="utf-8"))
    print()
    print(f"Phase 3 #4 hyperopt validation (Model A new defaults vs v1/v2):")
    print(f"  Mean improvement across folds: {val['mean_improvement_across_folds']:+.4f} c/L")
    print(f"  Stdev improvement across folds: {val['stdev_improvement_across_folds']:.4f} c/L")
    ratio = abs(val['mean_improvement_across_folds']) / val['stdev_improvement_across_folds']
    print(f"  Ratio |mean|/stdev: {ratio:.2f}")
    print(f"  Verdict: {'ROBUST' if val['verdict']['robust'] else 'WEAK' if val['verdict']['validated_significance'] else 'MARGINAL'}")
"""


REPLACEMENTS: dict[int, tuple[str, str]] = {
    # cell_idx: (cell_type, new_source)
    0: ("markdown", CELL_0_NEW),
    3: ("markdown", CELL_3_NEW),
    5: ("markdown", CELL_5_NEW),
    7: ("markdown", CELL_7_NEW),
    9: ("markdown", CELL_9_NEW),
    11: ("markdown", CELL_11_NEW),
    15: ("markdown", CELL_15_NEW),
    17: ("markdown", CELL_17_NEW),
    18: ("code", CELL_18_NEW),
}


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    for idx, (expected_type, new_text) in REPLACEMENTS.items():
        cell = cells[idx]
        if cell["cell_type"] != expected_type:
            raise RuntimeError(
                f"cell {idx} is {cell['cell_type']!r}, expected {expected_type!r}"
            )
        cells[idx]["source"] = _src(new_text)
        print(f"updated cell {idx} ({expected_type})")

    # Insert new v3.0 headline cells AFTER cell 12 (v2.x headline metrics code)
    # so the new cells live as 5b right after the v2.x table renders.
    new_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": _src(V3_HEADLINE_MARKDOWN),
    }
    new_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(V3_HEADLINE_CODE),
    }
    cells.insert(13, new_md)
    cells.insert(14, new_code)
    print("inserted 2 new cells at positions 13, 14 (v3.0 headline)")
    print(f"final cell count: {len(cells)}")

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
