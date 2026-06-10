"""One-off — refresh notebook 03_explainability for v3.0 framing.

Adds v3.0 callout, reframes the interaction plot as Phase 3 #3 evidence,
adds a Model A SHAP summary plot (production model) as new section 1b,
notes Model B as retired in section 4.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/03_explainability.ipynb")


def _src(text: str) -> list[str]:
    return text.splitlines(keepends=True)


CELL_0_NEW = """# 03 — Explainability

SHAP analysis per spec.md §9.3 + acceptance criterion §2 #6.

> **v3.0 update (2026-06-09):** the SHAP analysis below was originally set up to inspect Model B (with the SA2 augmentor block) — the v2.x production candidate. **Under v3.0 Model B is retired** (`spec.md` §15.6); Model A is the production model. This notebook is updated to:
>
> 1. **Show Model A's SHAP summary** as the new primary explainability artifact (production model → `results/shap/summary_a.png`).
> 2. **Keep Model B's SHAP** as the canonical "here's what the augmentor variants were splitting on" view — useful for understanding the null result.
> 3. **Reframe the `day_of_fortnight × seifa` interaction plot** as evidence supporting the Phase 3 #3 falsification: the SHAP-detected interaction is weak in Model B *and* an explicit `sa2_seifa_x_dof` feature added in Phase 3 #3 made Model B 3× worse. The two pieces of evidence triangulate: there's no useful Centrelink × SEIFA interaction for GBM to extract from this data.
>
> Closing summary: `docs/research/2026-06_v3.0_phase3_closing_summary.md`.

Reads saved Model A and Model B from `models/`. All plots write to `results/shap/`.

Run AFTER `make train` so the pickles are fresh. Notebook is feature-set-agnostic — works whether Model B has the 10-column original SA2 block or the 29-column broadened block.
"""

# Section 1 title — reframe as "Model B SHAP (historical)"
CELL_4_NEW = """## 1. SHAP summary plot — Model B (historical, with SA2 block)

The Model B summary plot — preserved for the augmentor explainability story. Each point is one row × one feature; x-axis is SHAP value (impact on prediction in cents/L); colour is the feature's value (red = high, blue = low). Features ranked by mean |SHAP| over the sample.

> **v3.0 note:** Model B is retired from production. The summary plot here shows what the augmentor variants *would* have split on if they were in production. The new section 1b shows the v3.0 production Model A's SHAP summary. The two summaries are useful side-by-side: section 1 shows the SA2 features ranked among the others (typically outside the top-30); section 1b shows the production model's actual feature ranking.

Saves to `results/shap/summary_b.png`.
"""

# Section 3 (interaction plot) — reframe as Phase 3 #3 evidence
CELL_8_NEW = """## 3. SHAP interaction plot ⭐ — falsification evidence

`cal_day_of_fortnight × sa2_seifa_irsd_score` — the chart that motivated the augmentor's central hypothesis.

> **v3.0 falsification (Phase 3 #3):** the hypothesis this chart was set up to support — "low-SEIFA SA2s have a different fortnight-cycle shape than high-SEIFA SA2s, and Model B exploits this" — was tested directly by adding an *explicit* `sa2_seifa_x_dof = sa2_seifa_irsd_score * cal_day_of_fortnight` feature to Model B's feature set. **Result: Model B got 3× worse** (Mean Δ MAE +0.670 c/L vs +0.215 baseline; fold_6 nearly doubled in harm). LightGBM actively splits on the explicit interaction feature (rank 46-58 of ~89 by gain) but the splits don't generalise across folds. Full report: `results/v3_phase3_e6_seifa_dof_interaction_headline.md`.
>
> The SHAP interaction plot below should show the same picture from the implicit-interaction side: even Model B's own implicit SHAP interaction signal is weak. The two pieces of evidence agree — there's no useful Centrelink × SEIFA interaction for GBM to extract.

Saves to `results/shap/interaction_dof_seifa.png`.
"""

# Section 4 (importances comparison) — note B retired
CELL_10_NEW = """## 4. Top-20 feature importances: Model A (production) vs Model B (retired)

Side-by-side bars. SA2 features highlighted in orange so it's visually obvious where (and whether) the augmentor block displaces non-SA2 features in Model B's top-20.

> **v3.0 framing:** **Model A is the production model** (left panel). Model B is included on the right for historical comparison — showing how the SA2 features ranked when they were modeled. Under v3.0 k-fold validation, the SA2 features didn't translate the importance ranking into robust predictive lift. Uses gain-importance (LightGBM's own), matching the table in `results/comparison.md` (v2.x) and the per-experiment reports under `results/v3_phase2_*_kfold.md` (v3.0).

Saves to `results/shap/importance_a_vs_b.png`.
"""


REPLACEMENTS: dict[int, str] = {
    0: CELL_0_NEW,
    4: CELL_4_NEW,
    8: CELL_8_NEW,
    10: CELL_10_NEW,
}


# NEW section 1b — Model A SHAP summary (production)
NEW_1B_MD = """## 1b. SHAP summary plot — Model A (production)

The v3.0 production model. Same sample as section 1; SHAP values computed against Model A's feature set (no `sa2_*`). This is the **canonical explainability artifact** for the shipped model.

Saves to `results/shap/summary_a.png`.
"""

NEW_1B_CODE = """# Model A SHAP — production model. Same X_b_sample but reduced to Model A's
# feature columns (no sa2_*). TreeExplainer is fast enough to re-run for the
# second model.
cols_a_present = [c for c in cols_a if c in X_b_sample.columns]
X_a_sample = X_b_sample[cols_a_present]

explainer_a = shap.TreeExplainer(model_a)
shap_values_a = explainer_a.shap_values(X_a_sample)
print(f"Model A SHAP values shape: {shap_values_a.shape}")
print(f"Model A expected_value: {explainer_a.expected_value:.3f}")

plt.figure(figsize=(10, 12))
shap.summary_plot(
    shap_values_a,
    X_a_sample,
    plot_type="dot",
    max_display=30,
    show=False,
)
plt.title(
    f"Model A (PRODUCTION) — SHAP feature impact (top 30 of {len(cols_a_present)}) "
    f"on a {len(test_sample):,}-row test_normal sample"
)
plt.tight_layout()
plt.savefig(config.SHAP_DIR / "summary_a.png", dpi=150, bbox_inches="tight")
plt.show()
"""


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    for idx, new_text in REPLACEMENTS.items():
        cell = cells[idx]
        if cell["cell_type"] != "markdown":
            raise RuntimeError(
                f"cell {idx} is {cell['cell_type']!r}, expected markdown"
            )
        cells[idx]["source"] = _src(new_text)
        print(f"updated cell {idx}")

    # Insert NEW Model A SHAP section AFTER existing section 1 (cells 4-5).
    # New cells go at positions 6 (markdown header for 1b) and 7 (code).
    new_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": _src(NEW_1B_MD),
    }
    new_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(NEW_1B_CODE),
    }
    cells.insert(6, new_md)
    cells.insert(7, new_code)
    print("inserted 2 new cells at 6, 7 (Model A SHAP)")
    print(f"final cell count: {len(cells)}")

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
