"""One-off — refresh notebook 01_eda for v3.0 framing.

Adds v3.0 outcome callout at the top, updates the test_crisis references
(v3.0 deprecated crisis-as-separate), and rewrites the §6 Centrelink-day
check commentary to reflect the Phase 3 #3 falsification.

Not part of the production pipeline — keep alongside other tools/research
spike scripts.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/01_eda.ipynb")


def _to_source(text: str) -> list[str]:
    """Jupyter stores source as list-of-lines. Split keeping newlines."""
    lines = text.splitlines(keepends=True)
    return lines


CELL_0_NEW = """# 01 — Exploratory Data Analysis

Reads `data/processed/features.parquet`. Per spec.md §9.1.

> **v3.0 update (2026-06-09):** the §6 Centrelink-day × SEIFA chart was originally framed as "if the lines diverge across quintiles, the augmentor has signal." Under v3.0 6-fold k-fold validation (spec §15.6 / `docs/research/2026-06_v3.0_phase3_closing_summary.md`), the augmentor surface produced no robust lift across 8 variants. Phase 3 #3 then added an *explicit* `sa2_seifa_irsd_score × cal_day_of_fortnight` feature and Model B got *3× worse*. The chart below remains useful EDA — it shows the cycle's main effect — but the interaction hypothesis it was set up to test is now falsified. §5 also references the v2.x `test_crisis` fold which v3.0 deprecates; treat the shading in §5/§6 as historical labelling rather than a current evaluation fold.

All eight sections are filled in. Sections 1, 6, 8 plus the validation extras (§9 below) were written during the Phase 7 validation pass — their job is to catch silent feature-engineering bugs *before* model interpretation. Sections 2, 3, 4, 5, 7 (geographic, price dispersion, petrol cycle, 2026 crisis, cross-correlations) were added afterwards as the descriptive-analytics layer.

**The keystone is §6** — the Centrelink-day × SEIFA chart, the augmentor story's central hypothesis (now falsified — see callout above). **§8** (missingness) confirms each feature actually has signal across the project span. The §9 extras pin down lag/cross-fuel/SA2-variance sanity.

> Note (per spec §9 line: *"none of them refit data or re-call APIs"*): every cell reads only `features.parquet`, except §2's optional coordinate map, which reads `lat`/`lon` from `data/interim/stations.parquet` (those columns are dropped from the modelling feature set) and degrades gracefully if that file is absent.
"""

CELL_14_NEW = """## 5. The 2026 crisis

Dual-axis Brent vs retail U91 (weekly means), with the v2.x `test_crisis` window shaded for historical reference. The v2.x methodology held 2026 out as a separate OOD fold; **v3.0 rotates it into a 6-fold k-fold test rotation** (spec §15.2) alongside every other year. The shading is now decorative rather than load-bearing for the evaluation story.

`crisis_events.csv` is informational-only (CLAUDE.md), so we mark the fold boundary rather than individual events.
"""

CELL_16_NEW = """## 6. Centrelink-day check ⭐

**Originally framed as the augmentor-story chart.** Average U91 price residual (vs the station's 28-day rolling mean) by `cal_day_of_fortnight`, segmented by SEIFA quintile.

> **v3.0 outcome (`docs/research/2026-06_v3.0_phase3_closing_summary.md`):** under proper k-fold validation, the augmentor surface produces no robust lift. **Phase 3 #3 explicitly added** the interaction feature this chart was set up to motivate (`sa2_seifa_irsd_score × cal_day_of_fortnight`) **and Model B got 3× worse**. So: if the lines below visibly diverge across quintiles, that visual divergence is a fortnight-cycle effect that LightGBM already captures via the `cal_day_of_fortnight` main effect — *not* a SEIFA-modulated interaction the model can exploit. The original hypothesis ("Centrelink-day price discrimination by SA2 demographics") is falsified.

The chart still has descriptive value — it shows the cycle's structure and any visual differences across SEIFA quintiles. We just no longer claim those differences indicate a useful model interaction.

Anchor for the cycle phase: NSW Centrelink fortnightly payday lands on Wednesdays in our anchored 14-day cycle.
"""

CELL_20_NEW = """**Read this chart as:** descriptive evidence of the petrol cycle's main effect by day-of-fortnight. **The lines visibly diverging** across SEIFA quintiles does *not* imply an exploitable model interaction — Phase 3 #3 tested that hypothesis directly with an explicit `seifa × dof` feature and Model B regressed by 0.45 c/L. The divergence is real in the EDA but doesn't generalise under k-fold.

*Caveat for v1 small-data validation runs: with only a few months of data and a small station roster, sample size per cell is thin and the pattern may be noisy. The full `make features` run produces 1000s of rows per cell and the pattern stabilises.*
"""

CELL_38_NEW = """**If the histograms above are flat / near-constant**, the augmentor block can't differentiate stations along that dimension — Model B's per-block contribution from `sa2_*` will be small. **If they're broadly distributed**, the block has the potential to help. SEIFA IRSD is the headline; aim for visible spread across deciles 1–10 (scores ~600–1200).

> **v3.0 finding (`spec.md` §15.6):** the histograms below show good spread across SEIFA deciles, but under k-fold validation that variance turned out *not* to translate to predictive lift over Model A. The augmentor surface adds nothing detectable on top of the lag-rich feature set. Model B (with `sa2_*`) is retired from production; **Model A ships**.

*Validation pass complete.* Modeling proceeds in `02_modeling.ipynb` — refreshed for v3.0 (Model A only on tuned hyperparameters; k-fold methodology).
"""


REPLACEMENTS: dict[int, str] = {
    0: CELL_0_NEW,
    14: CELL_14_NEW,
    16: CELL_16_NEW,
    20: CELL_20_NEW,
    38: CELL_38_NEW,
}


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    for idx, new_text in REPLACEMENTS.items():
        cell = cells[idx]
        if cell["cell_type"] != "markdown":
            raise RuntimeError(
                f"cell {idx} is {cell['cell_type']!r}, expected markdown — "
                f"notebook structure may have drifted"
            )
        cells[idx]["source"] = _to_source(new_text)
        print(f"updated cell {idx}")

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
