# v6 — the augmentor surface is ~4-dimensional: a mechanistic explanation for the fuel null

**Date:** 2026-06-11
**Branch:** `claude/v6-augmentor-pca`
**Status:** complete — characterises the augmentor data's internal structure (independent of the fuel problem)
**See also:** [`2026-06_v3.0_phase3_closing_summary.md`](2026-06_v3.0_phase3_closing_summary.md), [`docs/methodology_writeup.md`](../methodology_writeup.md)

## Why this analysis

Every prior experiment asked *"does the augmentor help predict fuel prices?"* — and the answer was a robust no (0 of 8 variants × 6 folds × 6 seeds, an explicit-interaction probe, a hyperparameter sweep, and a second forecast horizon all null). We concluded the augmentor is "redundant with what the lag features already capture," but we never quantified a second, complementary form of redundancy: **how internally redundant is the augmentor surface itself?**

This is a property of the augmentor *data*, independent of the fuel problem. If any one SA2 column is highly predictable from the others, then the surface isn't N independent signals — it's a handful of latent socioeconomic gradients wearing N hats. That would *mechanistically explain* the fuel null: a few broad, slow-moving cross-sectional gradients are exactly what a station's own price history already encodes implicitly (who shops there shapes how its prices behave), so the augmentor adds nothing the lags don't already carry.

## Method

Self-prediction + PCA on the **full 37-numeric-column** augmentor surface from `data/interim/stations.parquet` (the complete materialized request — 2.5× the curated 15-col model block). Three analyses:

1. **PCA / SVD** on the standardised matrix → how many components explain 80/90/95/99% of variance (the intrinsic dimensionality).
2. **Leave-one-column-out predictability** → for each column, 5-fold CV R² predicting it from the other 36 (Ridge linear + gradient-boosting nonlinear).
3. **Correlation structure + PC loadings** → names the latent axes.

**The one gotcha (handled):** SA2 columns are broadcast across ~15M station-days. Self-prediction on the raw panel would be trivially inflated by that duplication. The correct unit is the **unique SA2 profile** — one row per `sa2_code`. 580 of 589 NSW SA2s are complete across all 37 columns (≤1.5% null anywhere), so we simply drop-null.

Reproduce: `uv run python tools/research/v6_augmentor_pca.py` (seconds, no training/network).

## Result 1 — the surface is radically low-rank

37 columns collapse to a handful of effective dimensions:

| Variance explained | Components (of 37) |
|---|---:|
| 80% | **4** |
| 90% | 6 |
| 95% | 10 |
| 99% | 20 |

The first **two** principal components alone carry **64%** of all variation across 37 columns (PC1 40%, PC2 24%); the first three carry 77%. A 37-column "feature surface" is, in information terms, about a 4-dimensional one.

## Result 2 — almost every column is reconstructable from the others

Leave-one-column-out CV R² (predict each column from the other 36):

- **Mean R² = 0.92** (GBM), median **0.94**; Ridge mean 0.94
- **All 37 columns are majority-predictable** (R² > 0.5)
- **26 of 37 are near-perfectly reconstructable** (R² > 0.9)

This is *more* redundant than the curated 15-col block (which scored mean 0.88), exactly as expected — curation had already dropped 16 low-gain columns, so the full surface has more correlated siblings. You can rebuild almost any augmentor column from the rest.

Correlation confirms it: mean max-|r| to any other column = **0.885**; **35 of 37** columns have a >0.7 correlated partner, **21 of 37** have a >0.9 partner (e.g. the two carer-payment columns; median-age vs aged-65-plus; the four SEIFA indices).

## Result 3 — the latent axes name themselves

Top principal-component loadings:

| PC | Var | What it is |
|---|---:|---|
| **PC1** | 40% | **Welfare caseload / disadvantage volume** — DSS jobseeker, disability, family-tax-benefit, youth-allowance, rent-assistance counts all load together. |
| **PC2** | 24% | **Population scale × age structure** — income-earners (−), total population (−), median age (+), aged-65+ (+): "small-and-old vs large-and-young". |
| **PC3** | 13% | **Retiree concentration** — seniors health card, ERP 65+, age pension, median age of earners. |
| **PC4** | 7% | **Urban density / car-reliance** — motor-vehicles-per-dwelling (−), density (+), renters (+). |

Four nameable gradients account for 84% of the variance. The first two — a disadvantage axis and a size/age axis — carry nearly two-thirds on their own.

## What this adds to the augmentor verdict

This is the **mechanistic complement** to the fuel-prediction null. The fuel experiments showed the augmentor doesn't help; this shows *why it was always unlikely to*:

The augmentor's 37 columns encode only **~4 real degrees of freedom** — broad, slow-moving, cross-sectional socioeconomic gradients (disadvantage, size/age, retiree-concentration, density). A station's own price history already reflects its local socioeconomic context implicitly, so the augmentor's few genuine axes are largely redundant with what the lag features already carry. Low intrinsic rank + redundancy-with-lags ⇒ the null.

### Honesty caveats

- **This establishes the "low-rank" half, not the "already encoded by lags" half.** A feature can be low-rank and still useful if its few dimensions are otherwise unavailable to the model. The "lags already encode it" inference rests on the fuel experiments, not on this analysis. This is corroborating *mechanism*, not a second independent proof of the null.
- **Part of PC1's dominance is mechanical.** The DSS columns are recipient *counts*, which scale with SA2 population, so some of their mutual predictability is a "size" artifact rather than deep socioeconomic structure. The dimensionality collapse is dramatic regardless, but PC1 should be read as "welfare *volume*" (partly size-driven), not purely "disadvantage intensity".
- **Cross-sectional only.** This characterises the static per-SA2 surface. The DSS temporal columns add quarterly variation that this unique-SA2 view collapses; whether that time-variation adds independent rank is a separate (low-priority) question.

## Implication for feature engineering

If the augmentor were ever to be used, this says **don't feed it 15-37 raw, mutually-redundant columns** — feed it its first ~3-4 principal components (or a hand-picked representative per axis: one SEIFA score, one age measure, one welfare measure, one density measure). That would deliver essentially all the augmentor's information at a fraction of the model-capacity cost — and would have avoided the overfitting that the broadened 31-col v1.x experiment ran into. (It still wouldn't beat Model A on *this* problem — the fuel null stands — but it's the right shape for any future task where demographic context genuinely has no implicit proxy.)

## What's pinned

- `tools/research/v6_augmentor_pca.py` — the analysis
- `results/v6_augmentor_pca_summary.md` + `.json` — full per-column R², variance ratios, loadings

## Sources

- `data/interim/stations.parquet` — full materialized augmentor surface (37 numeric SA2 columns, 589 NSW SA2s)
- [`2026-06_v3.0_phase3_closing_summary.md`](2026-06_v3.0_phase3_closing_summary.md) — the fuel-utility null this explains
