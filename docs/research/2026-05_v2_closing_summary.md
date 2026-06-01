# v2.x closing summary — what we learned, what's pinned, what comes next

**Date:** 2026-05-31
**Span covered:** 2026-05-26 (v2.0 weather-fix planning) → 2026-05-31 (PR C close)
**Key artefacts shipped at v2.x:** PRs [#48](https://github.com/cauldnz/aus-fuel-forecaster/pull/48), [#49](https://github.com/cauldnz/aus-fuel-forecaster/pull/49), [#50](https://github.com/cauldnz/aus-fuel-forecaster/pull/50), [#51](https://github.com/cauldnz/aus-fuel-forecaster/pull/51), PR C (this branch)
**Next chapter:** v3.0 — methodology overhaul. See [`spec.md`](../../spec.md) §15.

## TL;DR

Over six days the project moved from leaky-weather v1 to a clean v2.0 baseline, exercised every meaningful adoption of the `abs-census-augmentor` v2.0.0 → v2.1.0 release surface (cross-sectional uplift in PR A, temporal-mode adoption in PR B, full split-pass + ablation in PR C), and surfaced a methodology gap that the next major version exists to close: **single-split A-vs-B evaluation isn't robust enough to distinguish real wins from fold-specific noise.** We're pinning the v2.x state and moving to v3.0 — a time-series k-fold cross-validation harness + possibly a Docker handoff to a home AMD server for the per-experiment retrain cost that pattern implies.

## The v2.x arc

### v2.0 — weather-leakage fix (landed [PR #48](https://github.com/cauldnz/aus-fuel-forecaster/pull/48))

v1's weather block (`wx_*`) joined Open-Meteo ERA5 *reanalysis actuals* on the prediction date — a leak, since real-time predictors only have the forecast for tomorrow. v2.0 replaces this with NOAA GFS day-ahead forecasts via anonymous AWS S3 byte-range subsetting (the strict-free path after Open-Meteo's free-tier rate limits proved unworkable at 4,587-station scale). Spec §13.7. Outcome doc: [`docs/research/2026-05_weather_leakage_fix_outcome.md`](2026-05_weather_leakage_fix_outcome.md).

Headline impact:
- **Absolute MAE rose 0.07–0.15 c/L** — within the predicted leakage-tax range, smaller than feared
- **test_normal Δ MAE (B vs A)**: −0.391 → −0.353 (A-vs-B comparison was indeed unbiased w.r.t. leakage)
- **test_crisis Δ MAE**: −0.183 → **−0.398** (more than doubled — SA2 lift is bigger AND more robust on OOD data once the model can't cheat with retrospective truth)

### PR A — abs-census-augmentor v1.5 → v2.0.0 + 5 new SA2 columns (landed [#50](https://github.com/cauldnz/aus-fuel-forecaster/pull/50))

v2.0.0 added 2016 SEIFA/GCP support, ERP age/sex columns (PR #82), three cross-dataset PRESETs (PR #86), and the temporal-mode plumbing we'd later adopt in PR B. PR A took the cross-sectional half — bumped the pin, renamed cache dirs, added 5 new columns to `AUGMENTOR_VARIABLES`, ran the headline experiment. Outcome: clean no-op on the existing 15-column SA2 surface (test_normal/test_crisis Δ MAE unchanged from PR B baseline). The new columns landed in `stations.parquet` as candidates for later curation; SA2_COLUMNS stayed at 15. Spec §7.7.5.

### PR B — temporal-mode adoption via split-pass architecture (landed [#51](https://github.com/cauldnz/aus-fuel-forecaster/pull/51))

PR B implemented the spec §7.7.2 deferral — moved SEIFA + ERP `population_total` to a per-(station, date) temporal augment pass. Required:
- New module `build/enrich_panel_temporal.py` against unique (station, date) panel keys
- Split `AUGMENTOR_VARIABLES` into `_CROSS_SECTIONAL` (GCP, ERP age/sex, ABS_PIA, cross-dataset PRESETs, DSS — what couldn't temporal-mode cleanly) + `_TEMPORAL` (SEIFA, ERP-total — what could)
- DSS held back from temporal pass pending upstream DSS 2022-Q4 parser fix (filed as augmentor #99, fixed in v2.1.0)

Result: **per-row temporal demographics regressed the headline** by 0.08-0.11 c/L on both folds. The hypothesis "per-row Census beats one-static-snapshot" wasn't supported on this problem. The architecture landed anyway as a no-regret platform for future column moves once upstream gaps closed.

### Upstream collaboration — 5 issues filed

PR A's research spike + PR B's adoption surfaced four real upstream bugs/limitations and one regression:

| Issue | Title | Resolution |
|---|---|---|
| [#91](https://github.com/cauldnz/abs-census-augmentor/issues/91) | GCP cross-edition NaN for 2016 release in temporal mode | Stage 1 (PR #94) loud-error guard; **Stage 2 (PR #96) full fix in v2.1.0** |
| [#92](https://github.com/cauldnz/abs-census-augmentor/issues/92) | ERP temporal-release resolution only sees latest publication | **Fully fixed (PR #95) — historical-year projection** |
| [#99](https://github.com/cauldnz/abs-census-augmentor/issues/99) | DSS XLSX parser fails on 2022-Q4 (and older) releases | **Fully fixed (PR #100) — 5-digit SA2 code translation** |
| [#101](https://github.com/cauldnz/abs-census-augmentor/issues/101) | `compute_sa2_areas_km2` crashes on null-geometry SA2s | **Fully fixed (PR #102) — null-geom guard** |
| (no number, internal note in `enrich_panel_temporal.py`) | DSS temporal-mode validator strictness — requires every column present in every release | Pending — worked around by trimming DSS_FAMILY to 9 universally-present cols |

The augmentor cut **v2.1.0** on 2026-05-31 bundling all four fixes plus the new `ERP.population_density_per_km2` column (PR #97), SEIFA 2011 / ASGS Edition 1 support (PR #89), and five more cross-dataset PRESETs (PRs #90 + #93).

### PR C — 7 experiments against v2.1.0 (pinned in main via this branch)

With every upstream blocker closed, PR C exercised the temporal/curation surface end-to-end. Single orchestrator (`tools/research/pr_c_overnight_runner.py`) ran 7 experiments (5 hours total wall-clock across two rounds), with each writing its own `features_*.parquet` + `models_*/` + `pr_c_*_comparison.md`.

| Experiment | Description | test_normal Δ MAE | vs baseline | test_crisis Δ MAE | vs baseline |
|---|---|---:|---:|---:|---:|
| **PR B baseline** | (committed main) | −0.239 | — | −0.321 | — |
| E1 — DSS temporal | Move 9 universal DSS cols to temporal | **−0.324** | **−0.085 ✅** | −0.170 | +0.151 |
| E2 — GCP temporal | Move 9 GCP-family cols to temporal | +0.470 | +0.709 ❌ | −0.255 | +0.066 |
| E3 — DSS + GCP combined | Both above | −0.041 | +0.198 | −0.320 | +0.001 |
| E4 — new ERP density + 21-col curated SA2 | density + 5 PR-A unmodeled cols added to model block | −0.088 | +0.151 | **−0.603** | **−0.282 ✅** |
| E5 — DSS temporal + curation | E1 + E4 together | +0.312 | +0.551 ❌ | −0.150 | +0.171 |
| E4a — density only | E4 ablation — just the new column | +0.127 | +0.366 | −0.550 | −0.229 |
| E4b — curation only | E4 ablation — just the PR-A candidates | +0.168 | +0.407 | +0.571 | +0.892 ❌❌ |

Full per-experiment write-up at [`results/pr_c_overnight_summary.md`](../../results/pr_c_overnight_summary.md). Per-experiment segmentation tables at [`results/pr_c_*_comparison.md`](../../results/).

## What we learned

### 1. The new `ERP.population_density_per_km2` column drives crisis-fold lift

E4a (density alone) captured ~91% of E4's test_crisis improvement (−0.550 vs −0.603) without the curation broadening. The new density column is a single-feature win on the OOD fold — bigger than anything else in PR C — but at a meaningful test_normal cost.

### 2. The 5 PR-A unmodeled candidates are harmful on their own

E4b (curation alone, no density) is the worst experiment in PR C — Model B *lost* by 0.571 c/L on test_crisis (vs baseline's win of 0.321). The candidates only "worked" in E4 because density was lifting hard enough to mask their regression. This is the **same v1.5-era lesson restated** (broadening hurt; curation by gain importance is required) — but the curation choice mattered more than the candidate identity.

### 3. Per-row temporal demographics didn't help (mostly)

PR B's SEIFA/ERP-total temporal swap regressed both folds by 0.08-0.11 c/L vs static. PR C's E1 (DSS temporal) was the one exception — it beat baseline on test_normal by 0.085 c/L (the only test_normal win in the entire v2.x arc). But the gain didn't compose: combining E1 with anything (E3 with GCP, E5 with curation) destroyed it. **Temporal-demographic signal exists but is brittle.**

### 4. Single-fold wins don't compose — the methodology gap

The strongest evidence: E5 hypothesised that adding E1's DSS-temporal change to E4's curation would land both wins simultaneously. Instead it produced the *worst test_normal regression* of any non-GCP experiment (+0.551 c/L vs baseline) and didn't even hold the crisis-fold gain. Single-split deltas are dominated by interaction effects we can't predict from individual ablations.

### 5. Crisis-fold and test_normal are different problems

Across PR B, PR C, and the v1.5-era experiments, every change has been a trade between these two folds. There is no configuration that beats the baseline on both. The augmentor surface has a real bias-variance trade — features that help OOD generalisation cost in-distribution accuracy and vice versa. **The right answer is probably not "pick the winning fold"; it's "stop relying on two folds."**

### 6. Augmentor versions are hyperparameters when they're not

Three out of four augmentor pin bumps in v2.x were no-ops on the modelled features:
- v1.5 → v2.0.0 (PR A): byte-identical predictions on the 15-col surface
- v2.0.0 → v2.0+main `65fd3fa6`: byte-identical (PR B baseline)
- v2.0+main → v2.1.0 (PR C round 1): byte-identical for E1-E4 baseline experiments

The one that wasn't (v1.4.2 → v1.5, +0.104 → −0.059 on the same 10 cols) was driven by upstream parsing fixes against the real GCP DataPack. **Re-run the headline experiment on every minor-version bump** (the v1.5 review's recommendation #1) keeps catching parsing surprises before they propagate.

### 7. Workarounds belong in shared modules

Both the cross-sectional pass (`enrich_census.py`) and the temporal pass (`enrich_panel_temporal.py`) hit the same upstream `cannot reindex on an axis with duplicate labels` collision bug. PR C round 1 had E2/E3 fail because the splitter only lived in the cross-sectional path. Extracting it to `build/_augmentor_helpers.py` and applying it in both call sites fixed it in one place. **One upstream bug, one workaround, no fingers crossed.**

## What's pinned at v2.x (committed in main)

- Pin: `abs-census-augmentor` at v2.1.0 (commit `2ea02fb8`)
- Weather: NOAA GFS via anonymous AWS S3, multi-horizon `wx_*_tN` block (spec §13.7)
- SA2 cross-sectional pass (`build.enrich_census`): GCP + 6 GCP-internal PRESETs + ERP age/sex + 4 ABS_PIA + 3 cross-dataset PRESETs + 13 DSS welfare cols → `stations.parquet`
- SA2 temporal pass (`build.enrich_panel_temporal`): 4 SEIFA + ERP `population_total` → `panel_sa2_temporal.parquet`, joined on (station_id, date) at make_features time
- Model block (`feature_blocks.SA2_COLUMNS`): 15 columns — unchanged since the v1.5 review's 31→15 curation
- Headline (Model A vs Model B, committed PR B + this PR C pin bump): test_normal Δ MAE **−0.239**, test_crisis Δ MAE **−0.321**

## Why v3.0 — methodology over features

**Every PR in v2.x added or moved features looking for the next 0.05-0.5 c/L of MAE.** The pattern that emerged: wins on one fold, losses on another, no robust both-fold improvement, no way to tell whether a 0.05 c/L delta is signal or noise from a single 2024-25 / 2026 split. The most important finding from PR C isn't any specific column — it's that **we ran out of confidence in the methodology before we ran out of features to try.**

Concretely, the cases v2.x has no good answer for:
- Is E4's test_crisis +0.282 c/L gain robust, or fold-specific?
- Would the candidates that didn't make the 15-col cut survive a different fold? (We curated by gain rank from a single 31-col fit.)
- Would temporal-mode benefits (E1) show up on different historical splits, or are they 2024-25 artefacts?
- How much of any reported Δ MAE is real and how much is what a 6-fold mean ± stdev would call within-noise?

The next major version exists to answer these — **time-series k-fold cross-validation**, possibly with a Docker handoff to a home AMD server for the per-experiment cost it implies. See [`spec.md`](../../spec.md) §15.

## See also

- [`spec.md`](../../spec.md) §13.7 (weather leakage fix), §7.7.2 (temporal-mode), §7.7.4 (block curation), §7.7.5 (PR A static surface), §15 (v3.0 plan)
- [`docs/research/2026-05_abs_census_augmentor_v2.0_review.md`](2026-05_abs_census_augmentor_v2.0_review.md) — full v2.0 review + PR B outcome
- [`docs/research/2026-05_abs_census_augmentor_v1.5_review.md`](2026-05_abs_census_augmentor_v1.5_review.md) — earlier upgrade review (the v1.4.2 → v1.5 swing)
- [`docs/research/2026-05_weather_leakage_fix_outcome.md`](2026-05_weather_leakage_fix_outcome.md) — v2.0 weather outcome
- [`docs/research/2026-05_sa2_feature_curation.md`](2026-05_sa2_feature_curation.md) — original 31 → 15 curation methodology
- [`results/pr_c_overnight_summary.md`](../../results/pr_c_overnight_summary.md) — PR C experiment table
- [`results/README.md`](../../results/README.md) — current headline numbers + iteration story
- Augmentor releases: [v2.0.0](https://github.com/cauldnz/abs-census-augmentor/releases/tag/v2.0.0), [v2.1.0](https://github.com/cauldnz/abs-census-augmentor/releases/tag/v2.1.0)
