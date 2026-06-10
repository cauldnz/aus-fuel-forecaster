# Results — Model A vs Model B (v3.0, k-fold revalidated)

**Question (spec §1):** does augmenting per-station features with SA2-level ABS
Census demographics — via [`abs-census-augmentor`](https://github.com/cauldnz/abs-census-augmentor) —
measurably improve next-day NSW fuel-price prediction?

**Answer: no, and that is the headline.** Under proper time-series 6-fold k-fold
cross-validation (spec §15.2), the v2.x augmentor surface produces **no robust
lift** on top of Model A's lag-rich feature set. Of 8 augmentor variants tested,
0 cleared the methodology's significance bar; the best variant came in at Mean
Δ MAE −0.036 c/L (Stdev 0.396) — basically flat. The v2.x single-split headline
of "Model B beats A by 0.24-0.32 c/L" was a fold-specific artifact that reverses
sign under rotating-window CV.

The contribution becomes a **methodology study**: how single-split misleads,
how k-fold + seed-noise floor + explicit-interaction probes triangulate, and
what the v2.x experimental record looks like when subjected to honest variance
quantification. The full evidence trail and ship decision live in
[`docs/research/2026-06_v3.0_phase3_closing_summary.md`](../docs/research/2026-06_v3.0_phase3_closing_summary.md);
the externally-facing write-up (including the v4/v5 follow-ups) is at
[`docs/methodology_writeup.md`](../docs/methodology_writeup.md).

**The null holds at a second horizon (v5, 2026-06-11).** Re-running the A-vs-B
comparison at the **7-day** target (`y_t1_t7`) gives Mean Δ MAE +0.041, Stdev
0.370 — the same noise-band verdict as t+1. The 7-day target is genuinely harder
(per-fold MAE 8-25 c/L vs 4-13 at t+1; lag features weaker), so the gap the
augmentor could fill really existed — it just didn't fill it. A two-horizon
null closes the "maybe the horizon was wrong" objection. See
[`docs/research/2026-06_v5_7day_horizon_outcome.md`](../docs/research/2026-06_v5_7day_horizon_outcome.md).

**Production model: Model A on v3.0-tuned hyperparameters.** No SA2 block.
Spec §8.2's hyperparameters were re-tuned via Optuna TPE (Phase 3 #4) and
validated across 6 seeds — mean improvement 0.170 c/L over the original
v1/v2 defaults (WEAK WIN, |mean|/stdev ratio 1.29). The new defaults are
smaller, more-regularized trees with no row bagging (see spec §8.2 for the
full table). The v1/v2 defaults were over-fitting.

**Stronger validation post-retune (2026-06-10):** when the v2.x single-split
A/B comparison is re-run with the new tuned defaults, **Model A now beats
Model B on both v2.x test folds** (test_normal Δ MAE +0.236 c/L, test_crisis
Δ MAE +0.316 c/L — both Model B *worse* — vs the v2.x committed numbers of
−0.239 and −0.321 where Model B won). The augmentor's apparent "win" in v2.x
was partly an artefact of the under-tuned baseline; once Model A is properly
regularized, the SA2 block adds nothing even on the original v2.x folds.
This is exactly the failure mode the v3.0 methodology was designed to catch.

---

## Headline (v3.0, 6-fold time-series k-fold, PR B baseline)

All metrics in cents/L. **Negative Δ MAE = Model B beats Model A.**

| Fold | Test window | n | MAE A | MAE B | Δ MAE |
|------|-------------|--:|------:|------:|------:|
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.219 | 6.293 | **+0.074** |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 8.799 | 8.664 | −0.135 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.331 | 13.591 | **+0.260** |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 6.954 | 6.855 | −0.098 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.181 | 4.327 | **+0.147** |
| fold_6 | 2025-05-01 → 2026-04-30 | 426,819 | 9.573 | 10.615 | **+1.042** |
| **Mean** | — | 2,647,445 | 8.176 | 8.391 | **+0.215** |
| Stdev | — | — | 2.893 | 3.037 | 0.394 |
| Min | — | — | 4.181 | 4.327 | −0.135 |
| Max | — | — | 13.331 | 13.591 | +1.042 |

Significance read: |Mean Δ| (0.215) < Stdev Δ (0.394) → **noise band**. The
augmentor's apparent "win" in any single fold is dwarfed by the across-fold
variance. The v2.x single-split picked two favourable folds (test_normal +
test_crisis, both 2024-25 / 2026-Q1, here folds 5 and part of 6) — every other
fold tells a different story.

Full per-fold report: [`v3_phase2_pr_b_baseline_kfold.md`](v3_phase2_pr_b_baseline_kfold.md).

### What the other 7 v2.x variants do under k-fold

Full table: [`v3_phase2_summary.md`](v3_phase2_summary.md).

| Experiment | SA2 cols | Mean Δ MAE | Stdev | Verdict |
|---|---:|---:|---:|---|
| pr_b_baseline (PR B as committed) | 15 | +0.215 | 0.394 | noise |
| pr_c_e1_dss_temporal | 15 | +0.218 | 0.734 | noise |
| pr_c_e2_gcp_temporal | 15 | +0.155 | 0.196 | noise |
| pr_c_e3_combined_temporal | 15 | +0.281 | 0.653 | noise |
| **pr_c_e4_density_plus_curation** | 15+6 | **−0.036** | 0.396 | **noise (best)** |
| pr_c_e4a_density_only | 15+1 | +0.506 | 0.492 | weak (B loses) |
| pr_c_e4b_curation_only | 15+5 | +0.377 | 0.286 | weak (B loses) |
| pr_c_e5_dss_temporal_plus_curation | 15+6 | +0.051 | 0.322 | noise |

0 robust wins. 6 of 8 variants have Model B *worse* than Model A on the mean.

---

## Why null — the Phase 3 postmortem evidence

Four follow-up experiments tested the three readings of the Phase 2 outcome
(genuinely flat / methodology too strict / wrong features). Full discussion:
[`docs/research/2026-06_v3.0_phase3_closing_summary.md`](../docs/research/2026-06_v3.0_phase3_closing_summary.md).

1. **Per-fold rank consistency** ([`v3_phase3_rank_consistency.md`](v3_phase3_rank_consistency.md))
   Mean pairwise Spearman ρ across the 8 Phase 2 experiments = +0.198 (cluster
   pattern; fold_6 alone = 61% of cross-experiment variance). No fold has
   unanimous sign across all 8 variants.

2. **Seed-noise floor** ([`v3_phase3_seed_noise_summary.md`](v3_phase3_seed_noise_summary.md))
   6× Model A runs with different LightGBM seeds across all 6 folds.
   Across-pairs Δ-stdev = **0.136 c/L**; published Δ-stdev = 0.394 c/L; ratio
   2.89. Folds 3 and 6 have 3-5× higher seed-stdev than other folds — most of
   the cross-experiment "noise" is intrinsic LightGBM training-instability on
   those folds, not augmentor behaviour. **Reading A confirmed.**

3. **Explicit SEIFA × day-of-fortnight interaction**
   ([`v3_phase3_e6_seifa_dof_interaction_headline.md`](v3_phase3_e6_seifa_dof_interaction_headline.md))
   Adding `sa2_seifa_x_dof = sa2_seifa_irsd_score * cal_day_of_fortnight` to
   Model B made things **3× worse** (Mean Δ MAE +0.670 c/L vs +0.215 baseline;
   nearly doubled the augmentor's harm on fold_6 from +1.04 to +2.00). The
   model splits on the new column (gain rank 46-58 of ~89 features) but the
   splits don't generalise. **Reading C2 (missing interaction feature)
   falsified.**

4. **Hyperparameter sweep on Model A** — **WEAK WIN, new defaults locked**
   ([`v3_phase3_hyperopt_summary.md`](v3_phase3_hyperopt_summary.md) +
   [`v3_phase3_hyperopt_validation.md`](v3_phase3_hyperopt_validation.md))
   Optuna TPE Bayesian search → trial 15 winner (num_leaves=31, min_data=544,
   lr=0.028, ff=0.85, bf=0.69/freq=0, l1=0.059). 6-seed validation: mean
   improvement **0.170 c/L** across folds, stdev **0.132**, ratio 1.29 →
   WEAK WIN (>1.0). 5 of 6 folds improve clearly; fold_2 dead-flat; biggest
   wins on fold_3 (−0.38), fold_4 (−0.23), fold_1 (−0.21), fold_6 (−0.18).
   Reading C1 partially confirmed — v1/v2 defaults were over-fitting; new
   smaller-trees + more-regularization config unlocks ~0.17 c/L. New
   defaults committed to spec §8.2 + `src/fuel_pred/config.py`.

---

## What we keep, what we retire

**Keep — production path:**

- **Model A** (lag, upstream, calendar, ctx, stn, wx blocks; no SA2). 73-ish
  feature columns including a 5-column GFS weather block.
- **Spec §8.2 LightGBM hyperparameters (v3.0 tuned)** — Optuna-tuned via Phase
  3 #4: num_leaves 31, min_data_in_leaf 544, learning_rate 0.028,
  feature_fraction 0.85, bagging_fraction 0.69, bagging_freq 0 (no row
  bagging), lambda_l1 0.059. Validated across 6 seeds: 0.170 c/L improvement
  over the original v1/v2 defaults.
- **6-fold k-fold methodology** (spec §15.2) as the evaluation harness. Single-
  split A/B is deprecated.
- **`evaluate.compare_kfold`** as the canonical comparison entry point.

**Retire — research artefacts kept for reproducibility:**

- **Model B + Model B'.** Still buildable in code; not built or evaluated in
  the production pipeline.
- **`abs-census-augmentor` dependency.** Still in `pyproject.toml` because
  `build.enrich_census` populates the `sa2_*` columns in `features.parquet` for
  the research surface, but no production code consumes those columns.
- **`results/comparison.md`** (the v2.x single-split headline). Preserved as
  historical record; superseded by the per-fold reports above + the Phase 3
  closing.

**Future augmentor work:** must first reproduce a Phase 2-style improvement
that survives the v3.0 6-fold methodology before going into production. The
v2.x variants didn't; the explicit-interaction probe actively regressed. The
augmentor surface is open for further research, but the ship bar is now
clearly defined.

---

## How we got here — the iteration story (v1 → v2 → v3)

The v3.0 closing doesn't replace the v2.x search history — it adds the
methodology layer that revealed it as fold-specific. The full iteration
table:

| Iteration | Methodology | Augmentor | Weather | SA2 cols | Headline | What we learned |
|-----------|-------------|-----------|---------|---------:|----------|-----------------|
| v1.0 | single-split | v1.4.2 | ERA5 (leaky) | 10 | test_normal Δ +0.104 | First real run; SA2 hurt the headline fold |
| v1.1 | single-split | v1.5 | ERA5 | 10 | test_normal Δ −0.059 | Augmentor's improved parsing was itself a win |
| v1.2 | single-split | v1.5 | ERA5 | 31 | test_normal Δ −0.025 | 21-col broadening overfit (better val, worse test) |
| v1.3 | single-split | v1.5 | ERA5 | 15 | test_normal Δ −0.391 | Curating to 10 + top-5 by gain importance |
| v2.0 weather fix | single-split | v1.5 | NOAA GFS | 15 | test_normal Δ −0.353; test_crisis Δ −0.398 | Leakage-corrected; crisis fold lift doubled |
| v2.0 augmentor bump | single-split | v2.0 | NOAA GFS | 15 | byte-identical to v1.5 row | Pin bump validated as no-op |
| v2.x PR B (temporal SA2) | single-split | v2.0+main | NOAA GFS | 15 | test_normal Δ −0.239; test_crisis Δ −0.321 | Final v2.x committed headline |
| **v3.0 Phase 2 (rotating CV)** | **6-fold k-fold** | v2.0+main | NOAA GFS | 15 (+ 7 variants) | **mean +0.215, stdev 0.394 → noise** | **Reverses sign vs v2.x single-split** |
| **v3.0 Phase 2.5 (postmortem)** | k-fold + seed + interaction | — | — | — | **Reading A confirmed, C2 falsified** | **Ship Model A** |

The single-split → k-fold methodology shift is the line that flipped the
answer. The two folds v2.x reported (test_normal = 2024-25, test_crisis =
2026-Q1) happened to be favourable; folds 1, 3, 6 (2020-21, 2022-23, 2025-26)
tell a very different story and dominate the cross-fold mean.

---

## The original spec §1 question, re-answered

> _Does augmenting per-station features with SA2-level demographics measurably
> improve next-day NSW fuel-price prediction?_

**On this model class (LightGBM with 73 lag-rich features), no.** Across 8
augmentor configurations × 6 folds × 6 random seeds × 1 explicit interaction
feature, the augmentor surface adds nothing detectable beyond the LightGBM
seed-noise floor (~0.09 c/L per-fold stdev; ~0.14 c/L across-pairs Δ-stdev).

**Why?** Best guess: the lag features already encode per-station demographic
behaviour implicitly. Each station's price history reflects who shops there;
adding aggregate SA2-level statistics on top is redundant. A different model
class (FT-Transformer, SAINT, GAM) might extract additional signal — but that
escalation isn't justified by anything in the v3.0 evidence. The GBM is at
capacity for this feature set.

**What this means for the methodology demo.** Both directions of the original
question — "demographics help" and "demographics don't help" — are
publishable findings. The v3.0 evidence trail is the contribution; the answer
is no.

---

## Reproduction

Full pipeline from a clean checkout (needs network for raw fetches, or a
pre-populated `data/raw/`):

```bash
make all          # fetch → clean → enrich → features → train (k-fold) → evaluate
```

Or the train + evaluate stages against an existing `data/processed/features.parquet`:

```bash
make train-kfold      # fits Model A across 6 folds (B and B' available via env override)
make evaluate-kfold   # → results/v3_phase2_pr_b_baseline_kfold.md (the canonical headline)
```

Spec §15.2 documents the fold geometry. Spec §8.2 documents the LightGBM
hyperparameters. Spec §15.6 documents the ship-Model-A decision.

---

## Artifact index

### v3.0 (current — canonical)

| File | Contents |
|------|----------|
| [`v3_phase2_summary.md`](v3_phase2_summary.md) | Cross-experiment k-fold summary (all 8 v2.x variants) |
| [`v3_phase2_pr_b_baseline_kfold.md`](v3_phase2_pr_b_baseline_kfold.md) | PR B baseline full per-fold report (the headline above) |
| [`v3_phase2_pr_c_*_kfold.md`](.) | Per-variant k-fold reports for the other 7 experiments |
| [`v3_phase2_metrics.json`](v3_phase2_metrics.json) | Raw metrics dump for the 8 experiments |
| [`v3_phase3_rank_consistency.md`](v3_phase3_rank_consistency.md) | Postmortem #1 — per-fold rank consistency |
| [`v3_phase3_seed_noise_summary.md`](v3_phase3_seed_noise_summary.md) | Postmortem #2 — 6× seed-noise floor |
| [`v3_phase3_e6_seifa_dof_interaction_headline.md`](v3_phase3_e6_seifa_dof_interaction_headline.md) | Postmortem #3 — explicit interaction (falsified Reading C2) |
| [`v3_phase3_hyperopt_summary.md`](v3_phase3_hyperopt_summary.md) | Postmortem #4 — Optuna hyperparameter sweep |
| [`v3_phase1_smoke_kfold.md`](v3_phase1_smoke_kfold.md) | Initial k-fold smoke milestone (Phase 1) |
| [`../docs/research/2026-06_v3.0_phase3_closing_summary.md`](../docs/research/2026-06_v3.0_phase3_closing_summary.md) | **Phase 3 closing summary — start here for the full v3.0 story** |

### v2.x (historical — preserved for reproducibility)

| File | Contents |
|------|----------|
| [`comparison.md`](comparison.md) | v2.x committed single-split headline (test_normal + test_crisis) |
| [`pr_c_overnight_summary.md`](pr_c_overnight_summary.md) | PR C 4-experiment overnight run summary |
| [`pr_c_*_comparison.md`](.) | Per-experiment v2.x single-split comparison reports |
| [`shap/`](shap/) | v1 SHAP plots (not refreshed against v3.0 — kept for historical reference) |

---

## Limitations & future work

- **Augmentor surface tested in depth, not exhausted.** The v3.0 evidence is for
  *this model class on this feature set*. A different model class (NN with
  explicit feature interactions; GAM with SEIFA × cycle terms) might extract
  signal GBM can't. Not justified by current evidence; on the backlog as a v4.0
  research direction.
- **Hyperparameter sweep is on Model A only.** Phase 3 #4 didn't re-sweep Model
  B's hyperparams under the new search space. Argument against doing so: the
  explicit-interaction experiment already showed adding a feature LightGBM
  could use (and did use, mid-pack by gain) doesn't generalise. Capacity isn't
  the constraint.
- **Single horizon.** v3.0 maintains the `y_t1` single-day target. The 7-day
  forecast horizon (spec §13.8) remains backlogged; Model B's relative value
  could differ at longer horizons (the lag advantage decays, the demographic
  context stays static).
- **Weather block.** Still NOAA GFS day-ahead with v2.0's compromises (`weather_code`
  null-stubbed, ~20% of training rows missing wx_* from archive gaps, UTC day
  boundary). Documented in
  [`docs/research/2026-05_weather_leakage_fix_outcome.md`](../docs/research/2026-05_weather_leakage_fix_outcome.md).
  These are stable across v2/v3 and don't bias the A/B comparison either way.
- **NSW only.** Spec §3 in-scope; the v3.0 ship-Model-A decision is for NSW
  fuel prices. Other-state generalisation hasn't been tested.
