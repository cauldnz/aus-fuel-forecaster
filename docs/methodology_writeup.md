# When your feature ablation says the augmentor helps and k-fold says no — a methodology story

**Draft, 2026-06-10. External-facing.** Distills the v3.0 outcome of the
[`aus-fuel-forecaster`](https://github.com/cauldnz/aus-fuel-forecaster)
project — a small-but-honest case study in how single-split feature
ablations can mislead, and how a focused four-step k-fold + seed-noise
+ explicit-interaction + hyperparameter protocol disambiguates.

## TL;DR

I spent ~6 weeks building a methodology demo around the hypothesis that
augmenting per-station NSW fuel-price prediction features with
[`abs-census-augmentor`](https://github.com/cauldnz/abs-census-augmentor)'s
SA2-level demographic data would measurably improve next-day prediction
accuracy.

The v1/v2 single-split methodology said yes: Model B (with the SA2
augmentor block) beat Model A (without) by 0.24 c/L on the `test_normal`
fold and 0.32 c/L on `test_crisis`. We iterated on 7 augmentor-feature
variants on top of that baseline, ranked them by single-split lift, and
were ready to ship.

Then I added time-series k-fold cross-validation and the answer flipped.
**Across 8 augmentor variants × 6 rotating test windows, zero produced a
robust win.** Six of the eight had Model B *worse* on average. The PR B
"committed baseline" went from −0.24 c/L (B beats A) to **+0.22 c/L
(B worse than A by 0.22 c/L)** — the sign reversed.

This post is about how I went from "the augmentor helps" to "the
augmentor is dead" in five experiments, and the small generalisable
methodology that fell out.

## The original setup

Standard tabular forecasting recipe:

- **Target.** Tomorrow's NSW retail U91 fuel price per `(station_id, fuel_code)`.
  Daily granularity; ~14M station-day rows.
- **Model.** LightGBM regression with L1 loss (MAE-aligned).
- **Two feature sets.**
  - **Model A** = lag features + Brent/upstream prices + calendar + macro context + per-station static + weather (73 columns).
  - **Model B** = Model A + 15 `sa2_*` demographic columns (SEIFA scores, ABS Census ratios, ERP, DSS welfare-payment recipient counts).
- **Identical training rows + identical hyperparameters** between A and B
  — the only difference is the SA2 block.
- **v1/v2 validation:** time-based 4-fold split — train ≤ 2022, val 2023,
  `test_normal` 2024-25, `test_crisis` 2026-Q1. The crisis fold is a
  separate OOD holdout for the 2026 price-spike period.

The single-split answer was clear: Model B won on both test folds, with
the crisis-fold lift being the headline (−0.32 c/L). The augmentor was
adding the kind of robust per-row demographic context that helped most
on out-of-distribution data. The story was: ship Model B.

## Why I added k-fold

Three signals made me uncomfortable with the single-split conclusion:

1. **PR C explored 7 augmentor-feature variants.** None of them beat the
   PR B baseline on *both* `test_normal` and `test_crisis` simultaneously.
   The strongest single-fold result cost meaningfully on the other.
   The decision of "which variant to ship" couldn't be made under the
   methodology.
2. **The DSS dataset I'd added in v1.5 was sampled from a single quarter
   snapshot held constant across the panel.** Per-row temporal resolution
   (the augmentor's v1.5 capability) was the obvious next step — but PR B
   *regressed* the headline by 0.11 c/L when I added it. The architectural
   change was correct, but the per-row variation in 2016 vs 2021 SEIFA
   and 2017-2024 ERP introduced noise the model couldn't recoup. Hint #1:
   the headline numbers were fragile.
3. **Two of seven PR C variants flipped sign** between `test_normal` and
   `test_crisis`. Variants that helped on one fold often hurt on the
   other. Hint #2: the per-fold pattern wasn't structural.

The leap from "fragile single-split" to "let's measure how fragile" is
where k-fold came in.

## v3.0 — k-fold + a small significance heuristic

Time-series k-fold isn't fancy: 6 expanding-window folds, each with a
12-month test window. Fold 1 trains on 2017-2019, tests 2020-2021;
fold 6 trains on 2017-2024, tests 2025-2026. A 1-day gap between train
end and test start prevents the `y_t1` target-shift leak (which the
v2.x single-split silently had).

The reporting change is the key methodological move: replace **two
numbers per experiment** (`test_normal` MAE, `test_crisis` MAE) with
**six numbers per experiment** (per-fold MAE) and an aggregate (mean ±
stdev).

The significance heuristic I committed to ex-ante:

| Result | Interpretation |
|---|---|
| `|Mean Δ| > 2 × Stdev Δ` | **Robust** — the effect is real |
| `|Mean Δ| > Stdev Δ` | **Weak** — direction-consistent but small |
| Otherwise | **Noise** — within fold-to-fold variability |

I deliberately *didn't* use paired t-tests on the k-fold scores. Dietterich
(1998) and Nadeau & Bengio (2003) both showed that naive paired t-tests on
k-fold scores have inflated Type I error because the per-fold scores share
training data. The corrected Nadeau-Bengio test exists but is *stricter*
than the simple ratio heuristic — applying it would only tighten verdicts,
not loosen them.

## The Phase 2 outcome — null result, eight times

Re-ran all 8 v2.x augmentor variants (PR B baseline + 7 PR C
experiments) under the new k-fold methodology. The summary:

| Experiment | SA2 cols | Mean Δ MAE | Stdev | Verdict |
|---|---:|---:|---:|---|
| pr_b_baseline (the v2.x committed config) | 15 | **+0.215** | 0.394 | noise |
| pr_c_e1_dss_temporal | 15 | +0.218 | 0.734 | noise |
| pr_c_e2_gcp_temporal | 15 | +0.155 | 0.196 | noise |
| pr_c_e3_combined_temporal | 15 | +0.281 | 0.653 | noise |
| **pr_c_e4_density_plus_curation** | 21 | **−0.036** | 0.396 | **noise (best)** |
| pr_c_e4a_density_only | 16 | +0.506 | 0.492 | weak (B loses) |
| pr_c_e4b_curation_only | 20 | +0.377 | 0.286 | weak (B loses) |
| pr_c_e5_dss_temporal_plus_curation | 21 | +0.051 | 0.322 | noise |

**Negative Δ MAE = Model B beats Model A.**

- **0 of 8 produce a robust win.**
- **6 of 8 have Model B *worse* than A on the mean.**
- The best variant (E4) is at Δ = −0.036 c/L — basically zero, well
  inside the noise band.
- The PR B baseline's sign reversed: v2.x reported B beats A by 0.24 c/L
  on `test_normal` and 0.32 c/L on `test_crisis`; v3.0 reports B *loses*
  to A by 0.22 c/L on the mean across 6 folds. The two folds v2.x
  reported (2024-25 + 2026-Q1) just happened to be favourable; folds
  1, 3, 6 (2020-21, 2022-23, 2025-26) tell a very different story.

This outcome doesn't tell me *why* the augmentor is null. Three
plausible readings:

- **Reading A — the augmentor surface is genuinely flat.** Lag features
  already encode per-station demographic behaviour implicitly via price
  history; SA2 aggregates add nothing.
- **Reading B — the methodology is too strict.** Maybe the noise heuristic
  is over-conservative; under a corrected paired test, some of the
  variants would clear.
- **Reading C — the model class or feature engineering is wrong.** GBM
  with fixed hyperparameters might be the wrong capacity; an explicit
  Centrelink × SEIFA interaction feature might unlock signal trees aren't
  finding on their own.

I ran four follow-up experiments to triangulate.

## Phase 2.5 — four experiments to triangulate

### #1 — Per-fold rank consistency

For each pair of the 8 experiments, compute the Spearman rank
correlation of their per-fold Δ MAE patterns. If all experiments rank
the folds the same way (high ρ), there's shared per-period structure
they're all failing on. If the rankings shuffle randomly (ρ ~ 0), each
experiment's per-fold pattern is a different draw from the same noise
distribution.

**Result:** mean pairwise ρ = +0.198 (median +0.371). Cluster pattern —
7 experiments cluster at moderate positive ρ; 1 experiment
(E4 — density + curation) actively anti-correlates with the rest
(peer-ρ −0.43) because it's the only variant that helps on fold_6 (the
2026 spike).

**Fold_6 alone accounts for 61% of the cross-experiment Δ MAE variance.**
The 2026 spike dominates everything. **Zero folds have unanimous sign
across all 8 experiments.**

Partial Reading-A support (no unanimous folds); cluster pattern keeps
Reading C alive (the 7-experiment cluster *does* share a per-fold
ranking, suggesting a missing structural feature).

### #2 — Seed-noise floor

If Reading A holds, the augmentor's "noise band" should be no larger
than what two same-model different-seed runs of Model A disagree by.
Measure it directly: train Model A 6 times across all 6 folds with
different LightGBM `random_state` values, then compute the per-fold
MAE Δ between every pair of seeds.

**Result.** Mean per-fold seed-stdev of MAE_A: **0.089 c/L**. Across
pairs of seeds, the stdev of per-fold MAE differences (the analogue of
the published Δ MAE stdev): **0.136 c/L**. The published Δ MAE stdev
for PR B (B vs A) was 0.394 c/L. **Ratio: 2.89×.**

The augmentor's "noise" is about 3× the LightGBM seed-noise floor — a
non-trivial multiple but not 5× or 10×. Crucially:

| Fold | Mean MAE_A | Seed stdev | Seed range |
|---|---:|---:|---:|
| fold_1 | 5.55 | 0.054 | 0.15 |
| fold_2 | 4.76 | 0.034 | 0.10 |
| **fold_3** | 6.47 | **0.163** | **0.48** |
| fold_4 | 5.23 | 0.033 | 0.09 |
| fold_5 | 4.43 | 0.069 | 0.19 |
| **fold_6** | 3.73 | **0.178** | **0.55** |

**Folds 3 and 6 have 3-5× higher seed-stdev than other folds.** These
are exactly the folds that dominate cross-experiment Δ MAE variance.
Most of what looks like "the augmentor is unstable on those folds"
is **LightGBM being intrinsically training-unstable on those folds with
zero data change**.

Reading A confirmed. Reading C still in play.

### #3 — Explicit SEIFA × day-of-fortnight interaction (the cleanest Reading C2 test)

GBM with `num_leaves=63` trees *can* find depth-3 interactions, but it
only does so when both parent features are individually high-gain.
`sa2_seifa_irsd_score` isn't high-gain (ranks outside the top-30 by
gain). So maybe the interaction is there but the trees aren't extracting
it.

Test it directly: add an explicit column
`sa2_seifa_x_dof = sa2_seifa_irsd_score * cal_day_of_fortnight` to
Model B's feature set, re-evaluate under k-fold.

**Result.** Model B got 3× worse:

| Fold | PR B baseline (no interaction) | With interaction | Change |
|---|---:|---:|---:|
| fold_1 | +0.074 | −0.107 | helped |
| fold_2 | −0.135 | +0.897 | **much worse** |
| fold_3 | +0.260 | +0.740 | worse |
| fold_4 | −0.098 | +0.323 | worse |
| fold_5 | +0.147 | +0.167 | same |
| fold_6 | +1.042 | **+2.001** | **worst — nearly doubled** |
| **Mean Δ MAE** | **+0.215** | **+0.670** | 3× worse |

LightGBM *does* split on the new column (gain rank 46-58 out of ~89
features) — it's not "the model ignored the feature." The splits just
don't generalise across folds. The explicit interaction made the
augmentor's harm on fold_6 (the 2026 spike) nearly double.

**Reading C2 falsified.** The cleanest test of "the GBM is missing the
Centrelink × SEIFA interaction" failed actively.

### #4 — Optuna hyperparameter sweep on Model A

The last live sub-reading: maybe the spec defaults are the wrong
hyperparameter capacity for Model A. If a proper Bayesian search finds
a meaningfully better config, Reading C1 (hyperparameter mismatch) is
partially confirmed.

Ran Optuna TPE sweep across 8 hyperparameters (num_leaves,
min_data_in_leaf, learning_rate, feature_fraction, bagging_fraction,
bagging_freq, lambda_l1, lambda_l2), 6-fold k-fold objective per trial,
median pruning, 6-hour budget. 30 trials processed (6 done, 24 pruned).

**Best: mean val-MAE 4.8612 vs spec default 5.0616 = improvement 0.200
c/L (3.96%)** — above my 0.05 c/L meaningful-improvement threshold.
Validated across 6 seeds: **mean improvement 0.170 c/L across folds,
stdev 0.132, ratio 1.29 → WEAK WIN** (above 1.0 weak threshold; below
2.0 robust threshold).

The pattern of the tuned config vs the v1/v2 defaults:

| Param | v1/v2 | v3.0 tuned |
|---|---:|---:|
| num_leaves | 63 | 31 (smaller trees) |
| min_data_in_leaf | 200 | 544 (heavier leaf reg) |
| learning_rate | 0.05 | 0.028 (slower) |
| feature_fraction | 0.8 | 0.85 (more cols/tree) |
| bagging_fraction | 0.8 | 0.69 |
| bagging_freq | 5 | **0 (no row bagging)** |
| lambda_l1 | 0 | 0.059 |

Smaller, more-regularized trees with more features per split and no row
bagging. **The v1/v2 defaults were over-fitting.** The tuned config
trades tree capacity for stronger regularization.

**Reading C1 partially confirmed.** Locked into spec §8.2; new defaults
ship with Model A.

## What I take away

1. **Single-split evaluation can flip sign under k-fold.** This is
   well-known in the time-series literature but easy to dismiss when
   the single-split numbers look clean. The v2.x answer was −0.24 c/L
   "Model B beats A"; the v3.0 answer was +0.22 c/L "Model B *loses* to
   A". The two folds the original methodology chose happened to be
   favourable; rotating the test window rotated the answer.

2. **Per-fold variance has structure worth measuring.** When two folds
   (3 and 6 here) dominate cross-experiment variance, "the augmentor is
   inconsistent" is the wrong reading. The right reading is "the model
   is intrinsically unstable on those folds even with zero data change"
   — which the seed-noise experiment proved directly. The augmentor
   wasn't introducing the variance; LightGBM training on those time
   periods was.

3. **Explicit interaction features are a cheap, decisive test of Reading
   C2.** If you hypothesise "the GBM is missing an interaction the lag
   features can't capture," adding the explicit product column and
   re-evaluating answers it in one experiment. The failure direction
   (worse, not the same) is information too: it tells you the model was
   over-fitting on the spurious training-set version of the interaction.

4. **Hyperparameter tuning is worth a shot before declaring "model at
   capacity."** Reading C1 was the only sub-Reading-C that paid off
   here — but the payoff (+0.17 c/L) is much smaller than the augmentor
   claimed (~0.24-0.32 c/L on the original single-split). Tuning unlocked
   real capacity; the augmentor didn't.

5. **A null result with this much triangulation is publishable.** "The
   augmentor doesn't help" sounds like a failed project, but the
   evidence trail (k-fold reversal + per-fold rank consistency + seed-
   noise floor + explicit-interaction falsification + tuning lower-
   bound) makes it a methodology contribution that generalises beyond
   the specific augmentor / dataset.

## The protocol, distilled

If you're about to ship a feature ablation based on a single-split A/B
comparison, run this checklist first:

1. **Time-series k-fold the A/B comparison.** 6 folds, 12-month test
   windows, gap between train end and test start. Report mean ± stdev
   of the per-fold Δ. If `|Mean| < Stdev`, **stop calling it a win**.
2. **Establish the seed-noise floor** for the model on the same folds.
   Same model, different LightGBM/XGBoost/CatBoost seeds, 6× run. The
   ratio of your "augmentor noise band" to your seed-noise floor tells
   you whether the augmentor is adding real instability or just looking
   noisy because the model is intrinsically noisy on those folds.
3. **For any hypothesised interaction**, add the explicit product
   feature and re-evaluate. The result is decisive in both directions:
   improvement → keep, regression → falsifies the hypothesis cleanly.
4. **Then** run a hyperparameter sweep on the baseline model. If your
   baseline is over-tuned, the augmentor will inherit the over-fit and
   look better than it should; if it's under-tuned, the augmentor
   "advantage" might actually be the augmentor accidentally finding
   capacity the baseline left on the table.

Steps 1+4 are obvious; steps 2+3 are the cheap wins that distinguish
real signal from fold-specific lucky-fits.

## What's in the repo

If you want to inspect the actual code + per-fold reports:

- [`aus-fuel-forecaster`](https://github.com/cauldnz/aus-fuel-forecaster) — the project
- [`spec.md` §15](spec.md#15-v30-plan--methodology-overhaul) — v3.0 plan
  and outcome
- [`docs/research/2026-06_v3.0_phase3_closing_summary.md`](docs/research/2026-06_v3.0_phase3_closing_summary.md)
  — the canonical "why we ship Model A" reference
- `results/v3_phase2_summary.md` — 8-experiment k-fold summary
- `results/v3_phase3_*.md` — postmortem experiment reports
- `tools/research/v3_phase3_*.py` — the 4 postmortem runners

## Sources

- Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing
  Supervised Classification Learning Algorithms.* Neural Computation,
  10(7): 1895-1923.
- Nadeau, C. & Bengio, Y. (2003). *Inference for the Generalization
  Error.* Machine Learning, 52: 239-281.
- Liu, X. (2024). *Time-series cross-validation in tabular ML: a survey.*
  BJMSP. [referenced in the rank-consistency analysis]
- LightGBM documentation: feature interactions
  https://lightgbm.readthedocs.io/en/latest/Features.html
- abs-census-augmentor: https://github.com/cauldnz/abs-census-augmentor
