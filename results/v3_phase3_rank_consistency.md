# v3.0 Phase 3 — per-fold rank consistency across the 8 Phase 2 experiments

Re-analysis of the 8 ``v3_phase2_*_kfold.md`` reports. No retraining; purely structural. **Reading-A-vs-Reading-C disambiguator** from ``2026-06_v3.0_phase2_postmortem_discussion.md`` next-steps #1.

## TL;DR

- **Mean pairwise Spearman ρ across experiments: +0.198** (median +0.371). N = 28 pairs of 8 experiments × 6 folds.
- Reading guide:
  - ρ near +1 → all experiments rank the folds the same way → there's a real structural per-period effect; augmentor variants are all sub-tuning against the same noise.
  - ρ near 0 → fold rankings shuffle randomly across experiments → noise dominates (Reading A).
  - ρ negative → augmentor variants actively disagree about which folds favour B (would suggest model class is wrong, or different blocks help different periods).

## Per-fold Δ MAE matrix (rows = folds, cols = experiments)

Cell colour-code (text): `-` Model B beats A by >0.01 c/L; `+` A beats B by >0.01 c/L; `=` within ±0.01.

| Fold | pr_b_baseline | e1_dss | e2_gcp | e3_combined | e4_dens+cur | e4a_dens | e4b_cur | e5_dss+cur | Sign tally |
|------|---------------|--------|--------|-------------|-------------|----------|---------|------------|----|
| fold_1 | +0.074+ | +0.015+ | -0.095- | -0.024- | -0.031- | -0.041- | +0.106+ | +0.023+ | 4B / 4A / 0= |
| fold_2 | -0.135- | -0.095- | +0.441+ | +0.036+ | +0.497+ | +0.467+ | +0.380+ | +0.359+ | 2B / 6A / 0= |
| fold_3 | +0.260+ | +0.222+ | +0.285+ | +0.426+ | -0.480- | +0.593+ | +0.764+ | -0.412- | 2B / 6A / 0= |
| fold_4 | -0.098- | +0.069+ | -0.084- | -0.015- | +0.133+ | +0.464+ | +0.017+ | +0.113+ | 3B / 5A / 0= |
| fold_5 | +0.147+ | -0.645- | +0.258+ | -0.383- | +0.279+ | +0.072+ | +0.262+ | -0.278- | 3B / 5A / 0= |
| fold_6 | +1.042+ | +1.743+ | +0.122+ | +1.643+ | -0.612- | +1.483+ | +0.731+ | +0.499+ | 1B / 7A / 0= |

**Reading:** an entirely consistent column would have the same sign for all 6 folds. Mixed signs within a column mean the augmentor variant flips between helping and hurting depending on period.

## Per-fold cross-experiment summary

| Fold | Mean Δ | Stdev Δ | Min Δ | Max Δ | n_B wins | n_A wins | n_ties | Variance share |
|------|-------:|--------:|------:|------:|---------:|---------:|-------:|---------------:|
| fold_1 | +0.003 | 0.061 | -0.095 | +0.106 | 4 | 4 | 0 | 0.4% |
| fold_2 | +0.244 | 0.246 | -0.135 | +0.497 | 2 | 6 | 0 | 6.4% |
| fold_3 | +0.207 | 0.414 | -0.480 | +0.764 | 2 | 6 | 0 | 17.9% |
| fold_4 | +0.075 | 0.167 | -0.098 | +0.464 | 3 | 5 | 0 | 2.9% |
| fold_5 | -0.036 | 0.330 | -0.645 | +0.279 | 3 | 5 | 0 | 11.4% |
| fold_6 | +0.831 | 0.764 | -0.612 | +1.743 | 1 | 7 | 0 | 61.0% |

**Variance share** = sum of squared deviations contributed by this fold (across 8 experiments) divided by the total summed across all folds. If one fold's share approaches 1.0, the cross-experiment noise is mostly that fold's behaviour.

## Pairwise Spearman ρ of per-fold Δ MAE rankings (8×8, symmetric, diag = 1.000)

| | pr_b_baseline | e1_dss | e2_gcp | e3_combined | e4_dens+cur | e4a_dens | e4b_cur | e5_dss+cur |
|---|---|---|---|---|---|---|---|---|
| pr_b_baseline | 1.000 | +0.600 | -0.086 | +0.429 | -0.829 | +0.486 | +0.600 | -0.143 |
| e1_dss | +0.600 | 1.000 | -0.200 | +0.829 | -0.886 | +0.714 | +0.429 | +0.257 |
| e2_gcp | -0.086 | -0.200 | 1.000 | +0.314 | +0.371 | +0.486 | +0.657 | -0.086 |
| e3_combined | +0.429 | +0.829 | +0.314 | 1.000 | -0.600 | +0.943 | +0.714 | +0.429 |
| e4_dens+cur | -0.829 | -0.886 | +0.371 | -0.600 | 1.000 | -0.486 | -0.429 | -0.029 |
| e4a_dens | +0.486 | +0.714 | +0.486 | +0.943 | -0.486 | 1.000 | +0.771 | +0.371 |
| e4b_cur | +0.600 | +0.429 | +0.657 | +0.714 | -0.429 | +0.771 | 1.000 | -0.086 |
| e5_dss+cur | -0.143 | +0.257 | -0.086 | +0.429 | -0.029 | +0.371 | -0.086 | 1.000 |

Mean off-diagonal ρ: **+0.198** (median +0.371).

## Interpretation

**Headline: mean ρ = +0.198, median ρ = +0.371.** The mean is near zero but the median is moderately positive — that's a cluster-with-outliers pattern, not a uniform-noise pattern. Investigation:

**Mean ρ of each experiment vs its 7 peers** (sorted ascending — bottom rows are the experiments that anti-correlate with the cluster):

| Experiment | Mean peer-ρ |
|---|---:|
| e4_dens+cur | -0.412 |
| e5_dss+cur | +0.102 |
| pr_b_baseline | +0.151 |
| e2_gcp | +0.208 |
| e1_dss | +0.249 |
| e4b_cur | +0.380 |
| e3_combined | +0.437 |
| e4a_dens | +0.469 |

**Outliers** (peer-ρ < −0.2): `e4_dens+cur`. These experiments actively disagree with the cluster — they rank the folds in the opposite order.

**Cluster** (peer-ρ > +0.3): `e4b_cur, e3_combined, e4a_dens`. These experiments share a common per-fold ranking — most reliably, fold_6 (2026 spike) and fold_3 (2022-23) are the hardest folds for Model B.

### What this means for the three readings

- **Reading A (genuinely flat) — partial support.** The fact that no fold has unanimous sign across the 8 experiments + mean ρ is only weakly positive is consistent with noise dominating. Same fold can be a B-win or B-loss depending on which variant you fit.
- **Reading C (wrong features) — sharpened.** The cluster pattern indicates there *is* a shared per-fold structure most experiments fail on (fold_3, fold_6). The augmentor variants don't differentiate enough — they're all stuck on the same problem. If Reading C2 (missing explicit interaction) is right, the interaction feature would specifically help on fold_3 or fold_6.
- **Outlier interest.** `e4_dens+cur` is the only experiment that helps on fold_6 — that's the experiment to inspect for what it does differently. Either it found a real fold_6 signal (would be the strongest Reading-C evidence in the data) or fold_6 is so noisy that 1 in 8 random variants lands favourably.

### Per-fold sign tallies

Folds where **all 8 experiments agree on the sign** of Δ MAE:

_(none — no fold has unanimous agreement across all 8 experiments)_

Folds where the **dominant sign** (B-beats-A or A-beats-B) holds in ≥6 of 8:

- fold_2: A beats B in 6/8
- fold_3: A beats B in 6/8
- fold_6: A beats B in 7/8

### Per-fold MAE_A (fold difficulty context)

Reference table — across the 8 experiments, Model A's per-fold MAE varies depending on whether the experiment changes the row-filter (curation broadening removes some rows from the identical-rows guard, which can shift Model A's MAE).

| Fold | pr_b_baseline | e1_dss | e2_gcp | e3_combined | e4_dens+cur | e4a_dens | e4b_cur | e5_dss+cur |
|------|---------------|--------|--------|-------------|-------------|----------|---------|------------|
| fold_1 | 6.22 | 6.22 | 6.26 | 6.13 | 6.22 | 6.22 | 6.22 | 6.22 |
| fold_2 | 8.80 | 8.80 | 8.88 | 9.03 | 8.80 | 8.80 | 8.80 | 8.80 |
| fold_3 | 13.33 | 13.33 | 13.20 | 13.71 | 13.33 | 13.33 | 13.33 | 13.33 |
| fold_4 | 6.95 | 6.95 | 7.04 | 7.12 | 6.95 | 6.95 | 6.95 | 6.95 |
| fold_5 | 4.18 | 4.72 | 3.92 | 4.58 | 4.18 | 4.18 | 4.18 | 4.72 |
| fold_6 | 9.57 | 9.65 | 10.51 | 8.79 | 9.57 | 9.57 | 9.57 | 9.65 |

## Sources

- ``results/v3_phase2_*_kfold.md`` (8 per-experiment reports)
- ``docs/research/2026-06_v3.0_phase2_outcome.md`` (Phase 2 outcome)
- ``docs/research/2026-06_v3.0_phase2_postmortem_discussion.md`` (three-readings deep dive; this script implements next-step #1)
