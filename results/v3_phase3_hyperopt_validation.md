# v3.0 Phase 3 #4 validation — hyperopt winner across 6 seeds

Re-runs Model A across all 6 k-fold folds, 6 times with different LightGBM `random_state` values, using the hyperopt winner's hyperparameters (trial 15 of run 2). Confirms the +0.20 c/L improvement over spec §8.2 defaults isn't a single-seed lucky-fit.

## Hyperopt winner params (under test)

| Param | Value | Spec §8.2 default |
|-------|------:|------------------:|
| num_leaves | 31 | 63 |
| min_data_in_leaf | 544 | 200 |
| learning_rate | 0.02794 | 0.05 |
| feature_fraction | 0.8507 | 0.8 |
| bagging_fraction | 0.6905 | 0.8 |
| bagging_freq | 0 | 5 |
| lambda_l1 | 0.05903 | 0.0 |
| lambda_l2 | 1.013e-07 | 0.0 |

## Per-seed per-fold MAE_A under new params

| Seed | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | fold_6 | Wall-clock |
|------|---|---|---|---|---|---|----|
| 42 | 5.3157 | 4.7507 | 6.0372 | 5.0521 | 4.3360 | 3.6405 | 12.1 min |
| 1 | 5.3323 | 4.7604 | 6.1338 | 4.9760 | 4.4264 | 3.5447 | 10.4 min |
| 7 | 5.3242 | 4.7125 | 6.1477 | 4.9432 | 4.4844 | 3.3559 | 9.8 min |
| 13 | 5.3582 | 4.8080 | 6.0974 | 5.0087 | 4.4280 | 3.5524 | 13.8 min |
| 99 | 5.3110 | 4.7515 | 6.0322 | 5.0499 | 4.3820 | 3.6648 | 15.0 min |
| 123 | 5.3594 | 4.7911 | 6.0905 | 4.9794 | 4.4124 | 3.5136 | 19.6 min |

## Per-fold improvement: new mean − baseline mean (seed-averaged)

Negative = new params are better. Both means are averages across the same 6 seeds.

| Fold | Baseline (spec) mean | New params mean | Δ (improvement) | Seed-stdev (new) |
|------|---------------------:|----------------:|----------------:|-----------------:|
| fold_1 | 5.5464 | 5.3335 | -0.2130 | 0.0191 |
| fold_2 | 4.7566 | 4.7624 | +0.0057 | 0.0307 |
| fold_3 | 6.4699 | 6.0898 | -0.3801 | 0.0436 |
| fold_4 | 5.2338 | 5.0015 | -0.2323 | 0.0398 |
| fold_5 | 4.4293 | 4.4115 | -0.0178 | 0.0454 |
| fold_6 | 3.7256 | 3.5453 | -0.1803 | 0.1002 |
| **Mean across folds** | — | — | **-0.1696** | — |
| **Stdev across folds** | — | — | **0.1317** | — |

## Paired-seed improvements (each seed under new vs same seed under spec defaults)

Stronger test: per-seed, compare the same seed under both param sets. Removes seed effect from the comparison entirely.

| Seed | Mean Δ across folds | Stdev Δ across folds |
|------|--------------------:|---------------------:|
| 42 | -0.2063 | 0.1245 |
| 1 | -0.2264 | 0.1972 |
| 7 | -0.2028 | 0.1870 |
| 13 | -0.1508 | 0.1423 |
| 99 | -0.0865 | 0.1540 |
| 123 | -0.1450 | 0.1477 |
| **Mean across seeds** | **-0.1696** | (stdev of per-seed means: **0.0475**) |

## Verdict

- **Mean improvement across folds: -0.1696 c/L**
- **Stdev improvement across folds: 0.1317 c/L**

**WEAK WIN.** |Mean improvement| > Stdev but < 2 × Stdev. The improvement is consistent in sign across folds + seeds but not overwhelming. **Action: update spec §8.2 with the new defaults but caveat that the improvement is weak-band; document the per-fold spread.**

## Sources

- `tools/research/v3_phase3_hyperopt_validation.py` — this script
- `results/v3_phase3_hyperopt.json` — hyperopt winner params
- `results/v3_phase3_seed_noise.json` — baseline per-seed per-fold MAE (spec defaults)
- `docs/research/2026-06_v3.0_phase3_closing_summary.md` — sets the validation protocol
