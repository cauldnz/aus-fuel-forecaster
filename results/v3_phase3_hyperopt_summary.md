# v3.0 Phase 3 #4 — Model A hyperparameter sweep (Reading C1)

Optuna TPE Bayesian search over Model A's LightGBM hyperparameters, evaluated on the v3.0 6-fold k-fold harness (mean val-MAE). Study: `phase3_model_a_hyperopt`.

## Headline

- **Best mean val-MAE: 4.8612 c/L**
- Reference (spec §8.2 defaults, seed 42): **5.0616 c/L**
- **Improvement: +0.2004 c/L (+3.96%)** — WIN over spec default
- Trials completed: 8
- Trials pruned: 36

## Best hyperparameters

| Hyperparameter | Best | Spec §8.2 default |
|----------------|------|-------------------|
| num_leaves | 31 | 63 |
| min_data_in_leaf | 544 | 200 |
| learning_rate | 0.02794 | 0.05 |
| feature_fraction | 0.8507 | 0.8 |
| bagging_fraction | 0.6905 | 0.8 |
| bagging_freq | 0 | 5 |
| lambda_l1 | 0.05903 | 0.0 |
| lambda_l2 | 1.013e-07 | 0.0 |

## Top 10 trials

| Rank | Trial | Mean val-MAE | num_leaves | min_data | lr | ff | bf/bf_freq | l1 | l2 |
|-----:|------:|-------------:|-----------:|---------:|---:|---:|-----------:|---:|---:|
| 1 | 15 | 4.8612 | 31 | 544 | 0.0279 | 0.85 | 0.69/0 | 5.90e-02 | 1.01e-07 |
| 2 | 14 | 4.8647 | 31 | 542 | 0.0278 | 0.88 | 0.59/0 | 3.39e-02 | 3.77e-06 |
| 3 | 39 | 4.9035 | 15 | 88 | 0.1170 | 0.95 | 0.43/0 | 1.09e+00 | 1.65e-06 |
| 4 | 6 | 4.9065 | 31 | 978 | 0.0495 | 0.72 | 0.42/0 | 2.17e-04 | 2.63e-08 |
| 5 | 32 | 4.9094 | 31 | 230 | 0.0634 | 0.95 | 0.52/0 | 1.49e-02 | 1.87e-04 |
| 6 | 13 | 4.9227 | 31 | 217 | 0.0340 | 0.88 | 0.54/0 | 6.16e-03 | 7.28e-06 |
| 7 | 3 | 4.9317 | 31 | 126 | 0.0409 | 0.73 | 0.51/0 | 2.40e-03 | 1.98e+00 |
| 8 | 2 | 4.9534 | 63 | 60 | 0.1306 | 0.98 | 0.89/5 | 1.25e-07 | 2.86e-04 |

## Reading

**Spec default beaten by 0.200 c/L (+4.0%)** — meaningful capacity to unlock. Recommend updating spec §8.2 with the new defaults before shipping Model A. Validate the chosen params with a 6× seed sanity check (re-run Phase 3 #2's seed-noise floor with the new params) before locking them in.

## Sources

- `tools/research/v3_phase3_hyperopt_runner.py` — this script
- `models_optuna_a\study.db` — Optuna SQLite study (gitignored)
- `docs/research/2026-06_v3.0_phase2_postmortem_discussion.md` — Reading C1 hypothesis
- `results/v3_phase3_seed_noise_summary.md` — reference for spec-default MAE
