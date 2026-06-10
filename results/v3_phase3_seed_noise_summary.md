# v3.0 Phase 3 #2 — Model A baseline-vs-baseline seed noise floor

Trained Model A only on the v3.0 6-fold k-fold harness 6 times, varying only LightGBM's `random_state`. Tests whether the published Δ MAE stdev for PR B (0.394 c/L) is a real augmentor signal or noise within the same model's seed-driven jitter.

## Methodology recap

- Seeds: [42, 1, 7, 13, 99, 123]
- Same features.parquet (committed PR B baseline) for every seed
- Same 6-fold k-fold geometry as Phase 2 (spec §15.2)
- **Model A only** per seed (skips B and B') — the question is seed effect on the same model, not augmentor behaviour
- MAE values reported here are LightGBM best_val_mae (early-stopping val MAE), NOT test MAE

## Per-seed per-fold MAE_A (validation)

| Seed | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | fold_6 | Wall-clock |
|------|---|---|---|---|---|---|----|
| 42 | 5.6122 | 4.7928 | 6.2463 | 5.1863 | 4.4682 | 4.0639 | 6.4 min |
| 1 | 5.6219 | 4.7459 | 6.7241 | 5.2687 | 4.5611 | 3.6101 | 6.9 min |
| 7 | 5.5427 | 4.7027 | 6.6266 | 5.2806 | 4.4265 | 3.6056 | 6.4 min |
| 13 | 5.4985 | 4.8072 | 6.4626 | 5.2334 | 4.3758 | 3.7799 | 8.6 min |
| 99 | 5.5281 | 4.7435 | 6.3399 | 5.2046 | 4.3761 | 3.5185 | 8.5 min |
| 123 | 5.4753 | 4.7476 | 6.4199 | 5.2294 | 4.3683 | 3.7757 | 8.4 min |

## Aggregate (1) — per-fold seed-stdev of MAE_A

How much does a single fold's val-MAE move when you only change the seed? This is the per-fold noise floor.

| Fold | Mean MAE_A | Seed stdev | Seed range |
|------|-----------:|-----------:|-----------:|
| fold_1 | 5.5464 | 0.0544 | 0.1466 |
| fold_2 | 4.7566 | 0.0345 | 0.1046 |
| fold_3 | 6.4699 | 0.1626 | 0.4778 |
| fold_4 | 5.2338 | 0.0330 | 0.0942 |
| fold_5 | 4.4293 | 0.0687 | 0.1928 |
| fold_6 | 3.7256 | 0.1782 | 0.5454 |
| **Mean across folds** | — | **0.0886** | — |

## Aggregate (2) — across seed-pairs, stdev of per-fold MAE_A difference

For each pair of seeds, compute the per-fold MAE_A difference (seed_i − seed_j) across 6 folds, then take the stdev across folds. **This is the direct analogue of the published Δ MAE stdev (B vs A).** Averages over all 15 pairs.

| Seed pair | Mean Δ across folds | Stdev Δ across folds |
|-----------|--------------------:|---------------------:|
| 13 vs 123 | +0.0235 | 0.0211 |
| 1 vs 7 | +0.0579 | 0.0514 |
| 13 vs 99 | +0.0745 | 0.0965 |
| 99 vs 123 | -0.0509 | 0.1004 |
| 7 vs 99 | +0.0790 | 0.1021 |
| 7 vs 13 | +0.0045 | 0.1117 |
| 7 vs 123 | +0.0281 | 0.1151 |
| 1 vs 99 | +0.1369 | 0.1230 |
| 42 vs 123 | +0.0589 | 0.1443 |
| 1 vs 13 | +0.0624 | 0.1463 |
| 1 vs 123 | +0.0859 | 0.1505 |
| 42 vs 13 | +0.0354 | 0.1548 |
| 42 vs 99 | +0.1099 | 0.2050 |
| 42 vs 7 | +0.0309 | 0.2492 |
| 42 vs 1 | -0.0270 | 0.2731 |
| **Mean across pairs** | **+0.0577** (avg abs) | **0.1363** |

## Comparison to published Phase 2 PR B baseline

Headline numbers from `results/v3_phase2_pr_b_baseline_kfold.md`:

- Δ MAE Mean (B − A):   **+0.215** c/L
- Δ MAE Stdev (across folds): **0.394** c/L
- Δ MAE Min:    -0.135 c/L
- Δ MAE Max:    +1.042 c/L

Seed-noise across-pairs stdev:  **0.1363** c/L (avg over 15 pairs)

**Ratio: published Δ stdev / seed Δ stdev = 2.89**

Reading guide:

- **Ratio ≤ 1.5** → augmentor's noise band is within 1.5× of two seeds disagreeing → **Reading A confirmed**: 0.394 is essentially seed jitter; there's nothing for Model B to be "better at".
- **Ratio ~2-4** → augmentor adds real per-fold noise on top of seed noise (the augmentor block is genuinely struggling fold-to-fold), but no robust signal in the mean — Reading A + Reading C2 hybrid.
- **Ratio ≥ 5** → augmentor noise dwarfs seed noise → there's real per-fold instability Model B is introducing that Model A doesn't have → sharpens Reading C: model class / interaction feature might be needed.

## Sources

- `tools/research/v3_phase3_seed_noise_runner.py` (this script)
- `docs/research/2026-06_v3.0_phase2_postmortem_discussion.md` (next-step #2 in the ranked list)
- `results/v3_phase2_pr_b_baseline_kfold.md` (reference numbers)
