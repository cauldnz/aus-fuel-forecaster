# v5 — SA2 augmentor at the 7-day horizon (y_t1_t7)

Re-runs the v3.0 Phase 2 PR-B-baseline A-vs-B comparison at the 7-day target. Tests whether the augmentor helps where the lag features carry less information.

Full report: `results\v5_7day_horizon_kfold.md`

## Headline — A vs B at t+7

- **Mean Δ MAE: +0.041 c/L** (negative = Model B with SA2 beats Model A)
- Stdev across 6 folds: 0.370
- **Verdict: noise**

## t+7 vs t+1 (the key comparison)

| Metric | t+1 (v3.0 Phase 2) | t+7 (this run) |
|--------|-------------------:|---------------:|
| Mean Δ MAE (B−A) | +0.215 | +0.041 |
| Stdev | 0.394 | 0.370 |

## Per-fold Δ MAE at t+7

| Fold | MAE A | MAE B | Δ MAE |
|------|------:|------:|------:|
| fold_1 | 8.217 | 8.211 | -0.007 |
| fold_2 | 13.835 | 14.172 | +0.337 |
| fold_3 | 25.157 | 25.217 | +0.060 |
| fold_4 | 16.162 | 16.752 | +0.590 |
| fold_5 | 8.817 | 8.666 | -0.152 |
| fold_6 | 18.807 | 18.222 | -0.585 |

## Reading

**Augmentor still null at t+7.** Same noise-band outcome as t+1 — the v3.0 conclusion generalises across horizons. The lag features weakening did NOT open a gap the augmentor fills; whatever predicts the 7-day mean, it isn't SA2 demographics. Strengthens the methodology story (null holds at 2 horizons).

## Sources

- `results\v5_7day_horizon_kfold.md` — full per-fold report
- `tools/research/v5_7day_horizon.py` — this script
- `results/v3_phase2_pr_b_baseline_kfold.md` — t+1 reference
