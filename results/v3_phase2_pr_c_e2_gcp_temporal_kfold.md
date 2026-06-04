# Model A vs Model B vs Model B' — k-fold CV comparison report (spec §15.2)

Generated: 2026-06-02 06:24:28 UTC
Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e2_gcp_temporal.parquet`
K-fold models root: `C:\repos\cauldnz\aus-fuel-forecaster\models_kfold_pr_c_e2_gcp_temporal`
K-fold config: k=6, test_window_months=12, val_window_days=365, gap_days=1, horizon_days=1, warmup_end=2016-12-31, panel_end=2026-04-30

v3.0 methodology: per-fold rows + aggregate (mean / stdev /
min / max) replace the single-split test_normal / test_crisis
of v2.x. **Crisis-as-separate is dropped** — 2026 data rotates
into the test windows like every other year. Per-fold spread
vs. mean is the significance signal; no p-values (see design
doc §2.5 for why naive k-fold paired t-tests are misleading).

- **Negative `Δ MAE` = Model B beats Model A** (augmentor adds value)
- **Negative `Δ MAE (B'−B)` = venue features add lift** beyond Model B
- All metrics in cents/L except MAPE (in %)

## Headline — A vs B (per-fold + aggregate)

| Fold | Test window | n | MAE A | MAE B | Δ MAE | RMSE A | RMSE B | MAPE A | MAPE B | Δ MAPE |
|------|-------------|--:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.262 | 6.167 | -0.095 | 10.403 | 10.377 | 5.076 | 4.995 | -0.082 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 8.875 | 9.316 | +0.441 | 14.208 | 14.717 | 5.191 | 5.431 | +0.240 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.198 | 13.484 | +0.285 | 18.800 | 18.823 | 6.818 | 6.975 | +0.156 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 7.037 | 6.953 | -0.084 | 12.454 | 12.165 | 3.468 | 3.428 | -0.040 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 3.919 | 4.178 | +0.258 | 7.992 | 8.294 | 2.059 | 2.200 | +0.141 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 10.505 | 10.627 | +0.122 | 14.348 | 14.793 | 5.347 | 5.413 | +0.066 |
| **Mean** | — | 2,647,445 | 8.300 | 8.454 | +0.155 | 13.034 | 13.195 | 4.660 | 4.740 | +0.080 |
| Stdev | — | — | 3.004 | 3.070 | +0.196 | 3.392 | 3.405 | 1.515 | 1.536 | +0.112 |
| Min | — | — | 3.919 | 4.178 | -0.095 | 7.992 | 8.294 | 2.059 | 2.200 | -0.082 |
| Max | — | — | 13.198 | 13.484 | +0.441 | 18.800 | 18.823 | 6.818 | 6.975 | +0.240 |

## Headline — B vs B' (venue-block additive sanity check, per-fold + aggregate)

| Fold | Test window | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|-------------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.167 | 6.124 | -0.043 | 10.334 | 4.940 | -0.139 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 9.316 | 8.719 | -0.597 | 13.923 | 5.118 | -0.156 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.484 | 14.262 | +0.779 | 19.219 | 7.384 | +1.064 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 6.953 | 7.436 | +0.483 | 12.585 | 3.669 | +0.399 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.178 | 4.259 | +0.081 | 8.259 | 2.250 | +0.340 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 10.627 | 11.467 | +0.840 | 15.269 | 5.873 | +0.962 |
| **Mean** | — | 2,647,445 | 8.454 | 8.711 | +0.257 | 13.265 | 4.872 | +0.412 |
| Stdev | — | — | 3.070 | 3.331 | +0.502 | 3.512 | 1.617 | +0.476 |
| Min | — | — | 4.178 | 4.259 | -0.597 | 8.259 | 2.250 | -0.156 |
| Max | — | — | 13.484 | 14.262 | +0.840 | 19.219 | 7.384 | +1.064 |

## Segmented by Metro / regional (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 2,638,213 | 8.372 | 8.528 | 8.810 | +0.156 | +0.281 | 4.882 |
| True | 9,232 | 8.525 | 8.545 | 8.749 | +0.020 | +0.204 | 4.682 |

## Segmented by Brand (across all folds combined; top 8 + Other)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 623,474 | 8.795 | 8.984 | 9.248 | +0.189 | +0.264 | 5.019 |
| Other | 432,131 | 7.061 | 7.239 | 7.573 | +0.179 | +0.333 | 4.231 |
| 7-Eleven | 359,079 | 9.894 | 9.991 | 10.255 | +0.097 | +0.264 | 5.679 |
| Metro | 288,873 | 7.571 | 7.676 | 7.908 | +0.105 | +0.232 | 4.648 |
| BP | 287,258 | 8.780 | 8.977 | 9.258 | +0.197 | +0.282 | 5.050 |
| Independent | 252,342 | 7.366 | 7.618 | 7.935 | +0.251 | +0.318 | 4.389 |
| Coles Express | 189,598 | 8.921 | 8.946 | 9.122 | +0.026 | +0.176 | 5.052 |
| United | 86,807 | 7.568 | 7.683 | 7.833 | +0.115 | +0.150 | 4.540 |
| Shell | 66,851 | 9.827 | 10.072 | 10.620 | +0.246 | +0.548 | 5.470 |
| Speedway | 41,300 | 6.569 | 6.780 | 7.048 | +0.211 | +0.268 | 4.488 |
| Reddy Express | 19,732 | 11.925 | 11.490 | 12.422 | -0.434 | +0.932 | 6.243 |

## Segmented by Fuel type (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 2,647,445 | 8.373 | 8.528 | 8.809 | +0.156 | +0.281 | 4.882 |

## Segmented by SEIFA quintile (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 487,022 | 8.025 | 8.191 | 8.499 | +0.166 | +0.308 | 4.833 |
| Q3 | 484,406 | 8.198 | 8.387 | 8.713 | +0.190 | +0.326 | 4.812 |
| Q4 | 481,950 | 8.368 | 8.494 | 8.761 | +0.126 | +0.267 | 4.876 |
| Q2 | 481,825 | 7.769 | 7.989 | 8.325 | +0.220 | +0.336 | 4.575 |
| Q5 | 480,223 | 9.366 | 9.346 | 9.627 | -0.020 | +0.281 | 5.249 |
| Unknown | 232,019 | 8.677 | 9.031 | 9.076 | +0.354 | +0.045 | 5.016 |

---

_Generated by `python -m fuel_pred.evaluate.compare_kfold`. v3.0
Phase 1 (spec §15.2). The "Mean" / "Stdev" / "Min" / "Max" rows
aggregate the per-fold metrics for the A-vs-B headline (and the
B-vs-B' table if Model B' was fit). Use stdev as the across-
fold variance signal: if |Mean Δ MAE| ≫ Stdev, the change is
robust; if comparable, fold-specific noise dominates._
