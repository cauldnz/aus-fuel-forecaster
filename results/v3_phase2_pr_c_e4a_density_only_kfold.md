# Model A vs Model B vs Model B' — k-fold CV comparison report (spec §15.2)

Generated: 2026-06-04 08:29:38 UTC
Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e4a_density_only.parquet`
K-fold models root: `C:\repos\cauldnz\aus-fuel-forecaster\models_kfold_pr_c_e4a_density_only`
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
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.219 | 6.178 | -0.041 | 10.422 | 10.400 | 5.040 | 4.994 | -0.046 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 8.799 | 9.266 | +0.467 | 13.862 | 14.460 | 5.149 | 5.412 | +0.263 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.331 | 13.924 | +0.593 | 18.771 | 19.012 | 6.897 | 7.208 | +0.311 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 6.954 | 7.418 | +0.464 | 12.290 | 12.495 | 3.425 | 3.657 | +0.232 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.181 | 4.253 | +0.072 | 8.157 | 8.396 | 2.203 | 2.239 | +0.036 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 9.573 | 11.056 | +1.483 | 13.435 | 14.888 | 4.861 | 5.657 | +0.796 |
| **Mean** | — | 2,647,445 | 8.176 | 8.683 | +0.506 | 12.823 | 13.275 | 4.596 | 4.861 | +0.265 |
| Stdev | — | — | 2.893 | 3.187 | +0.492 | 3.283 | 3.406 | 1.470 | 1.570 | +0.269 |
| Min | — | — | 4.181 | 4.253 | -0.041 | 8.157 | 8.396 | 2.203 | 2.239 | -0.046 |
| Max | — | — | 13.331 | 13.924 | +1.483 | 18.771 | 19.012 | 6.897 | 7.208 | +0.796 |

## Headline — B vs B' (venue-block additive sanity check, per-fold + aggregate)

| Fold | Test window | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|-------------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.178 | 6.107 | -0.071 | 10.294 | 4.924 | -0.112 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 9.266 | 9.013 | -0.253 | 14.285 | 5.268 | +0.214 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.924 | 13.264 | -0.661 | 18.460 | 6.875 | -0.067 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 7.418 | 6.955 | -0.462 | 12.253 | 3.420 | +0.002 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.253 | 4.179 | -0.074 | 8.220 | 2.213 | -0.002 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 11.056 | 9.198 | -1.858 | 13.008 | 4.645 | -0.375 |
| **Mean** | — | 2,647,445 | 8.683 | 8.119 | -0.563 | 12.753 | 4.557 | -0.057 |
| Stdev | — | — | 3.187 | 2.868 | +0.616 | 3.210 | 1.461 | +0.175 |
| Min | — | — | 4.253 | 4.179 | -1.858 | 8.220 | 2.213 | -0.375 |
| Max | — | — | 13.924 | 13.264 | -0.071 | 18.460 | 6.875 | +0.214 |

## Segmented by Metro / regional (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 2,638,213 | 8.258 | 8.769 | 8.202 | +0.511 | -0.567 | 4.560 |
| True | 9,232 | 8.325 | 8.844 | 8.202 | +0.519 | -0.643 | 4.387 |

## Segmented by Brand (across all folds combined; top 8 + Other)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 623,474 | 8.664 | 9.241 | 8.639 | +0.577 | -0.602 | 4.699 |
| Other | 432,131 | 6.933 | 7.391 | 6.879 | +0.458 | -0.512 | 3.856 |
| 7-Eleven | 359,079 | 9.766 | 10.325 | 9.721 | +0.559 | -0.604 | 5.399 |
| Metro | 288,873 | 7.481 | 7.789 | 7.372 | +0.308 | -0.416 | 4.352 |
| BP | 287,258 | 8.664 | 9.280 | 8.623 | +0.617 | -0.658 | 4.717 |
| Independent | 252,342 | 7.344 | 7.809 | 7.251 | +0.466 | -0.559 | 4.027 |
| Coles Express | 189,598 | 8.893 | 9.227 | 8.853 | +0.334 | -0.374 | 4.914 |
| United | 86,807 | 7.566 | 7.914 | 7.548 | +0.348 | -0.366 | 4.388 |
| Shell | 66,851 | 9.571 | 10.534 | 9.551 | +0.963 | -0.983 | 4.910 |
| Speedway | 41,300 | 6.428 | 6.772 | 6.211 | +0.343 | -0.561 | 4.014 |
| Reddy Express | 19,732 | 10.567 | 12.532 | 10.175 | +1.965 | -2.357 | 5.071 |

## Segmented by Fuel type (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 2,647,445 | 8.258 | 8.770 | 8.202 | +0.511 | -0.567 | 4.559 |

## Segmented by SEIFA quintile (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 487,022 | 7.900 | 8.398 | 7.902 | +0.498 | -0.496 | 4.511 |
| Q3 | 484,406 | 8.086 | 8.626 | 8.054 | +0.539 | -0.572 | 4.460 |
| Q4 | 481,950 | 8.235 | 8.720 | 8.185 | +0.485 | -0.534 | 4.569 |
| Q2 | 481,825 | 7.680 | 8.220 | 7.656 | +0.540 | -0.564 | 4.220 |
| Q5 | 480,223 | 9.211 | 9.633 | 9.112 | +0.422 | -0.520 | 4.981 |
| Unknown | 232,019 | 8.649 | 9.309 | 8.430 | +0.660 | -0.879 | 4.678 |

---

_Generated by `python -m fuel_pred.evaluate.compare_kfold`. v3.0
Phase 1 (spec §15.2). The "Mean" / "Stdev" / "Min" / "Max" rows
aggregate the per-fold metrics for the A-vs-B headline (and the
B-vs-B' table if Model B' was fit). Use stdev as the across-
fold variance signal: if |Mean Δ MAE| ≫ Stdev, the change is
robust; if comparable, fold-specific noise dominates._
