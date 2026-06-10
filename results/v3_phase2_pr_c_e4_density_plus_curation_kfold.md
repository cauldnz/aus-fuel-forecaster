# Model A vs Model B vs Model B' — k-fold CV comparison report (spec §15.2)

Generated: 2026-06-04 08:02:21 UTC
Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e4_new_erp_density_plus_curation.parquet`
K-fold models root: `C:\repos\cauldnz\aus-fuel-forecaster\models_kfold_pr_c_e4_density_plus_curation`
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
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.219 | 6.188 | -0.031 | 10.422 | 10.400 | 5.040 | 4.993 | -0.047 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 8.799 | 9.296 | +0.497 | 13.862 | 14.592 | 5.149 | 5.418 | +0.270 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.331 | 12.851 | -0.480 | 18.771 | 18.216 | 6.897 | 6.645 | -0.251 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 6.954 | 7.086 | +0.133 | 12.290 | 12.460 | 3.425 | 3.490 | +0.065 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.181 | 4.460 | +0.279 | 8.157 | 8.560 | 2.203 | 2.358 | +0.155 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 9.573 | 8.961 | -0.612 | 13.435 | 12.903 | 4.861 | 4.509 | -0.352 |
| **Mean** | — | 2,647,445 | 8.176 | 8.141 | -0.036 | 12.823 | 12.855 | 4.596 | 4.569 | -0.027 |
| Stdev | — | — | 2.893 | 2.667 | +0.396 | 3.283 | 3.066 | 1.470 | 1.370 | +0.218 |
| Min | — | — | 4.181 | 4.460 | -0.612 | 8.157 | 8.560 | 2.203 | 2.358 | -0.352 |
| Max | — | — | 13.331 | 12.851 | +0.497 | 18.771 | 18.216 | 6.897 | 6.645 | +0.270 |

## Headline — B vs B' (venue-block additive sanity check, per-fold + aggregate)

| Fold | Test window | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|-------------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.188 | 6.281 | +0.093 | 10.421 | 5.101 | +0.062 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 9.296 | 8.843 | -0.453 | 14.049 | 5.173 | +0.044 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 12.851 | 13.675 | +0.824 | 18.833 | 7.076 | +0.344 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 7.086 | 6.899 | -0.187 | 12.106 | 3.402 | -0.054 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.460 | 4.288 | -0.172 | 8.144 | 2.262 | +0.107 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 8.961 | 9.158 | +0.197 | 13.007 | 4.619 | -0.415 |
| **Mean** | — | 2,647,445 | 8.141 | 8.191 | +0.050 | 12.760 | 4.606 | +0.015 |
| Stdev | — | — | 2.667 | 2.944 | +0.404 | 3.312 | 1.506 | +0.227 |
| Min | — | — | 4.460 | 4.288 | -0.453 | 8.144 | 2.262 | -0.415 |
| Max | — | — | 12.851 | 13.675 | +0.824 | 18.833 | 7.076 | +0.344 |

## Segmented by Metro / regional (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 2,638,213 | 8.258 | 8.215 | 8.279 | -0.043 | +0.064 | 4.609 |
| True | 9,232 | 8.325 | 8.355 | 8.322 | +0.030 | -0.033 | 4.456 |

## Segmented by Brand (across all folds combined; top 8 + Other)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 623,474 | 8.664 | 8.662 | 8.687 | -0.002 | +0.025 | 4.736 |
| Other | 432,131 | 6.933 | 6.891 | 7.015 | -0.042 | +0.124 | 3.931 |
| 7-Eleven | 359,079 | 9.766 | 9.662 | 9.663 | -0.104 | +0.002 | 5.384 |
| Metro | 288,873 | 7.481 | 7.366 | 7.486 | -0.114 | +0.120 | 4.421 |
| BP | 287,258 | 8.664 | 8.652 | 8.699 | -0.012 | +0.048 | 4.766 |
| Independent | 252,342 | 7.344 | 7.333 | 7.429 | -0.011 | +0.096 | 4.123 |
| Coles Express | 189,598 | 8.893 | 8.919 | 8.948 | +0.027 | +0.029 | 4.978 |
| United | 86,807 | 7.566 | 7.636 | 7.684 | +0.070 | +0.048 | 4.470 |
| Shell | 66,851 | 9.571 | 9.491 | 9.645 | -0.079 | +0.153 | 4.953 |
| Speedway | 41,300 | 6.428 | 6.256 | 6.344 | -0.172 | +0.088 | 4.108 |
| Reddy Express | 19,732 | 10.567 | 9.708 | 9.904 | -0.858 | +0.196 | 4.914 |

## Segmented by Fuel type (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 2,647,445 | 8.258 | 8.215 | 8.279 | -0.043 | +0.064 | 4.609 |

## Segmented by SEIFA quintile (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 487,022 | 7.900 | 7.896 | 7.994 | -0.004 | +0.098 | 4.568 |
| Q3 | 484,406 | 8.086 | 8.076 | 8.179 | -0.010 | +0.102 | 4.535 |
| Q4 | 481,950 | 8.235 | 8.181 | 8.258 | -0.053 | +0.076 | 4.617 |
| Q2 | 481,825 | 7.680 | 7.708 | 7.798 | +0.028 | +0.090 | 4.300 |
| Q5 | 480,223 | 9.211 | 9.008 | 9.077 | -0.203 | +0.070 | 4.974 |
| Unknown | 232,019 | 8.649 | 8.661 | 8.482 | +0.012 | -0.180 | 4.715 |

---

_Generated by `python -m fuel_pred.evaluate.compare_kfold`. v3.0
Phase 1 (spec §15.2). The "Mean" / "Stdev" / "Min" / "Max" rows
aggregate the per-fold metrics for the A-vs-B headline (and the
B-vs-B' table if Model B' was fit). Use stdev as the across-
fold variance signal: if |Mean Δ MAE| ≫ Stdev, the change is
robust; if comparable, fold-specific noise dominates._
