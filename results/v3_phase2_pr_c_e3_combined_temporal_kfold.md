# Model A vs Model B vs Model B' — k-fold CV comparison report (spec §15.2)

Generated: 2026-06-04 07:35:36 UTC
Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e3_combined_temporal.parquet`
K-fold models root: `C:\repos\cauldnz\aus-fuel-forecaster\models_kfold_pr_c_e3_combined_temporal`
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
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.133 | 6.109 | -0.024 | 10.335 | 10.347 | 4.956 | 4.933 | -0.023 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 9.025 | 9.061 | +0.036 | 14.231 | 14.264 | 5.272 | 5.289 | +0.017 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.714 | 14.140 | +0.426 | 18.959 | 19.493 | 7.088 | 7.311 | +0.223 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 7.117 | 7.102 | -0.015 | 12.449 | 12.272 | 3.507 | 3.498 | -0.009 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.582 | 4.200 | -0.383 | 8.623 | 8.278 | 2.424 | 2.218 | -0.206 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 8.790 | 10.433 | +1.643 | 12.936 | 14.885 | 4.413 | 5.293 | +0.880 |
| **Mean** | — | 2,647,445 | 8.227 | 8.507 | +0.281 | 12.922 | 13.257 | 4.610 | 4.757 | +0.147 |
| Stdev | — | — | 2.886 | 3.215 | +0.653 | 3.254 | 3.577 | 1.457 | 1.589 | +0.351 |
| Min | — | — | 4.582 | 4.200 | -0.383 | 8.623 | 8.278 | 2.424 | 2.218 | -0.206 |
| Max | — | — | 13.714 | 14.140 | +1.643 | 18.959 | 19.493 | 7.088 | 7.311 | +0.880 |

## Headline — B vs B' (venue-block additive sanity check, per-fold + aggregate)

| Fold | Test window | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|-------------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.109 | 6.190 | +0.081 | 10.365 | 4.995 | +0.057 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 9.061 | 9.146 | +0.085 | 14.446 | 5.352 | +0.121 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 14.140 | 14.507 | +0.367 | 19.543 | 7.508 | +0.793 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 7.102 | 7.193 | +0.091 | 12.426 | 3.546 | +0.077 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.200 | 4.747 | +0.548 | 8.657 | 2.511 | +0.165 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 10.433 | 9.799 | -0.634 | 13.673 | 4.967 | +1.009 |
| **Mean** | — | 2,647,445 | 8.507 | 8.597 | +0.090 | 13.185 | 4.813 | +0.370 |
| Stdev | — | — | 3.215 | 3.143 | +0.367 | 3.447 | 1.555 | +0.382 |
| Min | — | — | 4.200 | 4.747 | -0.634 | 8.657 | 2.511 | +0.057 |
| Max | — | — | 14.140 | 14.507 | +0.548 | 19.543 | 7.508 | +1.009 |

## Segmented by Metro / regional (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 2,638,213 | 8.322 | 8.600 | 8.701 | +0.278 | +0.101 | 4.825 |
| True | 9,232 | 8.534 | 8.672 | 8.748 | +0.138 | +0.076 | 4.678 |

## Segmented by Brand (across all folds combined; top 8 + Other)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 623,474 | 8.704 | 9.061 | 9.141 | +0.357 | +0.080 | 4.966 |
| Other | 432,131 | 7.026 | 7.302 | 7.398 | +0.276 | +0.097 | 4.134 |
| 7-Eleven | 359,079 | 9.740 | 10.004 | 10.128 | +0.265 | +0.124 | 5.617 |
| Metro | 288,873 | 7.498 | 7.705 | 7.769 | +0.207 | +0.064 | 4.569 |
| BP | 287,258 | 8.754 | 9.040 | 9.199 | +0.286 | +0.160 | 5.022 |
| Independent | 252,342 | 7.486 | 7.740 | 7.841 | +0.254 | +0.101 | 4.337 |
| Coles Express | 189,598 | 9.119 | 9.138 | 9.300 | +0.019 | +0.162 | 5.152 |
| United | 86,807 | 7.827 | 7.832 | 8.065 | +0.005 | +0.232 | 4.661 |
| Shell | 66,851 | 9.653 | 10.238 | 10.299 | +0.585 | +0.061 | 5.302 |
| Speedway | 41,300 | 6.374 | 6.623 | 6.561 | +0.249 | -0.062 | 4.219 |
| Reddy Express | 19,732 | 9.520 | 11.467 | 10.854 | +1.946 | -0.613 | 5.419 |

## Segmented by Fuel type (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 2,647,445 | 8.323 | 8.601 | 8.701 | +0.278 | +0.100 | 4.825 |

## Segmented by SEIFA quintile (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 487,022 | 7.965 | 8.215 | 8.365 | +0.250 | +0.149 | 4.760 |
| Q3 | 484,406 | 8.188 | 8.475 | 8.596 | +0.287 | +0.121 | 4.750 |
| Q4 | 481,950 | 8.299 | 8.556 | 8.669 | +0.257 | +0.114 | 4.828 |
| Q2 | 481,825 | 7.775 | 8.102 | 8.193 | +0.327 | +0.091 | 4.505 |
| Q5 | 480,223 | 9.224 | 9.414 | 9.544 | +0.191 | +0.130 | 5.209 |
| Unknown | 232,019 | 8.679 | 9.116 | 9.002 | +0.437 | -0.114 | 4.979 |

---

_Generated by `python -m fuel_pred.evaluate.compare_kfold`. v3.0
Phase 1 (spec §15.2). The "Mean" / "Stdev" / "Min" / "Max" rows
aggregate the per-fold metrics for the A-vs-B headline (and the
B-vs-B' table if Model B' was fit). Use stdev as the across-
fold variance signal: if |Mean Δ MAE| ≫ Stdev, the change is
robust; if comparable, fold-specific noise dominates._
