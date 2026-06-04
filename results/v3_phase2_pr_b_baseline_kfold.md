# Model A vs Model B vs Model B' — k-fold CV comparison report (spec §15.2)

Generated: 2026-06-02 05:28:11 UTC
Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features.parquet`
K-fold models root: `C:\repos\cauldnz\aus-fuel-forecaster\models_kfold_pr_b_baseline`
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
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.219 | 6.293 | +0.074 | 10.422 | 10.537 | 5.040 | 5.110 | +0.070 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 8.799 | 8.664 | -0.135 | 13.862 | 13.860 | 5.149 | 5.070 | -0.079 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.331 | 13.591 | +0.260 | 18.771 | 18.831 | 6.897 | 7.027 | +0.130 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 6.954 | 6.855 | -0.098 | 12.290 | 12.007 | 3.425 | 3.376 | -0.049 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.181 | 4.327 | +0.147 | 8.157 | 8.376 | 2.203 | 2.289 | +0.086 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 9.573 | 10.615 | +1.042 | 13.435 | 14.672 | 4.861 | 5.399 | +0.538 |
| **Mean** | — | 2,647,445 | 8.176 | 8.391 | +0.215 | 12.823 | 13.047 | 4.596 | 4.712 | +0.116 |
| Stdev | — | — | 2.893 | 3.037 | +0.394 | 3.283 | 3.316 | 1.470 | 1.515 | +0.203 |
| Min | — | — | 4.181 | 4.327 | -0.135 | 8.157 | 8.376 | 2.203 | 2.289 | -0.079 |
| Max | — | — | 13.331 | 13.591 | +1.042 | 18.771 | 18.831 | 6.897 | 7.027 | +0.538 |

## Headline — B vs B' (venue-block additive sanity check, per-fold + aggregate)

| Fold | Test window | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|-------------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.293 | 6.207 | -0.085 | 10.382 | 5.021 | -0.012 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 8.664 | 8.907 | +0.243 | 14.091 | 5.216 | +0.108 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.591 | 14.228 | +0.637 | 19.222 | 7.365 | +0.897 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 6.855 | 7.012 | +0.157 | 12.135 | 3.451 | +0.059 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.327 | 4.310 | -0.017 | 8.466 | 2.276 | +0.130 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 10.615 | 11.599 | +0.984 | 15.385 | 5.940 | +2.025 |
| **Mean** | — | 2,647,445 | 8.391 | 8.711 | +0.320 | 13.280 | 4.878 | +0.534 |
| Stdev | — | — | 3.037 | 3.350 | +0.377 | 3.495 | 1.646 | +0.733 |
| Min | — | — | 4.327 | 4.310 | -0.085 | 8.466 | 2.276 | -0.012 |
| Max | — | — | 13.591 | 14.228 | +0.984 | 19.222 | 7.365 | +2.025 |

## Segmented by Metro / regional (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 2,638,213 | 8.258 | 8.472 | 8.798 | +0.214 | +0.327 | 4.882 |
| True | 9,232 | 8.325 | 8.398 | 8.732 | +0.073 | +0.334 | 4.678 |

## Segmented by Brand (across all folds combined; top 8 + Other)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 623,474 | 8.664 | 8.904 | 9.274 | +0.240 | +0.370 | 5.037 |
| Other | 432,131 | 6.933 | 7.200 | 7.540 | +0.267 | +0.340 | 4.218 |
| 7-Eleven | 359,079 | 9.766 | 9.982 | 10.281 | +0.216 | +0.299 | 5.699 |
| Metro | 288,873 | 7.481 | 7.652 | 7.887 | +0.171 | +0.235 | 4.646 |
| BP | 287,258 | 8.664 | 8.908 | 9.258 | +0.244 | +0.350 | 5.053 |
| Independent | 252,342 | 7.344 | 7.533 | 7.861 | +0.189 | +0.329 | 4.354 |
| Coles Express | 189,598 | 8.893 | 8.830 | 9.063 | -0.063 | +0.233 | 5.028 |
| United | 86,807 | 7.566 | 7.632 | 7.780 | +0.066 | +0.148 | 4.517 |
| Shell | 66,851 | 9.571 | 9.898 | 10.504 | +0.328 | +0.606 | 5.408 |
| Speedway | 41,300 | 6.428 | 6.849 | 7.128 | +0.421 | +0.279 | 4.545 |
| Reddy Express | 19,732 | 10.567 | 11.622 | 12.622 | +1.056 | +1.000 | 6.350 |

## Segmented by Fuel type (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 2,647,445 | 8.258 | 8.472 | 8.798 | +0.213 | +0.327 | 4.881 |

## Segmented by SEIFA quintile (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 487,022 | 7.900 | 8.108 | 8.433 | +0.208 | +0.325 | 4.805 |
| Q3 | 484,406 | 8.086 | 8.307 | 8.709 | +0.220 | +0.402 | 4.815 |
| Q4 | 481,950 | 8.235 | 8.429 | 8.766 | +0.194 | +0.337 | 4.883 |
| Q2 | 481,825 | 7.680 | 7.928 | 8.289 | +0.248 | +0.361 | 4.562 |
| Q5 | 480,223 | 9.211 | 9.292 | 9.672 | +0.081 | +0.380 | 5.278 |
| Unknown | 232,019 | 8.649 | 9.096 | 9.068 | +0.447 | -0.029 | 5.015 |

---

_Generated by `python -m fuel_pred.evaluate.compare_kfold`. v3.0
Phase 1 (spec §15.2). The "Mean" / "Stdev" / "Min" / "Max" rows
aggregate the per-fold metrics for the A-vs-B headline (and the
B-vs-B' table if Model B' was fit). Use stdev as the across-
fold variance signal: if |Mean Δ MAE| ≫ Stdev, the change is
robust; if comparable, fold-specific noise dominates._
