# Model A vs Model B vs Model B' — k-fold CV comparison report (spec §15.2)

Generated: 2026-06-04 12:29:08 UTC
Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e5_dss_temporal_plus_curation.parquet`
K-fold models root: `C:\repos\cauldnz\aus-fuel-forecaster\models_kfold_pr_c_e5_dss_temporal_plus_curation`
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
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.219 | 6.242 | +0.023 | 10.422 | 10.460 | 5.040 | 5.058 | +0.018 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 8.799 | 9.159 | +0.359 | 13.862 | 14.511 | 5.149 | 5.345 | +0.196 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.331 | 12.919 | -0.412 | 18.771 | 18.327 | 6.897 | 6.679 | -0.218 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 6.954 | 7.067 | +0.113 | 12.290 | 12.370 | 3.425 | 3.480 | +0.055 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.722 | 4.444 | -0.278 | 8.758 | 8.475 | 2.501 | 2.353 | -0.148 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 9.654 | 10.153 | +0.499 | 13.687 | 13.710 | 4.884 | 5.164 | +0.281 |
| **Mean** | — | 2,647,445 | 8.280 | 8.331 | +0.051 | 12.965 | 12.975 | 4.649 | 4.680 | +0.031 |
| Stdev | — | — | 2.779 | 2.772 | +0.322 | 3.153 | 3.122 | 1.392 | 1.395 | +0.175 |
| Min | — | — | 4.722 | 4.444 | -0.412 | 8.758 | 8.475 | 2.501 | 2.353 | -0.218 |
| Max | — | — | 13.331 | 12.919 | +0.499 | 18.771 | 18.327 | 6.897 | 6.679 | +0.281 |

## Headline — B vs B' (venue-block additive sanity check, per-fold + aggregate)

| Fold | Test window | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|-------------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.242 | 6.291 | +0.049 | 10.457 | 5.108 | +0.072 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 9.159 | 8.999 | -0.160 | 14.237 | 5.257 | +0.200 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 12.919 | 14.095 | +1.177 | 19.246 | 7.287 | +0.764 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 7.067 | 6.780 | -0.287 | 11.899 | 3.340 | -0.174 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.444 | 4.290 | -0.154 | 8.242 | 2.263 | -0.432 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 10.153 | 10.070 | -0.083 | 13.783 | 5.131 | +0.416 |
| **Mean** | — | 2,647,445 | 8.331 | 8.421 | +0.090 | 12.977 | 4.731 | +0.141 |
| Stdev | — | — | 2.772 | 3.150 | +0.496 | 3.451 | 1.588 | +0.387 |
| Min | — | — | 4.444 | 4.290 | -0.287 | 8.242 | 2.263 | -0.432 |
| Max | — | — | 12.919 | 14.095 | +1.177 | 19.246 | 7.287 | +0.764 |

## Segmented by Metro / regional (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 2,638,213 | 8.363 | 8.400 | 8.509 | +0.037 | +0.109 | 4.734 |
| True | 9,232 | 8.472 | 8.484 | 8.525 | +0.012 | +0.041 | 4.568 |

## Segmented by Brand (across all folds combined; top 8 + Other)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 623,474 | 8.774 | 8.815 | 8.950 | +0.040 | +0.136 | 4.873 |
| Other | 432,131 | 7.129 | 7.112 | 7.250 | -0.017 | +0.138 | 4.065 |
| 7-Eleven | 359,079 | 9.777 | 9.899 | 9.909 | +0.122 | +0.010 | 5.512 |
| Metro | 288,873 | 7.597 | 7.604 | 7.656 | +0.006 | +0.052 | 4.528 |
| BP | 287,258 | 8.750 | 8.836 | 8.941 | +0.086 | +0.105 | 4.893 |
| Independent | 252,342 | 7.483 | 7.443 | 7.663 | -0.041 | +0.220 | 4.253 |
| Coles Express | 189,598 | 8.940 | 8.919 | 8.985 | -0.020 | +0.066 | 4.993 |
| United | 86,807 | 7.663 | 7.624 | 7.750 | -0.039 | +0.126 | 4.510 |
| Shell | 66,851 | 9.631 | 9.751 | 10.059 | +0.120 | +0.308 | 5.174 |
| Speedway | 41,300 | 6.607 | 6.804 | 6.732 | +0.197 | -0.072 | 4.348 |
| Reddy Express | 19,732 | 10.483 | 11.007 | 10.845 | +0.525 | -0.162 | 5.425 |

## Segmented by Fuel type (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 2,647,445 | 8.364 | 8.400 | 8.509 | +0.037 | +0.108 | 4.733 |

## Segmented by SEIFA quintile (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 487,022 | 8.016 | 8.091 | 8.218 | +0.075 | +0.127 | 4.697 |
| Q3 | 484,406 | 8.219 | 8.246 | 8.415 | +0.028 | +0.169 | 4.664 |
| Q4 | 481,950 | 8.343 | 8.383 | 8.474 | +0.040 | +0.091 | 4.735 |
| Q2 | 481,825 | 7.830 | 7.822 | 8.010 | -0.007 | +0.188 | 4.417 |
| Q5 | 480,223 | 9.266 | 9.284 | 9.336 | +0.017 | +0.052 | 5.109 |
| Unknown | 232,019 | 8.680 | 8.778 | 8.708 | +0.098 | -0.069 | 4.834 |

---

_Generated by `python -m fuel_pred.evaluate.compare_kfold`. v3.0
Phase 1 (spec §15.2). The "Mean" / "Stdev" / "Min" / "Max" rows
aggregate the per-fold metrics for the A-vs-B headline (and the
B-vs-B' table if Model B' was fit). Use stdev as the across-
fold variance signal: if |Mean Δ MAE| ≫ Stdev, the change is
robust; if comparable, fold-specific noise dominates._
