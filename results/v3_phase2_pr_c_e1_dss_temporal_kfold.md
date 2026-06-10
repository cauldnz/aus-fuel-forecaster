# Model A vs Model B vs Model B' — k-fold CV comparison report (spec §15.2)

Generated: 2026-06-02 05:52:55 UTC
Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e1_dss_temporal.parquet`
K-fold models root: `C:\repos\cauldnz\aus-fuel-forecaster\models_kfold_pr_c_e1_dss_temporal`
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
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.219 | 6.235 | +0.015 | 10.422 | 10.455 | 5.040 | 5.052 | +0.012 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 8.799 | 8.704 | -0.095 | 13.862 | 13.965 | 5.149 | 5.091 | -0.058 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.331 | 13.553 | +0.222 | 18.771 | 18.833 | 6.897 | 7.000 | +0.103 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 6.954 | 7.023 | +0.069 | 12.290 | 12.225 | 3.425 | 3.455 | +0.030 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.722 | 4.077 | -0.645 | 8.758 | 8.185 | 2.501 | 2.150 | -0.351 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 9.654 | 11.397 | +1.743 | 13.687 | 15.331 | 4.884 | 5.836 | +0.952 |
| **Mean** | — | 2,647,445 | 8.280 | 8.498 | +0.218 | 12.965 | 13.166 | 4.649 | 4.764 | +0.115 |
| Stdev | — | — | 2.779 | 3.184 | +0.734 | 3.153 | 3.427 | 1.392 | 1.575 | +0.401 |
| Min | — | — | 4.722 | 4.077 | -0.645 | 8.758 | 8.185 | 2.501 | 2.150 | -0.351 |
| Max | — | — | 13.331 | 13.553 | +1.743 | 18.771 | 18.833 | 6.897 | 7.000 | +0.952 |

## Headline — B vs B' (venue-block additive sanity check, per-fold + aggregate)

| Fold | Test window | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|-------------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.235 | 6.103 | -0.132 | 10.338 | 4.916 | -0.116 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 8.704 | 9.254 | +0.549 | 14.591 | 5.412 | +0.454 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.553 | 14.588 | +1.035 | 19.561 | 7.557 | +1.257 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 7.023 | 6.999 | -0.024 | 12.041 | 3.446 | +0.045 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.077 | 4.323 | +0.246 | 8.273 | 2.284 | -0.399 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 11.397 | 11.467 | +0.071 | 15.090 | 5.888 | +1.813 |
| **Mean** | — | 2,647,445 | 8.498 | 8.789 | +0.291 | 13.316 | 4.917 | +0.509 |
| Stdev | — | — | 3.184 | 3.449 | +0.398 | 3.645 | 1.696 | +0.785 |
| Min | — | — | 4.077 | 4.323 | -0.132 | 8.273 | 2.284 | -0.399 |
| Max | — | — | 13.553 | 14.588 | +1.035 | 19.561 | 7.557 | +1.813 |

## Segmented by Metro / regional (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 2,638,213 | 8.363 | 8.575 | 8.882 | +0.212 | +0.306 | 4.924 |
| True | 9,232 | 8.472 | 8.659 | 8.848 | +0.186 | +0.189 | 4.738 |

## Segmented by Brand (across all folds combined; top 8 + Other)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 623,474 | 8.774 | 9.031 | 9.347 | +0.257 | +0.316 | 5.076 |
| Other | 432,131 | 7.129 | 7.326 | 7.556 | +0.197 | +0.230 | 4.226 |
| 7-Eleven | 359,079 | 9.777 | 10.034 | 10.457 | +0.257 | +0.422 | 5.791 |
| Metro | 288,873 | 7.597 | 7.742 | 7.937 | +0.144 | +0.195 | 4.665 |
| BP | 287,258 | 8.750 | 9.019 | 9.360 | +0.269 | +0.341 | 5.107 |
| Independent | 252,342 | 7.483 | 7.608 | 7.902 | +0.125 | +0.294 | 4.374 |
| Coles Express | 189,598 | 8.940 | 8.905 | 9.247 | -0.034 | +0.341 | 5.118 |
| United | 86,807 | 7.663 | 7.577 | 7.957 | -0.087 | +0.380 | 4.604 |
| Shell | 66,851 | 9.631 | 10.173 | 10.622 | +0.542 | +0.448 | 5.487 |
| Speedway | 41,300 | 6.607 | 7.011 | 6.933 | +0.404 | -0.079 | 4.414 |
| Reddy Express | 19,732 | 10.483 | 12.235 | 12.655 | +1.752 | +0.421 | 6.393 |

## Segmented by Fuel type (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 2,647,445 | 8.364 | 8.576 | 8.881 | +0.212 | +0.306 | 4.923 |

## Segmented by SEIFA quintile (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 487,022 | 8.016 | 8.215 | 8.535 | +0.199 | +0.320 | 4.855 |
| Q3 | 484,406 | 8.219 | 8.424 | 8.750 | +0.205 | +0.327 | 4.836 |
| Q4 | 481,950 | 8.343 | 8.548 | 8.839 | +0.205 | +0.291 | 4.920 |
| Q2 | 481,825 | 7.830 | 8.029 | 8.326 | +0.200 | +0.296 | 4.581 |
| Q5 | 480,223 | 9.266 | 9.420 | 9.781 | +0.154 | +0.361 | 5.333 |
| Unknown | 232,019 | 8.680 | 9.094 | 9.262 | +0.415 | +0.168 | 5.118 |

---

_Generated by `python -m fuel_pred.evaluate.compare_kfold`. v3.0
Phase 1 (spec §15.2). The "Mean" / "Stdev" / "Min" / "Max" rows
aggregate the per-fold metrics for the A-vs-B headline (and the
B-vs-B' table if Model B' was fit). Use stdev as the across-
fold variance signal: if |Mean Δ MAE| ≫ Stdev, the change is
robust; if comparable, fold-specific noise dominates._
