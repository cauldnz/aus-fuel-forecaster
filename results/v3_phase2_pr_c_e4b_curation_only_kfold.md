# Model A vs Model B vs Model B' — k-fold CV comparison report (spec §15.2)

Generated: 2026-06-04 11:59:40 UTC
Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e4b_curation_only.parquet`
K-fold models root: `C:\repos\cauldnz\aus-fuel-forecaster\models_kfold_pr_c_e4b_curation_only`
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
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.219 | 6.325 | +0.106 | 10.422 | 10.473 | 5.040 | 5.132 | +0.092 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 8.799 | 9.179 | +0.380 | 13.862 | 14.488 | 5.149 | 5.362 | +0.213 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 13.331 | 14.095 | +0.764 | 18.771 | 19.359 | 6.897 | 7.292 | +0.396 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 6.954 | 6.970 | +0.017 | 12.290 | 12.294 | 3.425 | 3.432 | +0.006 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.181 | 4.443 | +0.262 | 8.157 | 8.436 | 2.203 | 2.347 | +0.144 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 9.573 | 10.305 | +0.731 | 13.435 | 13.922 | 4.861 | 5.244 | +0.383 |
| **Mean** | — | 2,647,445 | 8.176 | 8.553 | +0.377 | 12.823 | 13.162 | 4.596 | 4.802 | +0.206 |
| Stdev | — | — | 2.893 | 3.123 | +0.286 | 3.283 | 3.441 | 1.470 | 1.567 | +0.144 |
| Min | — | — | 4.181 | 4.443 | +0.017 | 8.157 | 8.436 | 2.203 | 2.347 | +0.006 |
| Max | — | — | 13.331 | 14.095 | +0.764 | 18.771 | 19.359 | 6.897 | 7.292 | +0.396 |

## Headline — B vs B' (venue-block additive sanity check, per-fold + aggregate)

| Fold | Test window | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|-------------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| fold_1 | 2020-05-01 → 2021-04-30 | 392,049 | 6.325 | 6.174 | -0.151 | 10.424 | 4.983 | -0.046 |
| fold_2 | 2021-05-01 → 2022-04-30 | 411,168 | 9.179 | 8.847 | -0.332 | 14.169 | 5.178 | +0.048 |
| fold_3 | 2022-05-01 → 2023-04-30 | 488,617 | 14.095 | 13.138 | -0.957 | 18.224 | 6.793 | -0.193 |
| fold_4 | 2023-05-01 → 2024-04-30 | 477,729 | 6.970 | 6.849 | -0.122 | 12.000 | 3.373 | -0.105 |
| fold_5 | 2024-05-01 → 2025-04-30 | 451,063 | 4.443 | 4.438 | -0.004 | 8.472 | 2.352 | +0.257 |
| fold_6 | 2025-05-01 → 2026-04-29 | 426,819 | 10.305 | 9.361 | -0.944 | 13.582 | 4.697 | -0.213 |
| **Mean** | — | 2,647,445 | 8.553 | 8.134 | -0.418 | 12.812 | 4.563 | -0.042 |
| Stdev | — | — | 3.123 | 2.775 | +0.388 | 3.080 | 1.405 | +0.160 |
| Min | — | — | 4.443 | 4.438 | -0.957 | 8.472 | 2.352 | -0.213 |
| Max | — | — | 14.095 | 13.138 | -0.004 | 18.224 | 6.793 | +0.257 |

## Segmented by Metro / regional (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 2,638,213 | 8.258 | 8.640 | 8.214 | +0.381 | -0.425 | 4.563 |
| True | 9,232 | 8.325 | 8.667 | 8.212 | +0.342 | -0.455 | 4.393 |

## Segmented by Brand (across all folds combined; top 8 + Other)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 623,474 | 8.664 | 9.080 | 8.619 | +0.416 | -0.461 | 4.687 |
| Other | 432,131 | 6.933 | 7.317 | 6.977 | +0.384 | -0.339 | 3.900 |
| 7-Eleven | 359,079 | 9.766 | 10.145 | 9.575 | +0.379 | -0.569 | 5.321 |
| Metro | 288,873 | 7.481 | 7.766 | 7.342 | +0.285 | -0.424 | 4.336 |
| BP | 287,258 | 8.664 | 9.097 | 8.672 | +0.434 | -0.425 | 4.739 |
| Independent | 252,342 | 7.344 | 7.709 | 7.386 | +0.365 | -0.323 | 4.092 |
| Coles Express | 189,598 | 8.893 | 9.201 | 8.809 | +0.308 | -0.392 | 4.896 |
| United | 86,807 | 7.566 | 7.900 | 7.598 | +0.334 | -0.302 | 4.418 |
| Shell | 66,851 | 9.571 | 10.147 | 9.660 | +0.577 | -0.487 | 4.954 |
| Speedway | 41,300 | 6.428 | 6.766 | 6.371 | +0.338 | -0.395 | 4.107 |
| Reddy Express | 19,732 | 10.567 | 11.006 | 10.360 | +0.440 | -0.647 | 5.126 |

## Segmented by Fuel type (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 2,647,445 | 8.258 | 8.640 | 8.214 | +0.381 | -0.426 | 4.562 |

## Segmented by SEIFA quintile (across all folds combined)

### all_folds

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 487,022 | 7.900 | 8.292 | 7.946 | +0.393 | -0.347 | 4.532 |
| Q3 | 484,406 | 8.086 | 8.513 | 8.114 | +0.427 | -0.399 | 4.489 |
| Q4 | 481,950 | 8.235 | 8.602 | 8.182 | +0.367 | -0.420 | 4.564 |
| Q2 | 481,825 | 7.680 | 8.117 | 7.750 | +0.437 | -0.367 | 4.263 |
| Q5 | 480,223 | 9.211 | 9.487 | 9.013 | +0.276 | -0.474 | 4.928 |
| Unknown | 232,019 | 8.649 | 9.041 | 8.362 | +0.392 | -0.679 | 4.639 |

---

_Generated by `python -m fuel_pred.evaluate.compare_kfold`. v3.0
Phase 1 (spec §15.2). The "Mean" / "Stdev" / "Min" / "Max" rows
aggregate the per-fold metrics for the A-vs-B headline (and the
B-vs-B' table if Model B' was fit). Use stdev as the across-
fold variance signal: if |Mean Δ MAE| ≫ Stdev, the change is
robust; if comparable, fold-specific noise dominates._
