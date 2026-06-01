# Model A vs Model B vs Model B' — comparison report

        Generated: 2026-05-31 02:41:45 UTC
        Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e3_combined_temporal.parquet`
        Models:   `C:\repos\cauldnz\aus-fuel-forecaster\models_e3_combined_temporal/model_a.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e3_combined_temporal/model_b.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e3_combined_temporal/model_b_prime.pkl`

        Per spec §8.5: Model A uses lag + upstream + cal + ctx + stn + wx
        feature blocks. Model B adds the SA2 demographic block. **Both
        models train on identical rows** (those where every SA2 column
        is non-null) so the comparison isolates the augmentor's lift.

        Model B' (spec §13.6 Phase 1) extends Model B with the VENUE block (nearest-venue distance / capacity / type, venues-within-5km count, and `cal_is_pre_long_weekend`). It's the additive sanity check asking: do venue-distance features add lift over Model B, or do they just re-encode `stn_is_metro` and other existing features? Identical hyperparameters and identical training rows.

- **Negative `Δ MAE` = Model B beats Model A** (augmentor adds value)
        - **Negative `Δ MAE (B' vs B)` = venue features add lift** beyond Model B
        - All metrics in cents/L except MAPE (in %)

## Headline (overall) — A vs B

| Fold | n | MAE A | MAE B | Δ MAE | RMSE A | RMSE B | MAPE A | MAPE B | Δ MAPE |
|------|--:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|
| test_normal | 849,334 | 6.222 | 6.181 | -0.041 | 10.728 | 10.720 | 3.274 | 3.254 | -0.020 |
| test_crisis | 172,858 | 12.967 | 12.647 | -0.320 | 18.393 | 17.812 | 5.882 | 5.767 | -0.115 |

## Headline (overall) — B vs B' (venue-block additive sanity check)

| Fold | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| test_normal | 849,334 | 6.181 | 6.872 | +0.690 | 11.341 | 3.628 | +0.650 |
| test_crisis | 172,858 | 12.647 | 13.128 | +0.481 | 18.052 | 6.021 | +0.161 |

## Segmented by Metro / regional

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 846,336 | 6.226 | 6.185 | 6.877 | -0.041 | +0.691 | 3.631 |
| True | 2,998 | 5.059 | 5.089 | 5.475 | +0.030 | +0.386 | 2.929 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 172,336 | 12.970 | 12.650 | 13.132 | -0.319 | +0.482 | 6.024 |
| True | 522 | 11.965 | 11.573 | 11.661 | -0.392 | +0.088 | 5.246 |

## Segmented by Brand (top 8 + Other)

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 203,257 | 6.614 | 6.531 | 7.338 | -0.083 | +0.807 | 3.812 |
| Other | 163,936 | 5.288 | 5.299 | 5.946 | +0.012 | +0.647 | 3.207 |
| 7-Eleven | 120,966 | 8.355 | 8.163 | 9.034 | -0.193 | +0.871 | 4.694 |
| Metro | 94,517 | 5.650 | 5.744 | 6.180 | +0.094 | +0.436 | 3.411 |
| BP | 88,670 | 6.441 | 6.425 | 7.249 | -0.016 | +0.824 | 3.763 |
| Independent | 68,994 | 4.689 | 4.731 | 5.278 | +0.042 | +0.547 | 2.843 |
| Coles Express | 50,686 | 6.058 | 5.948 | 6.435 | -0.110 | +0.486 | 3.246 |
| Speedway | 29,238 | 5.975 | 5.987 | 6.492 | +0.012 | +0.505 | 3.589 |
| United | 29,070 | 5.231 | 5.186 | 5.860 | -0.045 | +0.674 | 3.148 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 39,875 | 13.171 | 12.807 | 13.400 | -0.364 | +0.593 | 6.121 |
| Other | 35,702 | 12.476 | 12.267 | 12.616 | -0.209 | +0.350 | 5.710 |
| 7-Eleven | 21,900 | 12.849 | 12.337 | 12.936 | -0.512 | +0.599 | 6.318 |
| BP | 18,770 | 13.427 | 13.061 | 13.600 | -0.366 | +0.539 | 6.188 |
| Metro | 17,740 | 11.322 | 11.053 | 11.306 | -0.269 | +0.252 | 5.334 |
| Independent | 16,184 | 15.159 | 14.683 | 15.137 | -0.476 | +0.454 | 6.548 |
| Shell | 9,321 | 13.940 | 13.708 | 14.309 | -0.233 | +0.601 | 6.436 |
| Reddy Express | 8,212 | 12.021 | 12.037 | 12.579 | +0.017 | +0.541 | 5.849 |
| United | 5,154 | 12.141 | 12.009 | 12.376 | -0.132 | +0.367 | 5.773 |

## Segmented by Fuel type

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 849,334 | 6.222 | 6.181 | 6.872 | -0.041 | +0.690 | 3.628 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 172,858 | 12.967 | 12.647 | 13.128 | -0.320 | +0.481 | 6.021 |

## Segmented by SEIFA quintile

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 159,692 | 5.852 | 5.886 | 6.509 | +0.034 | +0.624 | 3.508 |
| Q4 | 156,121 | 6.252 | 6.194 | 6.879 | -0.058 | +0.685 | 3.638 |
| Q3 | 154,732 | 5.701 | 5.678 | 6.340 | -0.023 | +0.663 | 3.362 |
| Q5 | 153,650 | 7.741 | 7.534 | 8.330 | -0.207 | +0.797 | 4.309 |
| Q2 | 151,464 | 5.603 | 5.583 | 6.240 | -0.020 | +0.657 | 3.305 |
| Unknown | 73,675 | 6.159 | 6.262 | 7.012 | +0.103 | +0.750 | 3.676 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 32,137 | 12.926 | 12.629 | 13.066 | -0.297 | +0.437 | 5.989 |
| Q4 | 31,839 | 12.485 | 12.201 | 12.635 | -0.284 | +0.434 | 5.856 |
| Q5 | 31,650 | 12.625 | 12.215 | 12.757 | -0.410 | +0.543 | 6.046 |
| Q3 | 31,643 | 13.165 | 12.817 | 13.285 | -0.348 | +0.469 | 5.994 |
| Q2 | 31,555 | 13.880 | 13.491 | 13.956 | -0.389 | +0.465 | 6.226 |
| Unknown | 14,034 | 12.423 | 12.397 | 13.007 | -0.026 | +0.609 | 6.016 |

## Feature importance

### Model A — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 31,602,822 | 53.60 | 3,755 |
| 2 | `lag_price_2` | lag | 8,766,564 | 14.87 | 2,159 |
| 3 | `days_since_last_price_change` | lag | 3,671,691 | 6.23 | 2,636 |
| 4 | `upstream_brent_aud_lag_0` | upstream | 1,350,377 | 2.29 | 673 |
| 5 | `price_minus_28d_min` | lag | 1,323,789 | 2.25 | 1,231 |
| 6 | `upstream_brent_lag_0` | upstream | 1,269,976 | 2.15 | 555 |
| 7 | `lag_price_3` | lag | 1,262,691 | 2.14 | 823 |
| 8 | `cal_day_of_month` | cal | 1,136,546 | 1.93 | 1,353 |
| 9 | `price_minus_28d_max` | lag | 1,074,671 | 1.82 | 1,073 |
| 10 | `lag_price_28` | lag | 829,867 | 1.41 | 993 |
| 11 | `upstream_brent_lag_1` | upstream | 672,098 | 1.14 | 366 |
| 12 | `roll_price_mean_28` | lag | 613,659 | 1.04 | 925 |
| 13 | `upstream_brent_lag_14` | upstream | 564,984 | 0.96 | 594 |
| 14 | `upstream_brent_aud_lag_14` | upstream | 503,333 | 0.85 | 545 |
| 15 | `upstream_brent_lag_3` | upstream | 468,297 | 0.79 | 319 |
| 16 | `cal_year` | cal | 458,736 | 0.78 | 331 |
| 17 | `upstream_audusd_lag_0` | upstream | 322,867 | 0.55 | 599 |
| 18 | `xfuel_dl_price_lag_0` | lag | 319,383 | 0.54 | 435 |
| 19 | `upstream_brent_lag_7` | upstream | 308,485 | 0.52 | 281 |
| 20 | `stn_brand_raw` | stn | 214,221 | 0.36 | 526 |

### Model B — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 35,086,010 | 57.44 | 5,873 |
| 2 | `lag_price_2` | lag | 5,161,030 | 8.45 | 2,680 |
| 3 | `days_since_last_price_change` | lag | 3,916,808 | 6.41 | 4,158 |
| 4 | `lag_price_3` | lag | 1,454,153 | 2.38 | 1,488 |
| 5 | `price_minus_28d_min` | lag | 1,336,507 | 2.19 | 1,961 |
| 6 | `upstream_brent_lag_0` | upstream | 1,198,719 | 1.96 | 856 |
| 7 | `price_minus_28d_max` | lag | 1,137,895 | 1.86 | 1,589 |
| 8 | `cal_day_of_month` | cal | 1,076,021 | 1.76 | 1,994 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 837,905 | 1.37 | 761 |
| 10 | `lag_price_28` | lag | 742,233 | 1.22 | 1,461 |
| 11 | `upstream_brent_lag_3` | upstream | 729,513 | 1.19 | 716 |
| 12 | `upstream_brent_lag_7` | upstream | 710,454 | 1.16 | 644 |
| 13 | `upstream_brent_lag_1` | upstream | 658,883 | 1.08 | 650 |
| 14 | `roll_price_mean_7` | lag | 656,654 | 1.08 | 842 |
| 15 | `upstream_brent_lag_14` | upstream | 636,669 | 1.04 | 975 |
| 16 | `roll_price_mean_28` | lag | 606,743 | 0.99 | 1,414 |
| 17 | `upstream_brent_aud_lag_14` | upstream | 573,852 | 0.94 | 775 |
| 18 | `cal_year` | cal | 561,589 | 0.92 | 536 |
| 19 | `upstream_audusd_lag_0` | upstream | 350,427 | 0.57 | 1,028 |
| 20 | `upstream_brent_aud_lag_7` | upstream | 291,307 | 0.48 | 576 |

### Model B' — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 30,679,063 | 51.43 | 5,074 |
| 2 | `lag_price_2` | lag | 8,101,355 | 13.58 | 2,647 |
| 3 | `days_since_last_price_change` | lag | 3,423,746 | 5.74 | 3,675 |
| 4 | `lag_price_3` | lag | 1,614,044 | 2.71 | 1,167 |
| 5 | `price_minus_28d_min` | lag | 1,479,015 | 2.48 | 1,818 |
| 6 | `upstream_brent_lag_0` | upstream | 1,441,481 | 2.42 | 857 |
| 7 | `price_minus_28d_max` | lag | 1,250,042 | 2.10 | 1,637 |
| 8 | `upstream_brent_aud_lag_0` | upstream | 1,195,757 | 2.00 | 872 |
| 9 | `cal_day_of_month` | cal | 1,184,851 | 1.99 | 1,923 |
| 10 | `upstream_brent_lag_1` | upstream | 997,375 | 1.67 | 570 |
| 11 | `lag_price_28` | lag | 773,503 | 1.30 | 1,277 |
| 12 | `roll_price_mean_28` | lag | 613,762 | 1.03 | 1,265 |
| 13 | `upstream_brent_lag_3` | upstream | 587,901 | 0.99 | 629 |
| 14 | `roll_price_mean_7` | lag | 571,410 | 0.96 | 759 |
| 15 | `upstream_brent_lag_14` | upstream | 523,817 | 0.88 | 793 |
| 16 | `cal_year` | cal | 496,414 | 0.83 | 536 |
| 17 | `stn_nearest_venue_km` | venue | 388,501 | 0.65 | 743 |
| 18 | `upstream_brent_lag_7` | upstream | 381,469 | 0.64 | 465 |
| 19 | `upstream_audusd_lag_0` | upstream | 354,988 | 0.60 | 975 |
| 20 | `xfuel_dl_price_lag_0` | lag | 344,936 | 0.58 | 593 |

### Where SA2 features rank in Model B

| SA2 feature | Rank in B | Gain | Gain % |
|-------------|----------:|-----:|-------:|
| `sa2_pct_drive_to_work` | 33 | 94,239 | 0.15 |
| `sa2_dss_parenting_payment_partnered_recipients` | 45 | 31,627 | 0.05 |
| `sa2_seifa_ieo_score` | 47 | 23,453 | 0.04 |
| `sa2_pct_aged_65_plus` | 53 | 13,823 | 0.02 |
| `sa2_dss_carer_allowance_recipients` | 54 | 13,275 | 0.02 |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | 56 | 11,751 | 0.02 |
| `sa2_median_age` | 57 | 10,482 | 0.02 |
| `sa2_seifa_irsd_score` | 58 | 10,466 | 0.02 |
| `sa2_pct_renters` | 59 | 10,399 | 0.02 |
| `sa2_pct_employed_full_time` | 60 | 9,198 | 0.02 |
| `sa2_median_household_income_weekly` | 61 | 8,305 | 0.01 |
| `sa2_dss_carer_payment_recipients` | 64 | 5,959 | 0.01 |
| `sa2_pct_one_parent_family` | 65 | 5,285 | 0.01 |
| `sa2_motor_vehicles_per_dwelling` | 67 | 4,661 | 0.01 |
| `sa2_total_population` | 78 | 89 | 0.00 |

### Where VENUE-block features rank in Model B' (spec §13.6 Phase 1)

| Venue feature | Rank in B' | Gain | Gain % |
|---------------|-----------:|-----:|-------:|
| `stn_nearest_venue_km` | 17 | 388,501 | 0.65 |
| `stn_nearest_venue_capacity` | 53 | 8,077 | 0.01 |
| `stn_nearest_venue_type` | 69 | 741 | 0.00 |
| `stn_n_venues_within_5km` | 71 | 586 | 0.00 |
| `cal_is_pre_long_weekend` | 72 | 565 | 0.00 |

## SA2 ↔ non-SA2 feature correlation

_Pearson `r` between each SA2 feature and the most-correlated non-SA2 numeric feature, computed on a sample of 100,000 rows. Categoricals are excluded (Pearson is numeric-only). High correlation (`|r| > 0.5`) flags features the model could already infer from existing inputs._

### Top 3 correlated non-SA2 features per SA2 feature

| SA2 feature | #1 (|r|) | #2 (|r|) | #3 (|r|) |
|-------------|----------|----------|----------|
| `sa2_total_population` | `wx_temp_min_c_t3` (+0.510) ⚠️ | `wx_temp_min_c_t7` (+0.484) | `wx_wind_speed_max_kmh_t6` (+0.478) |
| `sa2_median_age` | `wx_temp_min_c_t4` (-0.577) ⚠️ | `wx_temp_min_c_t3` (-0.563) ⚠️ | `stn_competitors_within_5km` (-0.542) ⚠️ |
| `sa2_median_household_income_weekly` | `wx_temp_min_c_t2` (+0.521) ⚠️ | `stn_nearest_venue_km` (-0.445) | `wx_temp_min_c_t3` (+0.443) |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t6` (-0.808) ⚠️ | `ctx_traffic_5km_radius_count` (-0.760) ⚠️ | `wx_temp_min_c_t2` (-0.749) ⚠️ |
| `sa2_motor_vehicles_per_dwelling` | `stn_competitors_within_5km` (-0.537) ⚠️ | `ctx_traffic_5km_radius_count` (-0.535) ⚠️ | `stn_competitors_within_2km` (-0.530) ⚠️ |
| `sa2_pct_renters` | `stn_competitors_within_2km` (+0.575) ⚠️ | `stn_competitors_within_5km` (+0.569) ⚠️ | `ctx_traffic_5km_radius_count` (+0.488) |
| `sa2_pct_employed_full_time` | `wx_temp_max_c_t2` (-0.506) ⚠️ | `wx_temp_max_c_t3` (-0.470) | `wx_wind_speed_max_kmh_t3` (+0.450) |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t4` (-0.573) ⚠️ | `wx_temp_min_c_t2` (-0.540) ⚠️ | `wx_temp_min_c_t3` (-0.515) ⚠️ |
| `sa2_pct_one_parent_family` | `ctx_traffic_5km_radius_count` (-0.467) | `wx_temp_min_c_t2` (-0.441) | `ctx_traffic_top1_lag_7` (-0.418) |
| `sa2_seifa_irsd_score` | `ctx_traffic_top1_lag_7` (+0.269) | `ctx_traffic_top1_lag_1` (+0.268) | `stn_nearest_venue_km` (-0.231) |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t2` (+0.612) ⚠️ | `wx_temp_min_c_t3` (+0.582) ⚠️ | `wx_temp_min_c_t7` (+0.561) ⚠️ |
| `sa2_dss_parenting_payment_partnered_recipients` | `stn_competitors_within_5km` (+0.487) | `stn_competitors_within_2km` (+0.351) | `ctx_traffic_top2_lag_1` (+0.280) |
| `sa2_dss_carer_payment_recipients` | `stn_competitors_within_5km` (+0.408) | `wx_wind_speed_max_kmh_t3` (-0.377) | `wx_wind_speed_max_kmh_t7` (-0.355) |
| `sa2_dss_carer_allowance_recipients` | `wx_wind_speed_max_kmh_t3` (-0.444) | `wx_wind_speed_max_kmh_t2` (-0.436) | `stn_competitors_within_5km` (+0.348) |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | `stn_competitors_within_5km` (+0.665) ⚠️ | `ctx_traffic_5km_radius_count` (+0.502) ⚠️ | `stn_competitors_within_2km` (+0.489) |

### High correlations (|r| ≥ 0.5)

| SA2 feature | Non-SA2 feature | r | Block |
|-------------|------------------|--:|-------|
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t6` | -0.808 | wx |
| `sa2_pct_drive_to_work` | `ctx_traffic_5km_radius_count` | -0.760 | ctx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t2` | -0.749 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t7` | -0.714 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t5` | -0.682 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t4` | -0.668 | wx |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | `stn_competitors_within_5km` | +0.665 | stn |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t3` | -0.659 | wx |
| `sa2_pct_drive_to_work` | `stn_competitors_within_5km` | -0.622 | stn |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t2` | +0.612 | wx |
| `sa2_pct_drive_to_work` | `stn_nearest_venue_km` | +0.588 | venue |
| `sa2_pct_drive_to_work` | `ctx_traffic_top1_lag_7` | -0.587 | ctx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top1_lag_1` | -0.586 | ctx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top3_lag_1` | -0.586 | ctx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top3_lag_7` | -0.586 | ctx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t2` | -0.585 | wx |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t3` | +0.582 | wx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top2_lag_1` | -0.579 | ctx |
| `sa2_median_age` | `wx_temp_min_c_t4` | -0.577 | wx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top2_lag_7` | -0.576 | ctx |
| `sa2_pct_renters` | `stn_competitors_within_2km` | +0.575 | stn |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t4` | -0.573 | wx |
| `sa2_pct_renters` | `stn_competitors_within_5km` | +0.569 | stn |
| `sa2_median_age` | `wx_temp_min_c_t3` | -0.563 | wx |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t7` | +0.561 | wx |
| `sa2_seifa_ieo_score` | `ctx_traffic_5km_radius_count` | +0.547 | ctx |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t6` | +0.546 | wx |
| `sa2_median_age` | `stn_competitors_within_5km` | -0.542 | stn |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t2` | -0.540 | wx |
| `sa2_motor_vehicles_per_dwelling` | `stn_competitors_within_5km` | -0.537 | stn |
| `sa2_motor_vehicles_per_dwelling` | `ctx_traffic_5km_radius_count` | -0.535 | ctx |
| `sa2_motor_vehicles_per_dwelling` | `stn_competitors_within_2km` | -0.530 | stn |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t4` | +0.525 | wx |
| `sa2_median_household_income_weekly` | `wx_temp_min_c_t2` | +0.521 | wx |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t3` | -0.515 | wx |
| `sa2_median_age` | `wx_temp_min_c_t2` | -0.514 | wx |
| `sa2_total_population` | `wx_temp_min_c_t3` | +0.510 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t3` | -0.507 | wx |
| `sa2_pct_employed_full_time` | `wx_temp_max_c_t2` | -0.506 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t6` | -0.504 | wx |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | `ctx_traffic_5km_radius_count` | +0.502 | ctx |

---

_Generated by `python -m fuel_pred.evaluate.compare`. Re-run after
`make train` to refresh; predictions are read from
`models/predictions_*.parquet` rather than re-loading the pickles
for speed._
