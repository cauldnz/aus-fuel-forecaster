# Model A vs Model B vs Model B' — comparison report

        Generated: 2026-06-01 01:39:07 UTC
        Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e5_dss_temporal_plus_curation.parquet`
        Models:   `C:\repos\cauldnz\aus-fuel-forecaster\models_e5_dss_temporal_plus_curation/model_a.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e5_dss_temporal_plus_curation/model_b.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e5_dss_temporal_plus_curation/model_b_prime.pkl`

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
| test_normal | 849,334 | 6.373 | 6.685 | +0.312 | 10.953 | 11.207 | 3.352 | 3.525 | +0.173 |
| test_crisis | 172,858 | 13.616 | 13.466 | -0.150 | 19.054 | 18.711 | 6.181 | 6.128 | -0.053 |

## Headline (overall) — B vs B' (venue-block additive sanity check)

| Fold | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| test_normal | 849,334 | 6.685 | 6.546 | -0.139 | 11.065 | 3.431 | +0.173 |
| test_crisis | 172,858 | 13.466 | 13.687 | +0.221 | 19.088 | 6.194 | +0.071 |

## Segmented by Metro / regional

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 846,336 | 6.377 | 6.690 | 6.551 | +0.313 | -0.139 | 3.433 |
| True | 2,998 | 5.209 | 5.286 | 5.138 | +0.077 | -0.148 | 2.721 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 172,336 | 13.618 | 13.470 | 13.691 | -0.148 | +0.220 | 6.196 |
| True | 522 | 12.811 | 12.212 | 12.505 | -0.599 | +0.293 | 5.578 |

## Segmented by Brand (top 8 + Other)

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 203,257 | 6.787 | 7.186 | 7.002 | +0.399 | -0.184 | 3.608 |
| Other | 163,936 | 5.401 | 5.675 | 5.492 | +0.274 | -0.183 | 2.944 |
| 7-Eleven | 120,966 | 8.585 | 8.919 | 8.867 | +0.334 | -0.052 | 4.573 |
| Metro | 94,517 | 5.739 | 5.924 | 5.829 | +0.185 | -0.095 | 3.203 |
| BP | 88,670 | 6.652 | 7.055 | 6.856 | +0.403 | -0.198 | 3.530 |
| Independent | 68,994 | 4.802 | 5.080 | 4.863 | +0.278 | -0.218 | 2.601 |
| Coles Express | 50,686 | 6.176 | 6.432 | 6.540 | +0.256 | +0.108 | 3.277 |
| Speedway | 29,238 | 6.087 | 6.271 | 6.150 | +0.184 | -0.120 | 3.386 |
| United | 29,070 | 5.327 | 5.602 | 5.435 | +0.275 | -0.167 | 2.898 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 39,875 | 13.894 | 13.816 | 13.938 | -0.078 | +0.121 | 6.277 |
| Other | 35,702 | 13.155 | 12.908 | 13.155 | -0.247 | +0.246 | 5.875 |
| 7-Eleven | 21,900 | 13.494 | 13.148 | 13.458 | -0.346 | +0.310 | 6.502 |
| BP | 18,770 | 14.011 | 13.998 | 14.176 | -0.013 | +0.178 | 6.360 |
| Metro | 17,740 | 11.737 | 11.576 | 11.939 | -0.161 | +0.363 | 5.544 |
| Independent | 16,184 | 15.707 | 15.671 | 15.957 | -0.035 | +0.285 | 6.828 |
| Shell | 9,321 | 14.706 | 14.578 | 14.777 | -0.128 | +0.199 | 6.550 |
| Reddy Express | 8,212 | 12.915 | 12.811 | 12.946 | -0.104 | +0.135 | 5.939 |
| United | 5,154 | 12.782 | 12.659 | 12.727 | -0.123 | +0.068 | 5.850 |

## Segmented by Fuel type

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 849,334 | 6.373 | 6.685 | 6.546 | +0.312 | -0.139 | 3.431 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 172,858 | 13.616 | 13.466 | 13.687 | -0.150 | +0.221 | 6.194 |

## Segmented by SEIFA quintile

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 159,692 | 5.979 | 6.273 | 6.134 | +0.294 | -0.139 | 3.283 |
| Q4 | 156,121 | 6.412 | 6.674 | 6.558 | +0.261 | -0.116 | 3.442 |
| Q3 | 154,732 | 5.839 | 6.129 | 5.992 | +0.290 | -0.137 | 3.152 |
| Q5 | 153,650 | 7.983 | 8.284 | 8.151 | +0.301 | -0.133 | 4.186 |
| Q2 | 151,464 | 5.717 | 6.056 | 5.876 | +0.339 | -0.180 | 3.087 |
| Unknown | 73,675 | 6.255 | 6.731 | 6.610 | +0.475 | -0.120 | 3.441 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 32,137 | 13.576 | 13.379 | 13.645 | -0.197 | +0.266 | 6.167 |
| Q4 | 31,839 | 13.172 | 12.932 | 13.176 | -0.240 | +0.244 | 6.023 |
| Q5 | 31,650 | 13.296 | 13.096 | 13.277 | -0.201 | +0.181 | 6.211 |
| Q3 | 31,643 | 13.797 | 13.670 | 13.923 | -0.127 | +0.253 | 6.201 |
| Q2 | 31,555 | 14.492 | 14.432 | 14.628 | -0.060 | +0.196 | 6.438 |
| Unknown | 14,034 | 13.059 | 13.086 | 13.218 | +0.027 | +0.132 | 6.039 |

## Feature importance

### Model A — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 32,224,072 | 52.56 | 5,955 |
| 2 | `lag_price_2` | lag | 9,008,490 | 14.69 | 3,369 |
| 3 | `days_since_last_price_change` | lag | 3,870,998 | 6.31 | 4,134 |
| 4 | `upstream_brent_lag_0` | upstream | 1,464,467 | 2.39 | 1,014 |
| 5 | `lag_price_3` | lag | 1,406,060 | 2.29 | 1,530 |
| 6 | `price_minus_28d_min` | lag | 1,364,600 | 2.23 | 1,945 |
| 7 | `price_minus_28d_max` | lag | 1,166,075 | 1.90 | 1,821 |
| 8 | `cal_day_of_month` | cal | 1,149,506 | 1.87 | 2,106 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 1,147,787 | 1.87 | 1,070 |
| 10 | `upstream_brent_lag_14` | upstream | 772,259 | 1.26 | 1,071 |
| 11 | `lag_price_28` | lag | 755,051 | 1.23 | 1,543 |
| 12 | `upstream_brent_lag_3` | upstream | 651,778 | 1.06 | 695 |
| 13 | `roll_price_mean_28` | lag | 639,272 | 1.04 | 1,431 |
| 14 | `upstream_brent_lag_1` | upstream | 620,831 | 1.01 | 575 |
| 15 | `upstream_brent_aud_lag_14` | upstream | 494,249 | 0.81 | 864 |
| 16 | `cal_year` | cal | 470,084 | 0.77 | 525 |
| 17 | `upstream_audusd_lag_0` | upstream | 386,377 | 0.63 | 1,148 |
| 18 | `xfuel_dl_price_lag_0` | lag | 297,961 | 0.49 | 621 |
| 19 | `upstream_brent_lag_7` | upstream | 292,396 | 0.48 | 592 |
| 20 | `stn_brand_raw` | stn | 264,758 | 0.43 | 1,231 |

### Model B — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 30,345,503 | 51.86 | 4,018 |
| 2 | `lag_price_2` | lag | 5,962,529 | 10.19 | 1,668 |
| 3 | `lag_price_3` | lag | 4,702,322 | 8.04 | 1,640 |
| 4 | `days_since_last_price_change` | lag | 3,590,927 | 6.14 | 3,086 |
| 5 | `price_minus_28d_min` | lag | 1,466,710 | 2.51 | 1,413 |
| 6 | `upstream_brent_lag_0` | upstream | 1,255,177 | 2.14 | 677 |
| 7 | `cal_day_of_month` | cal | 1,132,072 | 1.93 | 1,462 |
| 8 | `price_minus_28d_max` | lag | 1,077,463 | 1.84 | 1,137 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 954,251 | 1.63 | 621 |
| 10 | `lag_price_28` | lag | 728,009 | 1.24 | 926 |
| 11 | `upstream_brent_lag_14` | upstream | 667,117 | 1.14 | 557 |
| 12 | `roll_price_mean_28` | lag | 613,535 | 1.05 | 981 |
| 13 | `upstream_brent_lag_1` | upstream | 612,147 | 1.05 | 378 |
| 14 | `upstream_brent_lag_3` | upstream | 520,737 | 0.89 | 405 |
| 15 | `cal_year` | cal | 467,311 | 0.80 | 348 |
| 16 | `upstream_brent_aud_lag_14` | upstream | 461,645 | 0.79 | 576 |
| 17 | `upstream_audusd_lag_0` | upstream | 347,451 | 0.59 | 725 |
| 18 | `upstream_brent_lag_7` | upstream | 311,203 | 0.53 | 341 |
| 19 | `xfuel_dl_price_lag_0` | lag | 297,246 | 0.51 | 387 |
| 20 | `roll_price_mean_7` | lag | 293,685 | 0.50 | 401 |

### Model B' — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 33,671,564 | 57.50 | 3,747 |
| 2 | `lag_price_2` | lag | 4,105,886 | 7.01 | 1,362 |
| 3 | `days_since_last_price_change` | lag | 3,415,165 | 5.83 | 2,746 |
| 4 | `lag_price_3` | lag | 3,046,747 | 5.20 | 1,352 |
| 5 | `upstream_brent_lag_0` | upstream | 1,507,031 | 2.57 | 659 |
| 6 | `price_minus_28d_min` | lag | 1,393,104 | 2.38 | 1,203 |
| 7 | `price_minus_28d_max` | lag | 1,127,023 | 1.92 | 1,111 |
| 8 | `cal_day_of_month` | cal | 1,036,820 | 1.77 | 1,294 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 879,273 | 1.50 | 463 |
| 10 | `lag_price_28` | lag | 737,879 | 1.26 | 870 |
| 11 | `upstream_brent_lag_7` | upstream | 734,203 | 1.25 | 378 |
| 12 | `upstream_brent_aud_lag_14` | upstream | 671,621 | 1.15 | 546 |
| 13 | `roll_price_mean_28` | lag | 599,460 | 1.02 | 883 |
| 14 | `upstream_brent_lag_14` | upstream | 575,897 | 0.98 | 631 |
| 15 | `upstream_brent_lag_3` | upstream | 566,435 | 0.97 | 373 |
| 16 | `upstream_brent_lag_1` | upstream | 457,497 | 0.78 | 294 |
| 17 | `cal_year` | cal | 434,203 | 0.74 | 398 |
| 18 | `xfuel_dl_price_lag_0` | lag | 322,587 | 0.55 | 467 |
| 19 | `upstream_audusd_lag_0` | upstream | 313,382 | 0.54 | 632 |
| 20 | `stn_nearest_venue_km` | venue | 262,799 | 0.45 | 452 |

### Where SA2 features rank in Model B

| SA2 feature | Rank in B | Gain | Gain % |
|-------------|----------:|-----:|-------:|
| `sa2_pct_drive_to_work` | 33 | 83,528 | 0.14 |
| `sa2_erp_population_density_per_km2` | 38 | 48,949 | 0.08 |
| `sa2_dss_parenting_payment_partnered_recipients` | 42 | 32,168 | 0.05 |
| `sa2_seifa_ieo_score` | 48 | 15,296 | 0.03 |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | 52 | 9,804 | 0.02 |
| `sa2_pct_jobseeker_recipients` | 53 | 9,649 | 0.02 |
| `sa2_erp_median_age` | 54 | 9,597 | 0.02 |
| `sa2_pct_aged_65_plus` | 55 | 8,282 | 0.01 |
| `sa2_dss_carer_payment_recipients` | 56 | 8,000 | 0.01 |
| `sa2_dss_carer_allowance_recipients` | 61 | 4,707 | 0.01 |
| `sa2_pct_employed_full_time` | 62 | 4,496 | 0.01 |
| `sa2_pct_renters` | 63 | 4,385 | 0.01 |
| `sa2_welfare_density_index` | 64 | 4,045 | 0.01 |
| `sa2_pct_age_pension_recipients` | 65 | 3,934 | 0.01 |
| `sa2_median_age` | 67 | 3,674 | 0.01 |
| `sa2_motor_vehicles_per_dwelling` | 69 | 3,524 | 0.01 |
| `sa2_pct_one_parent_family` | 70 | 3,123 | 0.01 |
| `sa2_median_household_income_weekly` | 71 | 3,013 | 0.01 |
| `sa2_total_population` | 73 | 2,695 | 0.00 |
| `sa2_seifa_irsd_score` | 74 | 2,257 | 0.00 |
| `sa2_erp_population_65_plus` | 75 | 1,095 | 0.00 |

### Where VENUE-block features rank in Model B' (spec §13.6 Phase 1)

| Venue feature | Rank in B' | Gain | Gain % |
|---------------|-----------:|-----:|-------:|
| `stn_nearest_venue_km` | 20 | 262,799 | 0.45 |
| `stn_nearest_venue_capacity` | 57 | 5,128 | 0.01 |
| `stn_n_venues_within_5km` | 76 | 202 | 0.00 |
| `stn_nearest_venue_type` | 80 | 60 | 0.00 |
| `cal_is_pre_long_weekend` | 94 | 0 | 0.00 |

## SA2 ↔ non-SA2 feature correlation

_Pearson `r` between each SA2 feature and the most-correlated non-SA2 numeric feature, computed on a sample of 100,000 rows. Categoricals are excluded (Pearson is numeric-only). High correlation (`|r| > 0.5`) flags features the model could already infer from existing inputs._

### Top 3 correlated non-SA2 features per SA2 feature

| SA2 feature | #1 (|r|) | #2 (|r|) | #3 (|r|) |
|-------------|----------|----------|----------|
| `sa2_total_population` | `stn_competitors_within_5km` (+0.387) | `stn_nearest_venue_km` (-0.380) | `wx_temp_min_c_t3` (+0.358) |
| `sa2_median_age` | `wx_temp_min_c_t4` (-0.577) ⚠️ | `wx_temp_min_c_t3` (-0.563) ⚠️ | `stn_competitors_within_5km` (-0.516) ⚠️ |
| `sa2_median_household_income_weekly` | `wx_temp_min_c_t2` (+0.521) ⚠️ | `stn_nearest_venue_km` (-0.467) | `ctx_traffic_top1_lag_7` (+0.461) |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t6` (-0.808) ⚠️ | `ctx_traffic_5km_radius_count` (-0.760) ⚠️ | `wx_temp_min_c_t2` (-0.749) ⚠️ |
| `sa2_motor_vehicles_per_dwelling` | `stn_competitors_within_5km` (-0.537) ⚠️ | `ctx_traffic_5km_radius_count` (-0.535) ⚠️ | `stn_competitors_within_2km` (-0.530) ⚠️ |
| `sa2_pct_renters` | `stn_competitors_within_2km` (+0.575) ⚠️ | `stn_competitors_within_5km` (+0.569) ⚠️ | `ctx_traffic_5km_radius_count` (+0.488) |
| `sa2_pct_employed_full_time` | `wx_temp_max_c_t2` (-0.506) ⚠️ | `wx_temp_max_c_t3` (-0.470) | `wx_wind_speed_max_kmh_t3` (+0.450) |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t4` (-0.573) ⚠️ | `wx_temp_min_c_t2` (-0.540) ⚠️ | `wx_temp_min_c_t3` (-0.515) ⚠️ |
| `sa2_pct_one_parent_family` | `ctx_traffic_5km_radius_count` (-0.467) | `wx_temp_min_c_t2` (-0.441) | `ctx_traffic_top1_lag_7` (-0.418) |
| `sa2_erp_population_65_plus` | `wx_wind_speed_max_kmh_t2` (-0.417) | `wx_wind_speed_max_kmh_t3` (-0.296) | `wx_precipitation_mm_t2` (-0.278) |
| `sa2_erp_median_age` | `wx_temp_min_c_t3` (-0.583) ⚠️ | `wx_temp_min_c_t4` (-0.579) ⚠️ | `stn_competitors_within_5km` (-0.523) ⚠️ |
| `sa2_pct_age_pension_recipients` | `wx_wind_speed_max_kmh_t7` (-0.386) | `wx_wind_speed_max_kmh_t3` (-0.331) | `wx_wind_speed_max_kmh_t2` (-0.308) |
| `sa2_pct_jobseeker_recipients` | `wx_temp_min_c_t2` (-0.395) | `wx_wind_speed_max_kmh_t3` (+0.392) | `wx_temp_min_c_t3` (-0.377) |
| `sa2_welfare_density_index` | `wx_temp_min_c_t2` (-0.454) | `wx_temp_min_c_t3` (-0.435) | `wx_temp_min_c_t4` (-0.420) |
| `sa2_erp_population_density_per_km2` | `ctx_traffic_5km_radius_count` (+0.782) ⚠️ | `stn_competitors_within_5km` (+0.773) ⚠️ | `wx_temp_min_c_t6` (+0.645) ⚠️ |
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
| `sa2_erp_population_density_per_km2` | `ctx_traffic_5km_radius_count` | +0.782 | ctx |
| `sa2_erp_population_density_per_km2` | `stn_competitors_within_5km` | +0.773 | stn |
| `sa2_pct_drive_to_work` | `ctx_traffic_5km_radius_count` | -0.760 | ctx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t2` | -0.749 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t7` | -0.714 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t5` | -0.682 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t4` | -0.668 | wx |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | `stn_competitors_within_5km` | +0.665 | stn |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t3` | -0.659 | wx |
| `sa2_erp_population_density_per_km2` | `wx_temp_min_c_t6` | +0.645 | wx |
| `sa2_erp_population_density_per_km2` | `wx_temp_min_c_t2` | +0.633 | wx |
| `sa2_pct_drive_to_work` | `stn_competitors_within_5km` | -0.622 | stn |
| `sa2_erp_population_density_per_km2` | `wx_temp_min_c_t3` | +0.612 | wx |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t2` | +0.612 | wx |
| `sa2_pct_drive_to_work` | `stn_nearest_venue_km` | +0.588 | venue |
| `sa2_pct_drive_to_work` | `ctx_traffic_top1_lag_7` | -0.587 | ctx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top1_lag_1` | -0.586 | ctx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top3_lag_1` | -0.586 | ctx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top3_lag_7` | -0.586 | ctx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t2` | -0.585 | wx |
| `sa2_erp_median_age` | `wx_temp_min_c_t3` | -0.583 | wx |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t3` | +0.582 | wx |
| `sa2_erp_median_age` | `wx_temp_min_c_t4` | -0.579 | wx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top2_lag_1` | -0.579 | ctx |
| `sa2_erp_population_density_per_km2` | `wx_temp_min_c_t4` | +0.578 | wx |
| `sa2_median_age` | `wx_temp_min_c_t4` | -0.577 | wx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top2_lag_7` | -0.576 | ctx |
| `sa2_pct_renters` | `stn_competitors_within_2km` | +0.575 | stn |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t4` | -0.573 | wx |
| `sa2_pct_renters` | `stn_competitors_within_5km` | +0.569 | stn |
| `sa2_erp_population_density_per_km2` | `stn_competitors_within_2km` | +0.568 | stn |
| `sa2_median_age` | `wx_temp_min_c_t3` | -0.563 | wx |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t7` | +0.561 | wx |
| `sa2_seifa_ieo_score` | `ctx_traffic_5km_radius_count` | +0.547 | ctx |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t6` | +0.546 | wx |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t2` | -0.540 | wx |
| `sa2_motor_vehicles_per_dwelling` | `stn_competitors_within_5km` | -0.537 | stn |
| `sa2_motor_vehicles_per_dwelling` | `ctx_traffic_5km_radius_count` | -0.535 | ctx |
| `sa2_motor_vehicles_per_dwelling` | `stn_competitors_within_2km` | -0.530 | stn |
| `sa2_erp_population_density_per_km2` | `wx_temp_min_c_t7` | +0.527 | wx |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t4` | +0.525 | wx |
| `sa2_erp_median_age` | `stn_competitors_within_5km` | -0.523 | stn |
| `sa2_median_household_income_weekly` | `wx_temp_min_c_t2` | +0.521 | wx |
| `sa2_median_age` | `stn_competitors_within_5km` | -0.516 | stn |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t3` | -0.515 | wx |
| `sa2_erp_median_age` | `wx_temp_min_c_t2` | -0.515 | wx |
| `sa2_median_age` | `wx_temp_min_c_t2` | -0.514 | wx |
| `sa2_erp_population_density_per_km2` | `ctx_traffic_top2_lag_1` | +0.511 | ctx |
| `sa2_erp_population_density_per_km2` | `ctx_traffic_top2_lag_7` | +0.510 | ctx |
| `sa2_erp_population_density_per_km2` | `ctx_traffic_top3_lag_7` | +0.507 | ctx |
| `sa2_erp_population_density_per_km2` | `wx_temp_min_c_t5` | +0.507 | wx |
| `sa2_erp_population_density_per_km2` | `ctx_traffic_top3_lag_1` | +0.507 | ctx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t3` | -0.507 | wx |
| `sa2_pct_employed_full_time` | `wx_temp_max_c_t2` | -0.506 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t6` | -0.504 | wx |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | `ctx_traffic_5km_radius_count` | +0.502 | ctx |

---

_Generated by `python -m fuel_pred.evaluate.compare`. Re-run after
`make train` to refresh; predictions are read from
`models/predictions_*.parquet` rather than re-loading the pickles
for speed._
