# Model A vs Model B vs Model B' — comparison report

        Generated: 2026-05-31 03:20:36 UTC
        Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e4_new_erp_density_plus_curation.parquet`
        Models:   `C:\repos\cauldnz\aus-fuel-forecaster\models_e4_new_erp_density_plus_curation/model_a.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e4_new_erp_density_plus_curation/model_b.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e4_new_erp_density_plus_curation/model_b_prime.pkl`

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
| test_normal | 849,334 | 6.373 | 6.285 | -0.088 | 10.953 | 10.862 | 3.352 | 3.309 | -0.043 |
| test_crisis | 172,858 | 13.616 | 13.013 | -0.603 | 19.054 | 18.128 | 6.181 | 5.929 | -0.252 |

## Headline (overall) — B vs B' (venue-block additive sanity check)

| Fold | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| test_normal | 849,334 | 6.285 | 6.524 | +0.239 | 10.994 | 3.416 | +0.151 |
| test_crisis | 172,858 | 13.013 | 13.574 | +0.562 | 19.023 | 6.125 | -0.042 |

## Segmented by Metro / regional

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 846,336 | 6.377 | 6.290 | 6.529 | -0.087 | +0.239 | 3.419 |
| True | 2,998 | 5.209 | 4.949 | 5.061 | -0.260 | +0.112 | 2.677 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 172,336 | 13.618 | 13.016 | 13.577 | -0.602 | +0.561 | 6.126 |
| True | 522 | 12.811 | 11.835 | 12.580 | -0.976 | +0.745 | 5.610 |

## Segmented by Brand (top 8 + Other)

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 203,257 | 6.787 | 6.718 | 7.035 | -0.069 | +0.317 | 3.622 |
| Other | 163,936 | 5.401 | 5.296 | 5.413 | -0.105 | +0.118 | 2.900 |
| 7-Eleven | 120,966 | 8.585 | 8.483 | 8.848 | -0.102 | +0.365 | 4.560 |
| Metro | 94,517 | 5.739 | 5.635 | 5.788 | -0.104 | +0.153 | 3.179 |
| BP | 88,670 | 6.652 | 6.560 | 6.801 | -0.092 | +0.241 | 3.498 |
| Independent | 68,994 | 4.802 | 4.727 | 4.775 | -0.075 | +0.048 | 2.553 |
| Coles Express | 50,686 | 6.176 | 6.085 | 6.627 | -0.091 | +0.541 | 3.322 |
| Speedway | 29,238 | 6.087 | 6.049 | 6.156 | -0.038 | +0.108 | 3.390 |
| United | 29,070 | 5.327 | 5.252 | 5.428 | -0.075 | +0.176 | 2.891 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 39,875 | 13.894 | 13.298 | 13.828 | -0.596 | +0.530 | 6.209 |
| Other | 35,702 | 13.155 | 12.547 | 13.088 | -0.608 | +0.541 | 5.834 |
| 7-Eleven | 21,900 | 13.494 | 12.756 | 13.240 | -0.737 | +0.484 | 6.359 |
| BP | 18,770 | 14.011 | 13.480 | 14.127 | -0.531 | +0.647 | 6.320 |
| Metro | 17,740 | 11.737 | 11.145 | 11.802 | -0.592 | +0.657 | 5.472 |
| Independent | 16,184 | 15.707 | 15.145 | 15.860 | -0.561 | +0.714 | 6.778 |
| Shell | 9,321 | 14.706 | 14.150 | 14.712 | -0.556 | +0.562 | 6.503 |
| Reddy Express | 8,212 | 12.915 | 12.300 | 12.689 | -0.614 | +0.388 | 5.799 |
| United | 5,154 | 12.782 | 12.220 | 12.664 | -0.562 | +0.443 | 5.801 |

## Segmented by Fuel type

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 849,334 | 6.373 | 6.285 | 6.524 | -0.088 | +0.239 | 3.416 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 172,858 | 13.616 | 13.013 | 13.574 | -0.603 | +0.562 | 6.125 |

## Segmented by SEIFA quintile

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 159,692 | 5.979 | 5.960 | 6.119 | -0.019 | +0.159 | 3.274 |
| Q4 | 156,121 | 6.412 | 6.290 | 6.519 | -0.122 | +0.228 | 3.418 |
| Q3 | 154,732 | 5.839 | 5.757 | 5.954 | -0.082 | +0.196 | 3.129 |
| Q5 | 153,650 | 7.983 | 7.776 | 8.149 | -0.207 | +0.373 | 4.183 |
| Q2 | 151,464 | 5.717 | 5.673 | 5.816 | -0.043 | +0.143 | 3.053 |
| Unknown | 73,675 | 6.255 | 6.233 | 6.675 | -0.022 | +0.442 | 3.472 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 32,137 | 13.576 | 12.907 | 13.552 | -0.669 | +0.646 | 6.110 |
| Q4 | 31,839 | 13.172 | 12.541 | 13.080 | -0.631 | +0.539 | 5.962 |
| Q5 | 31,650 | 13.296 | 12.667 | 13.087 | -0.630 | +0.420 | 6.098 |
| Q3 | 31,643 | 13.797 | 13.176 | 13.795 | -0.621 | +0.619 | 6.127 |
| Q2 | 31,555 | 14.492 | 13.932 | 14.496 | -0.559 | +0.563 | 6.367 |
| Unknown | 14,034 | 13.059 | 12.667 | 13.276 | -0.391 | +0.608 | 6.037 |

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
| 1 | `lag_price_1` | lag | 30,320,595 | 51.39 | 4,446 |
| 2 | `lag_price_2` | lag | 6,020,381 | 10.20 | 1,920 |
| 3 | `lag_price_3` | lag | 4,756,355 | 8.06 | 1,749 |
| 4 | `days_since_last_price_change` | lag | 3,616,619 | 6.13 | 3,444 |
| 5 | `price_minus_28d_min` | lag | 1,438,733 | 2.44 | 1,502 |
| 6 | `upstream_brent_lag_0` | upstream | 1,230,382 | 2.09 | 718 |
| 7 | `cal_day_of_month` | cal | 1,133,836 | 1.92 | 1,532 |
| 8 | `price_minus_28d_max` | lag | 1,101,786 | 1.87 | 1,295 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 976,179 | 1.65 | 692 |
| 10 | `lag_price_28` | lag | 765,869 | 1.30 | 1,137 |
| 11 | `upstream_brent_lag_14` | upstream | 743,675 | 1.26 | 692 |
| 12 | `roll_price_mean_28` | lag | 618,734 | 1.05 | 1,081 |
| 13 | `upstream_brent_lag_1` | upstream | 609,465 | 1.03 | 417 |
| 14 | `upstream_brent_lag_3` | upstream | 521,472 | 0.88 | 530 |
| 15 | `cal_year` | cal | 486,117 | 0.82 | 433 |
| 16 | `upstream_brent_aud_lag_14` | upstream | 446,601 | 0.76 | 576 |
| 17 | `upstream_audusd_lag_0` | upstream | 357,680 | 0.61 | 854 |
| 18 | `upstream_brent_lag_7` | upstream | 336,065 | 0.57 | 423 |
| 19 | `xfuel_dl_price_lag_0` | lag | 298,001 | 0.51 | 430 |
| 20 | `roll_price_mean_7` | lag | 288,472 | 0.49 | 448 |

### Model B' — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 33,747,856 | 57.23 | 4,143 |
| 2 | `lag_price_2` | lag | 4,112,961 | 6.97 | 1,555 |
| 3 | `days_since_last_price_change` | lag | 3,469,864 | 5.88 | 3,025 |
| 4 | `lag_price_3` | lag | 3,004,149 | 5.09 | 1,437 |
| 5 | `upstream_brent_lag_0` | upstream | 1,494,531 | 2.53 | 698 |
| 6 | `price_minus_28d_min` | lag | 1,422,451 | 2.41 | 1,364 |
| 7 | `price_minus_28d_max` | lag | 1,144,017 | 1.94 | 1,242 |
| 8 | `cal_day_of_month` | cal | 1,041,188 | 1.77 | 1,399 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 940,869 | 1.60 | 597 |
| 10 | `lag_price_28` | lag | 736,336 | 1.25 | 963 |
| 11 | `upstream_brent_lag_7` | upstream | 707,018 | 1.20 | 437 |
| 12 | `upstream_brent_aud_lag_14` | upstream | 680,530 | 1.15 | 620 |
| 13 | `roll_price_mean_28` | lag | 621,914 | 1.05 | 986 |
| 14 | `upstream_brent_lag_14` | upstream | 592,033 | 1.00 | 687 |
| 15 | `upstream_brent_lag_3` | upstream | 571,905 | 0.97 | 474 |
| 16 | `upstream_brent_lag_1` | upstream | 513,444 | 0.87 | 385 |
| 17 | `cal_year` | cal | 440,498 | 0.75 | 418 |
| 18 | `xfuel_dl_price_lag_0` | lag | 327,258 | 0.55 | 498 |
| 19 | `upstream_audusd_lag_0` | upstream | 317,332 | 0.54 | 655 |
| 20 | `stn_nearest_venue_km` | venue | 280,852 | 0.48 | 514 |

### Where SA2 features rank in Model B

| SA2 feature | Rank in B | Gain | Gain % |
|-------------|----------:|-----:|-------:|
| `sa2_pct_drive_to_work` | 33 | 87,737 | 0.15 |
| `sa2_erp_population_density_per_km2` | 41 | 40,638 | 0.07 |
| `sa2_dss_parenting_payment_partnered_recipients` | 44 | 31,323 | 0.05 |
| `sa2_seifa_ieo_score` | 51 | 12,440 | 0.02 |
| `sa2_dss_carer_allowance_recipients` | 52 | 12,375 | 0.02 |
| `sa2_erp_median_age` | 53 | 11,282 | 0.02 |
| `sa2_pct_jobseeker_recipients` | 56 | 9,160 | 0.02 |
| `sa2_pct_aged_65_plus` | 57 | 8,749 | 0.01 |
| `sa2_dss_carer_payment_recipients` | 58 | 7,164 | 0.01 |
| `sa2_pct_employed_full_time` | 60 | 6,935 | 0.01 |
| `sa2_pct_renters` | 61 | 5,949 | 0.01 |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | 63 | 5,534 | 0.01 |
| `sa2_pct_age_pension_recipients` | 64 | 5,164 | 0.01 |
| `sa2_welfare_density_index` | 67 | 4,089 | 0.01 |
| `sa2_seifa_irsd_score` | 68 | 3,562 | 0.01 |
| `sa2_median_household_income_weekly` | 69 | 3,446 | 0.01 |
| `sa2_pct_one_parent_family` | 70 | 3,249 | 0.01 |
| `sa2_erp_population_65_plus` | 71 | 3,208 | 0.01 |
| `sa2_median_age` | 72 | 3,005 | 0.01 |
| `sa2_motor_vehicles_per_dwelling` | 74 | 2,549 | 0.00 |
| `sa2_total_population` | 75 | 2,222 | 0.00 |

### Where VENUE-block features rank in Model B' (spec §13.6 Phase 1)

| Venue feature | Rank in B' | Gain | Gain % |
|---------------|-----------:|-----:|-------:|
| `stn_nearest_venue_km` | 20 | 280,852 | 0.48 |
| `stn_nearest_venue_capacity` | 59 | 5,283 | 0.01 |
| `stn_n_venues_within_5km` | 75 | 580 | 0.00 |
| `stn_nearest_venue_type` | 93 | 0 | 0.00 |
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
| `sa2_dss_parenting_payment_partnered_recipients` | `stn_competitors_within_5km` (+0.455) | `stn_competitors_within_2km` (+0.328) | `wx_wind_speed_max_kmh_t3` (-0.302) |
| `sa2_dss_carer_payment_recipients` | `wx_wind_speed_max_kmh_t3` (-0.385) | `stn_competitors_within_5km` (+0.381) | `wx_wind_speed_max_kmh_t7` (-0.365) |
| `sa2_dss_carer_allowance_recipients` | `wx_wind_speed_max_kmh_t3` (-0.464) | `wx_wind_speed_max_kmh_t2` (-0.447) | `wx_wind_speed_max_kmh_t7` (-0.369) |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | `stn_competitors_within_5km` (+0.656) ⚠️ | `stn_competitors_within_2km` (+0.481) | `ctx_traffic_5km_radius_count` (+0.473) |
| `sa2_erp_population_65_plus` | `wx_wind_speed_max_kmh_t2` (-0.417) | `wx_wind_speed_max_kmh_t3` (-0.296) | `wx_precipitation_mm_t2` (-0.278) |
| `sa2_erp_median_age` | `wx_temp_min_c_t3` (-0.583) ⚠️ | `wx_temp_min_c_t4` (-0.579) ⚠️ | `stn_competitors_within_5km` (-0.523) ⚠️ |
| `sa2_pct_age_pension_recipients` | `wx_wind_speed_max_kmh_t7` (-0.386) | `wx_wind_speed_max_kmh_t3` (-0.331) | `wx_wind_speed_max_kmh_t2` (-0.308) |
| `sa2_pct_jobseeker_recipients` | `wx_temp_min_c_t2` (-0.395) | `wx_wind_speed_max_kmh_t3` (+0.392) | `wx_temp_min_c_t3` (-0.377) |
| `sa2_welfare_density_index` | `wx_temp_min_c_t2` (-0.454) | `wx_temp_min_c_t3` (-0.435) | `wx_temp_min_c_t4` (-0.420) |
| `sa2_erp_population_density_per_km2` | `ctx_traffic_5km_radius_count` (+0.782) ⚠️ | `stn_competitors_within_5km` (+0.773) ⚠️ | `wx_temp_min_c_t6` (+0.645) ⚠️ |
| `sa2_seifa_irsd_score` | `ctx_traffic_top1_lag_7` (+0.269) | `ctx_traffic_top1_lag_1` (+0.268) | `stn_nearest_venue_km` (-0.231) |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t2` (+0.612) ⚠️ | `wx_temp_min_c_t3` (+0.582) ⚠️ | `wx_temp_min_c_t7` (+0.561) ⚠️ |

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
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t3` | -0.659 | wx |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | `stn_competitors_within_5km` | +0.656 | stn |
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

---

_Generated by `python -m fuel_pred.evaluate.compare`. Re-run after
`make train` to refresh; predictions are read from
`models/predictions_*.parquet` rather than re-loading the pickles
for speed._
