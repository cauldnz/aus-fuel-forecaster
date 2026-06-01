# Model A vs Model B vs Model B' — comparison report

        Generated: 2026-06-01 02:51:35 UTC
        Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e4b_curation_only.parquet`
        Models:   `C:\repos\cauldnz\aus-fuel-forecaster\models_e4b_curation_only/model_a.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e4b_curation_only/model_b.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e4b_curation_only/model_b_prime.pkl`

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
| test_normal | 849,334 | 6.373 | 6.541 | +0.168 | 10.953 | 11.106 | 3.352 | 3.445 | +0.093 |
| test_crisis | 172,858 | 13.616 | 14.187 | +0.571 | 19.054 | 19.550 | 6.181 | 6.431 | +0.250 |

## Headline (overall) — B vs B' (venue-block additive sanity check)

| Fold | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| test_normal | 849,334 | 6.541 | 6.652 | +0.110 | 11.231 | 3.508 | +0.279 |
| test_crisis | 172,858 | 14.187 | 13.585 | -0.602 | 18.516 | 6.207 | -0.031 |

## Segmented by Metro / regional

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 846,336 | 6.377 | 6.546 | 6.657 | +0.169 | +0.111 | 3.511 |
| True | 2,998 | 5.209 | 5.184 | 5.065 | -0.025 | -0.118 | 2.701 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 172,336 | 13.618 | 14.190 | 13.589 | +0.571 | -0.600 | 6.209 |
| True | 522 | 12.811 | 13.132 | 12.088 | +0.321 | -1.044 | 5.442 |

## Segmented by Brand (top 8 + Other)

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 203,257 | 6.787 | 7.002 | 7.115 | +0.215 | +0.113 | 3.690 |
| Other | 163,936 | 5.401 | 5.484 | 5.617 | +0.083 | +0.133 | 3.025 |
| 7-Eleven | 120,966 | 8.585 | 8.822 | 9.064 | +0.237 | +0.243 | 4.711 |
| Metro | 94,517 | 5.739 | 5.951 | 5.957 | +0.212 | +0.006 | 3.289 |
| BP | 88,670 | 6.652 | 6.785 | 6.926 | +0.133 | +0.141 | 3.588 |
| Independent | 68,994 | 4.802 | 4.907 | 4.951 | +0.105 | +0.044 | 2.663 |
| Coles Express | 50,686 | 6.176 | 6.326 | 6.342 | +0.150 | +0.015 | 3.196 |
| Speedway | 29,238 | 6.087 | 6.259 | 6.332 | +0.172 | +0.073 | 3.502 |
| United | 29,070 | 5.327 | 5.505 | 5.530 | +0.178 | +0.025 | 2.964 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 39,875 | 13.894 | 14.568 | 13.934 | +0.674 | -0.634 | 6.338 |
| Other | 35,702 | 13.155 | 13.639 | 12.993 | +0.484 | -0.647 | 5.861 |
| 7-Eleven | 21,900 | 13.494 | 13.955 | 13.694 | +0.461 | -0.261 | 6.660 |
| BP | 18,770 | 14.011 | 14.731 | 14.047 | +0.720 | -0.684 | 6.360 |
| Metro | 17,740 | 11.737 | 12.185 | 11.590 | +0.448 | -0.595 | 5.436 |
| Independent | 16,184 | 15.707 | 16.328 | 15.481 | +0.622 | -0.847 | 6.679 |
| Shell | 9,321 | 14.706 | 15.432 | 14.647 | +0.726 | -0.785 | 6.560 |
| Reddy Express | 8,212 | 12.915 | 13.323 | 13.018 | +0.409 | -0.306 | 6.035 |
| United | 5,154 | 12.782 | 13.319 | 12.734 | +0.536 | -0.584 | 5.909 |

## Segmented by Fuel type

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 849,334 | 6.373 | 6.541 | 6.652 | +0.168 | +0.110 | 3.508 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 172,858 | 13.616 | 14.187 | 13.585 | +0.571 | -0.602 | 6.207 |

## Segmented by SEIFA quintile

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 159,692 | 5.979 | 6.190 | 6.227 | +0.211 | +0.037 | 3.352 |
| Q4 | 156,121 | 6.412 | 6.563 | 6.659 | +0.150 | +0.096 | 3.517 |
| Q3 | 154,732 | 5.839 | 5.968 | 6.066 | +0.129 | +0.097 | 3.211 |
| Q5 | 153,650 | 7.983 | 8.120 | 8.266 | +0.137 | +0.146 | 4.275 |
| Q2 | 151,464 | 5.717 | 5.867 | 5.925 | +0.150 | +0.059 | 3.132 |
| Unknown | 73,675 | 6.255 | 6.554 | 6.916 | +0.299 | +0.362 | 3.624 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 32,137 | 13.576 | 14.117 | 13.417 | +0.541 | -0.700 | 6.117 |
| Q4 | 31,839 | 13.172 | 13.675 | 13.071 | +0.503 | -0.605 | 6.031 |
| Q5 | 31,650 | 13.296 | 13.824 | 13.329 | +0.528 | -0.495 | 6.293 |
| Q3 | 31,643 | 13.797 | 14.332 | 13.687 | +0.535 | -0.645 | 6.154 |
| Q2 | 31,555 | 14.492 | 15.106 | 14.325 | +0.615 | -0.781 | 6.371 |
| Unknown | 14,034 | 13.059 | 13.927 | 13.818 | +0.868 | -0.109 | 6.367 |

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
| 1 | `lag_price_1` | lag | 33,398,374 | 56.20 | 4,930 |
| 2 | `lag_price_2` | lag | 5,778,440 | 9.72 | 2,229 |
| 3 | `days_since_last_price_change` | lag | 3,298,639 | 5.55 | 3,041 |
| 4 | `lag_price_3` | lag | 2,473,003 | 4.16 | 1,127 |
| 5 | `price_minus_28d_min` | lag | 1,322,946 | 2.23 | 1,513 |
| 6 | `cal_day_of_month` | cal | 1,156,728 | 1.95 | 1,683 |
| 7 | `price_minus_28d_max` | lag | 1,152,292 | 1.94 | 1,390 |
| 8 | `upstream_brent_lag_0` | upstream | 1,051,770 | 1.77 | 673 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 900,135 | 1.51 | 754 |
| 10 | `upstream_brent_lag_14` | upstream | 840,014 | 1.41 | 666 |
| 11 | `lag_price_28` | lag | 784,610 | 1.32 | 1,303 |
| 12 | `upstream_brent_lag_3` | upstream | 755,262 | 1.27 | 622 |
| 13 | `upstream_brent_lag_1` | upstream | 651,208 | 1.10 | 433 |
| 14 | `cal_year` | cal | 642,489 | 1.08 | 419 |
| 15 | `roll_price_mean_28` | lag | 527,377 | 0.89 | 1,044 |
| 16 | `upstream_brent_lag_7` | upstream | 402,573 | 0.68 | 464 |
| 17 | `upstream_brent_aud_lag_14` | upstream | 386,017 | 0.65 | 580 |
| 18 | `xfuel_dl_price_lag_0` | lag | 331,573 | 0.56 | 532 |
| 19 | `upstream_audusd_lag_0` | upstream | 294,626 | 0.50 | 808 |
| 20 | `stn_brand_raw` | stn | 232,338 | 0.39 | 801 |

### Model B' — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 30,280,046 | 50.94 | 5,020 |
| 2 | `lag_price_2` | lag | 6,045,877 | 10.17 | 2,220 |
| 3 | `lag_price_3` | lag | 4,766,135 | 8.02 | 1,916 |
| 4 | `days_since_last_price_change` | lag | 3,661,761 | 6.16 | 3,742 |
| 5 | `price_minus_28d_min` | lag | 1,483,401 | 2.50 | 1,723 |
| 6 | `upstream_brent_lag_0` | upstream | 1,195,808 | 2.01 | 812 |
| 7 | `cal_day_of_month` | cal | 1,140,436 | 1.92 | 1,773 |
| 8 | `upstream_brent_aud_lag_0` | upstream | 1,133,092 | 1.91 | 832 |
| 9 | `price_minus_28d_max` | lag | 1,129,731 | 1.90 | 1,454 |
| 10 | `lag_price_28` | lag | 776,051 | 1.31 | 1,264 |
| 11 | `upstream_brent_lag_1` | upstream | 723,598 | 1.22 | 584 |
| 12 | `upstream_brent_lag_14` | upstream | 714,039 | 1.20 | 772 |
| 13 | `roll_price_mean_28` | lag | 626,170 | 1.05 | 1,191 |
| 14 | `upstream_brent_lag_3` | upstream | 493,631 | 0.83 | 636 |
| 15 | `cal_year` | cal | 468,232 | 0.79 | 503 |
| 16 | `upstream_brent_lag_7` | upstream | 363,226 | 0.61 | 463 |
| 17 | `stn_nearest_venue_km` | venue | 357,362 | 0.60 | 767 |
| 18 | `upstream_brent_aud_lag_14` | upstream | 354,858 | 0.60 | 649 |
| 19 | `upstream_audusd_lag_0` | upstream | 336,955 | 0.57 | 994 |
| 20 | `roll_price_mean_7` | lag | 313,890 | 0.53 | 515 |

### Where SA2 features rank in Model B

| SA2 feature | Rank in B | Gain | Gain % |
|-------------|----------:|-----:|-------:|
| `sa2_pct_drive_to_work` | 29 | 137,616 | 0.23 |
| `sa2_dss_parenting_payment_partnered_recipients` | 47 | 24,429 | 0.04 |
| `sa2_seifa_ieo_score` | 49 | 18,254 | 0.03 |
| `sa2_erp_median_age` | 50 | 16,197 | 0.03 |
| `sa2_dss_carer_allowance_recipients` | 52 | 13,084 | 0.02 |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | 53 | 11,642 | 0.02 |
| `sa2_dss_carer_payment_recipients` | 55 | 9,154 | 0.02 |
| `sa2_pct_renters` | 58 | 7,523 | 0.01 |
| `sa2_pct_employed_full_time` | 59 | 7,503 | 0.01 |
| `sa2_pct_jobseeker_recipients` | 60 | 7,295 | 0.01 |
| `sa2_pct_aged_65_plus` | 62 | 6,630 | 0.01 |
| `sa2_pct_age_pension_recipients` | 64 | 6,012 | 0.01 |
| `sa2_median_age` | 65 | 5,475 | 0.01 |
| `sa2_welfare_density_index` | 67 | 4,449 | 0.01 |
| `sa2_pct_one_parent_family` | 68 | 4,338 | 0.01 |
| `sa2_motor_vehicles_per_dwelling` | 69 | 3,698 | 0.01 |
| `sa2_seifa_irsd_score` | 70 | 3,577 | 0.01 |
| `sa2_erp_population_65_plus` | 71 | 2,896 | 0.00 |
| `sa2_median_household_income_weekly` | 72 | 2,596 | 0.00 |
| `sa2_total_population` | 73 | 2,584 | 0.00 |

### Where VENUE-block features rank in Model B' (spec §13.6 Phase 1)

| Venue feature | Rank in B' | Gain | Gain % |
|---------------|-----------:|-----:|-------:|
| `stn_nearest_venue_km` | 17 | 357,362 | 0.60 |
| `stn_nearest_venue_capacity` | 50 | 10,687 | 0.02 |
| `cal_is_pre_long_weekend` | 78 | 567 | 0.00 |
| `stn_nearest_venue_type` | 82 | 179 | 0.00 |
| `stn_n_venues_within_5km` | 83 | 104 | 0.00 |

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
| `sa2_seifa_irsd_score` | `ctx_traffic_top1_lag_7` (+0.269) | `ctx_traffic_top1_lag_1` (+0.268) | `stn_nearest_venue_km` (-0.231) |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t2` (+0.612) ⚠️ | `wx_temp_min_c_t3` (+0.582) ⚠️ | `wx_temp_min_c_t7` (+0.561) ⚠️ |

### High correlations (|r| ≥ 0.5)

| SA2 feature | Non-SA2 feature | r | Block |
|-------------|------------------|--:|-------|
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t6` | -0.808 | wx |
| `sa2_pct_drive_to_work` | `ctx_traffic_5km_radius_count` | -0.760 | ctx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t2` | -0.749 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t7` | -0.714 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t5` | -0.682 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t4` | -0.668 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_min_c_t3` | -0.659 | wx |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | `stn_competitors_within_5km` | +0.656 | stn |
| `sa2_pct_drive_to_work` | `stn_competitors_within_5km` | -0.622 | stn |
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
| `sa2_median_age` | `wx_temp_min_c_t4` | -0.577 | wx |
| `sa2_pct_drive_to_work` | `ctx_traffic_top2_lag_7` | -0.576 | ctx |
| `sa2_pct_renters` | `stn_competitors_within_2km` | +0.575 | stn |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t4` | -0.573 | wx |
| `sa2_pct_renters` | `stn_competitors_within_5km` | +0.569 | stn |
| `sa2_median_age` | `wx_temp_min_c_t3` | -0.563 | wx |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t7` | +0.561 | wx |
| `sa2_seifa_ieo_score` | `ctx_traffic_5km_radius_count` | +0.547 | ctx |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t6` | +0.546 | wx |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t2` | -0.540 | wx |
| `sa2_motor_vehicles_per_dwelling` | `stn_competitors_within_5km` | -0.537 | stn |
| `sa2_motor_vehicles_per_dwelling` | `ctx_traffic_5km_radius_count` | -0.535 | ctx |
| `sa2_motor_vehicles_per_dwelling` | `stn_competitors_within_2km` | -0.530 | stn |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t4` | +0.525 | wx |
| `sa2_erp_median_age` | `stn_competitors_within_5km` | -0.523 | stn |
| `sa2_median_household_income_weekly` | `wx_temp_min_c_t2` | +0.521 | wx |
| `sa2_median_age` | `stn_competitors_within_5km` | -0.516 | stn |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t3` | -0.515 | wx |
| `sa2_erp_median_age` | `wx_temp_min_c_t2` | -0.515 | wx |
| `sa2_median_age` | `wx_temp_min_c_t2` | -0.514 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t3` | -0.507 | wx |
| `sa2_pct_employed_full_time` | `wx_temp_max_c_t2` | -0.506 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t6` | -0.504 | wx |

---

_Generated by `python -m fuel_pred.evaluate.compare`. Re-run after
`make train` to refresh; predictions are read from
`models/predictions_*.parquet` rather than re-loading the pickles
for speed._
