# Model A vs Model B vs Model B' — comparison report

        Generated: 2026-06-01 02:25:18 UTC
        Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e4a_density_only.parquet`
        Models:   `C:\repos\cauldnz\aus-fuel-forecaster\models_e4a_density_only/model_a.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e4a_density_only/model_b.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e4a_density_only/model_b_prime.pkl`

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
| test_normal | 849,334 | 6.373 | 6.500 | +0.127 | 10.953 | 10.970 | 3.352 | 3.424 | +0.072 |
| test_crisis | 172,858 | 13.616 | 13.066 | -0.550 | 19.054 | 18.088 | 6.181 | 5.965 | -0.216 |

## Headline (overall) — B vs B' (venue-block additive sanity check)

| Fold | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| test_normal | 849,334 | 6.500 | 6.840 | +0.340 | 11.237 | 3.619 | +0.467 |
| test_crisis | 172,858 | 13.066 | 13.377 | +0.310 | 18.356 | 6.125 | -0.239 |

## Segmented by Metro / regional

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 846,336 | 6.377 | 6.504 | 6.845 | +0.127 | +0.341 | 3.621 |
| True | 2,998 | 5.209 | 5.365 | 5.420 | +0.156 | +0.056 | 2.912 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 172,336 | 13.618 | 13.069 | 13.381 | -0.549 | +0.312 | 6.127 |
| True | 522 | 12.811 | 12.141 | 12.027 | -0.670 | -0.114 | 5.425 |

## Segmented by Brand (top 8 + Other)

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 203,257 | 6.787 | 6.930 | 7.308 | +0.143 | +0.378 | 3.803 |
| Other | 163,936 | 5.401 | 5.571 | 5.850 | +0.170 | +0.279 | 3.165 |
| 7-Eleven | 120,966 | 8.585 | 8.547 | 9.138 | -0.038 | +0.590 | 4.758 |
| Metro | 94,517 | 5.739 | 5.847 | 6.177 | +0.108 | +0.329 | 3.421 |
| BP | 88,670 | 6.652 | 6.861 | 7.170 | +0.209 | +0.309 | 3.725 |
| Independent | 68,994 | 4.802 | 5.020 | 5.160 | +0.218 | +0.140 | 2.784 |
| Coles Express | 50,686 | 6.176 | 6.270 | 6.395 | +0.094 | +0.125 | 3.227 |
| Speedway | 29,238 | 6.087 | 6.165 | 6.479 | +0.078 | +0.315 | 3.596 |
| United | 29,070 | 5.327 | 5.483 | 5.860 | +0.156 | +0.377 | 3.158 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 39,875 | 13.894 | 13.305 | 13.667 | -0.589 | +0.363 | 6.232 |
| Other | 35,702 | 13.155 | 12.680 | 12.739 | -0.476 | +0.059 | 5.748 |
| 7-Eleven | 21,900 | 13.494 | 12.607 | 13.440 | -0.887 | +0.834 | 6.576 |
| BP | 18,770 | 14.011 | 13.600 | 13.901 | -0.411 | +0.301 | 6.307 |
| Metro | 17,740 | 11.737 | 11.293 | 11.563 | -0.444 | +0.271 | 5.443 |
| Independent | 16,184 | 15.707 | 15.202 | 15.239 | -0.504 | +0.036 | 6.569 |
| Shell | 9,321 | 14.706 | 14.277 | 14.525 | -0.429 | +0.248 | 6.519 |
| Reddy Express | 8,212 | 12.915 | 12.293 | 12.759 | -0.622 | +0.467 | 5.910 |
| United | 5,154 | 12.782 | 12.351 | 12.665 | -0.431 | +0.314 | 5.895 |

## Segmented by Fuel type

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 849,334 | 6.373 | 6.500 | 6.840 | +0.127 | +0.340 | 3.619 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 172,858 | 13.616 | 13.066 | 13.377 | -0.550 | +0.310 | 6.125 |

## Segmented by SEIFA quintile

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 159,692 | 5.979 | 6.165 | 6.442 | +0.186 | +0.277 | 3.481 |
| Q4 | 156,121 | 6.412 | 6.522 | 6.842 | +0.109 | +0.321 | 3.627 |
| Q3 | 154,732 | 5.839 | 6.013 | 6.280 | +0.174 | +0.267 | 3.336 |
| Q5 | 153,650 | 7.983 | 7.953 | 8.383 | -0.030 | +0.429 | 4.344 |
| Q2 | 151,464 | 5.717 | 5.941 | 6.135 | +0.224 | +0.194 | 3.254 |
| Unknown | 73,675 | 6.255 | 6.318 | 7.102 | +0.062 | +0.784 | 3.734 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 32,137 | 13.576 | 13.019 | 13.278 | -0.557 | +0.260 | 6.071 |
| Q4 | 31,839 | 13.172 | 12.613 | 12.887 | -0.559 | +0.274 | 5.962 |
| Q5 | 31,650 | 13.296 | 12.644 | 13.146 | -0.652 | +0.502 | 6.230 |
| Q3 | 31,643 | 13.797 | 13.280 | 13.473 | -0.518 | +0.193 | 6.064 |
| Q2 | 31,555 | 14.492 | 14.027 | 14.106 | -0.465 | +0.079 | 6.271 |
| Unknown | 14,034 | 13.059 | 12.515 | 13.377 | -0.543 | +0.861 | 6.184 |

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
| 1 | `lag_price_1` | lag | 31,121,424 | 51.95 | 6,121 |
| 2 | `lag_price_2` | lag | 8,134,583 | 13.58 | 3,029 |
| 3 | `days_since_last_price_change` | lag | 3,386,546 | 5.65 | 4,254 |
| 4 | `upstream_brent_lag_0` | upstream | 1,691,505 | 2.82 | 895 |
| 5 | `lag_price_3` | lag | 1,653,831 | 2.76 | 1,563 |
| 6 | `price_minus_28d_min` | lag | 1,458,994 | 2.44 | 1,990 |
| 7 | `price_minus_28d_max` | lag | 1,262,013 | 2.11 | 1,752 |
| 8 | `cal_day_of_month` | cal | 1,093,745 | 1.83 | 2,216 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 995,915 | 1.66 | 1,015 |
| 10 | `upstream_brent_lag_1` | upstream | 741,335 | 1.24 | 626 |
| 11 | `lag_price_28` | lag | 727,871 | 1.22 | 1,529 |
| 12 | `roll_price_mean_28` | lag | 621,874 | 1.04 | 1,435 |
| 13 | `upstream_brent_lag_14` | upstream | 578,943 | 0.97 | 1,089 |
| 14 | `roll_price_mean_7` | lag | 576,957 | 0.96 | 965 |
| 15 | `cal_year` | cal | 536,926 | 0.90 | 594 |
| 16 | `upstream_brent_lag_3` | upstream | 513,612 | 0.86 | 748 |
| 17 | `upstream_brent_aud_lag_14` | upstream | 415,074 | 0.69 | 844 |
| 18 | `xfuel_dl_price_lag_0` | lag | 350,128 | 0.58 | 694 |
| 19 | `upstream_audusd_lag_0` | upstream | 347,484 | 0.58 | 1,144 |
| 20 | `upstream_brent_lag_7` | upstream | 327,017 | 0.55 | 510 |

### Model B' — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 34,160,951 | 57.63 | 4,198 |
| 2 | `lag_price_2` | lag | 4,565,322 | 7.70 | 1,670 |
| 3 | `lag_price_3` | lag | 3,482,842 | 5.88 | 1,470 |
| 4 | `days_since_last_price_change` | lag | 3,286,376 | 5.54 | 2,785 |
| 5 | `upstream_brent_lag_0` | upstream | 1,268,391 | 2.14 | 624 |
| 6 | `cal_day_of_month` | cal | 1,236,416 | 2.09 | 1,490 |
| 7 | `price_minus_28d_min` | lag | 1,224,586 | 2.07 | 1,170 |
| 8 | `price_minus_28d_max` | lag | 1,159,485 | 1.96 | 1,170 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 928,637 | 1.57 | 643 |
| 10 | `lag_price_28` | lag | 749,932 | 1.27 | 937 |
| 11 | `roll_price_mean_28` | lag | 643,333 | 1.09 | 932 |
| 12 | `upstream_brent_aud_lag_14` | upstream | 604,271 | 1.02 | 495 |
| 13 | `cal_year` | cal | 592,990 | 1.00 | 419 |
| 14 | `upstream_brent_lag_3` | upstream | 565,955 | 0.95 | 458 |
| 15 | `upstream_brent_lag_14` | upstream | 562,875 | 0.95 | 549 |
| 16 | `upstream_brent_lag_1` | upstream | 390,792 | 0.66 | 280 |
| 17 | `upstream_brent_lag_7` | upstream | 354,880 | 0.60 | 366 |
| 18 | `stn_nearest_venue_km` | venue | 350,164 | 0.59 | 561 |
| 19 | `xfuel_dl_price_lag_0` | lag | 291,211 | 0.49 | 394 |
| 20 | `upstream_audusd_lag_0` | upstream | 252,819 | 0.43 | 590 |

### Where SA2 features rank in Model B

| SA2 feature | Rank in B | Gain | Gain % |
|-------------|----------:|-----:|-------:|
| `sa2_pct_drive_to_work` | 27 | 156,723 | 0.26 |
| `sa2_erp_population_density_per_km2` | 32 | 131,761 | 0.22 |
| `sa2_dss_parenting_payment_partnered_recipients` | 46 | 24,881 | 0.04 |
| `sa2_seifa_ieo_score` | 47 | 22,752 | 0.04 |
| `sa2_dss_carer_allowance_recipients` | 50 | 19,283 | 0.03 |
| `sa2_pct_aged_65_plus` | 54 | 11,964 | 0.02 |
| `sa2_median_age` | 55 | 11,944 | 0.02 |
| `sa2_dss_carer_payment_recipients` | 56 | 11,936 | 0.02 |
| `sa2_pct_employed_full_time` | 59 | 8,784 | 0.01 |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | 61 | 8,538 | 0.01 |
| `sa2_pct_renters` | 62 | 8,246 | 0.01 |
| `sa2_seifa_irsd_score` | 63 | 8,144 | 0.01 |
| `sa2_median_household_income_weekly` | 64 | 6,653 | 0.01 |
| `sa2_pct_one_parent_family` | 66 | 6,262 | 0.01 |
| `sa2_total_population` | 68 | 4,561 | 0.01 |
| `sa2_motor_vehicles_per_dwelling` | 70 | 4,190 | 0.01 |

### Where VENUE-block features rank in Model B' (spec §13.6 Phase 1)

| Venue feature | Rank in B' | Gain | Gain % |
|---------------|-----------:|-----:|-------:|
| `stn_nearest_venue_km` | 18 | 350,164 | 0.59 |
| `stn_nearest_venue_capacity` | 55 | 7,080 | 0.01 |
| `stn_nearest_venue_type` | 73 | 293 | 0.00 |
| `stn_n_venues_within_5km` | 76 | 144 | 0.00 |
| `cal_is_pre_long_weekend` | 89 | 0 | 0.00 |

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
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t3` | +0.582 | wx |
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
| `sa2_median_household_income_weekly` | `wx_temp_min_c_t2` | +0.521 | wx |
| `sa2_median_age` | `stn_competitors_within_5km` | -0.516 | stn |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t3` | -0.515 | wx |
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
