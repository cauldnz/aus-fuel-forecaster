# Model A vs Model B vs Model B' — comparison report

        Generated: 2026-05-31 01:34:35 UTC
        Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e2_gcp_temporal.parquet`
        Models:   `C:\repos\cauldnz\aus-fuel-forecaster\models_e2_gcp_temporal/model_a.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e2_gcp_temporal/model_b.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e2_gcp_temporal/model_b_prime.pkl`

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
| test_normal | 849,334 | 6.185 | 6.655 | +0.470 | 10.680 | 11.183 | 3.253 | 3.512 | +0.259 |
| test_crisis | 172,858 | 13.016 | 12.761 | -0.255 | 18.213 | 17.873 | 5.924 | 5.833 | -0.091 |

## Headline (overall) — B vs B' (venue-block additive sanity check)

| Fold | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| test_normal | 849,334 | 6.655 | 6.616 | -0.038 | 10.957 | 3.479 | +0.432 |
| test_crisis | 172,858 | 12.761 | 13.407 | +0.646 | 18.631 | 6.073 | +0.391 |

## Segmented by Metro / regional

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 846,336 | 6.189 | 6.659 | 6.621 | +0.471 | -0.038 | 3.482 |
| True | 2,998 | 5.056 | 5.378 | 5.286 | +0.321 | -0.091 | 2.819 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 172,336 | 13.018 | 12.765 | 13.411 | -0.254 | +0.646 | 6.075 |
| True | 522 | 12.157 | 11.583 | 12.150 | -0.574 | +0.567 | 5.426 |

## Segmented by Brand (top 8 + Other)

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 203,257 | 6.568 | 7.075 | 7.093 | +0.506 | +0.019 | 3.669 |
| Other | 163,936 | 5.276 | 5.728 | 5.610 | +0.451 | -0.118 | 3.018 |
| 7-Eleven | 120,966 | 8.202 | 8.891 | 8.833 | +0.689 | -0.058 | 4.568 |
| Metro | 94,517 | 5.638 | 6.050 | 5.823 | +0.412 | -0.227 | 3.208 |
| BP | 88,670 | 6.451 | 6.913 | 6.997 | +0.463 | +0.084 | 3.617 |
| Independent | 68,994 | 4.788 | 5.011 | 5.073 | +0.223 | +0.062 | 2.724 |
| Coles Express | 50,686 | 5.949 | 6.297 | 6.446 | +0.348 | +0.149 | 3.238 |
| Speedway | 29,238 | 5.943 | 6.404 | 6.148 | +0.462 | -0.256 | 3.395 |
| United | 29,070 | 5.163 | 5.590 | 5.580 | +0.427 | -0.009 | 2.987 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 39,875 | 13.236 | 12.978 | 13.724 | -0.258 | +0.746 | 6.188 |
| Other | 35,702 | 12.532 | 12.272 | 12.842 | -0.261 | +0.570 | 5.745 |
| 7-Eleven | 21,900 | 12.882 | 12.727 | 13.102 | -0.155 | +0.375 | 6.317 |
| BP | 18,770 | 13.464 | 13.147 | 13.940 | -0.317 | +0.793 | 6.262 |
| Metro | 17,740 | 11.319 | 11.182 | 11.603 | -0.137 | +0.421 | 5.406 |
| Independent | 16,184 | 15.066 | 14.680 | 15.670 | -0.386 | +0.990 | 6.710 |
| Shell | 9,321 | 14.113 | 13.728 | 14.558 | -0.385 | +0.830 | 6.465 |
| Reddy Express | 8,212 | 12.357 | 12.047 | 12.595 | -0.310 | +0.548 | 5.781 |
| United | 5,154 | 12.060 | 12.011 | 12.536 | -0.049 | +0.525 | 5.782 |

## Segmented by Fuel type

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 849,334 | 6.185 | 6.655 | 6.616 | +0.470 | -0.038 | 3.479 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 172,858 | 13.016 | 12.761 | 13.407 | -0.255 | +0.646 | 6.073 |

## Segmented by SEIFA quintile

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 159,692 | 5.848 | 6.316 | 6.169 | +0.468 | -0.147 | 3.314 |
| Q4 | 156,121 | 6.221 | 6.665 | 6.617 | +0.445 | -0.048 | 3.485 |
| Q3 | 154,732 | 5.693 | 6.115 | 6.079 | +0.421 | -0.035 | 3.210 |
| Q5 | 153,650 | 7.654 | 8.129 | 8.166 | +0.475 | +0.037 | 4.205 |
| Q2 | 151,464 | 5.606 | 6.015 | 5.955 | +0.409 | -0.060 | 3.141 |
| Unknown | 73,675 | 5.995 | 6.741 | 6.841 | +0.746 | +0.099 | 3.572 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 32,137 | 12.986 | 12.651 | 13.294 | -0.335 | +0.643 | 6.015 |
| Q4 | 31,839 | 12.571 | 12.377 | 12.897 | -0.194 | +0.520 | 5.906 |
| Q5 | 31,650 | 12.777 | 12.462 | 13.009 | -0.315 | +0.547 | 6.091 |
| Q3 | 31,643 | 13.156 | 12.931 | 13.575 | -0.225 | +0.644 | 6.046 |
| Q2 | 31,555 | 13.868 | 13.564 | 14.281 | -0.304 | +0.717 | 6.291 |
| Unknown | 14,034 | 12.399 | 12.372 | 13.375 | -0.027 | +1.003 | 6.113 |

## Feature importance

### Model A — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 32,216,619 | 52.89 | 5,679 |
| 2 | `lag_price_2` | lag | 8,955,291 | 14.70 | 3,231 |
| 3 | `days_since_last_price_change` | lag | 3,883,321 | 6.38 | 4,018 |
| 4 | `lag_price_3` | lag | 1,383,334 | 2.27 | 1,468 |
| 5 | `price_minus_28d_min` | lag | 1,364,358 | 2.24 | 1,940 |
| 6 | `upstream_brent_lag_0` | upstream | 1,336,112 | 2.19 | 880 |
| 7 | `cal_day_of_month` | cal | 1,158,373 | 1.90 | 2,160 |
| 8 | `price_minus_28d_max` | lag | 1,138,362 | 1.87 | 1,763 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 1,109,992 | 1.82 | 1,079 |
| 10 | `lag_price_28` | lag | 813,974 | 1.34 | 1,547 |
| 11 | `upstream_brent_lag_14` | upstream | 802,779 | 1.32 | 1,012 |
| 12 | `roll_price_mean_28` | lag | 648,543 | 1.06 | 1,438 |
| 13 | `upstream_brent_lag_1` | upstream | 579,238 | 0.95 | 563 |
| 14 | `upstream_brent_aud_lag_14` | upstream | 546,413 | 0.90 | 846 |
| 15 | `upstream_brent_lag_7` | upstream | 473,968 | 0.78 | 578 |
| 16 | `cal_year` | cal | 451,529 | 0.74 | 544 |
| 17 | `upstream_brent_lag_3` | upstream | 377,222 | 0.62 | 554 |
| 18 | `upstream_audusd_lag_0` | upstream | 299,873 | 0.49 | 1,008 |
| 19 | `xfuel_dl_price_lag_0` | lag | 284,880 | 0.47 | 573 |
| 20 | `roll_price_mean_7` | lag | 284,126 | 0.47 | 612 |

### Model B — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 34,769,923 | 58.22 | 4,809 |
| 2 | `lag_price_2` | lag | 5,160,496 | 8.64 | 2,073 |
| 3 | `days_since_last_price_change` | lag | 3,692,296 | 6.18 | 3,248 |
| 4 | `lag_price_3` | lag | 1,371,127 | 2.30 | 1,137 |
| 5 | `price_minus_28d_min` | lag | 1,305,812 | 2.19 | 1,504 |
| 6 | `upstream_brent_lag_0` | upstream | 1,216,679 | 2.04 | 643 |
| 7 | `price_minus_28d_max` | lag | 1,042,658 | 1.75 | 1,271 |
| 8 | `cal_day_of_month` | cal | 1,014,511 | 1.70 | 1,563 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 862,644 | 1.44 | 691 |
| 10 | `upstream_brent_lag_3` | upstream | 735,157 | 1.23 | 581 |
| 11 | `upstream_brent_lag_1` | upstream | 734,170 | 1.23 | 476 |
| 12 | `roll_price_mean_7` | lag | 693,215 | 1.16 | 727 |
| 13 | `upstream_brent_lag_7` | upstream | 674,295 | 1.13 | 478 |
| 14 | `upstream_brent_aud_lag_14` | upstream | 662,536 | 1.11 | 675 |
| 15 | `lag_price_28` | lag | 657,318 | 1.10 | 999 |
| 16 | `upstream_brent_lag_14` | upstream | 586,952 | 0.98 | 779 |
| 17 | `roll_price_mean_28` | lag | 581,014 | 0.97 | 1,065 |
| 18 | `cal_year` | cal | 512,330 | 0.86 | 447 |
| 19 | `upstream_audusd_lag_0` | upstream | 292,289 | 0.49 | 706 |
| 20 | `xfuel_dl_price_lag_0` | lag | 268,713 | 0.45 | 478 |

### Model B' — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 30,542,017 | 52.16 | 3,891 |
| 2 | `lag_price_2` | lag | 8,023,579 | 13.70 | 1,990 |
| 3 | `days_since_last_price_change` | lag | 3,290,864 | 5.62 | 2,790 |
| 4 | `lag_price_3` | lag | 1,583,855 | 2.71 | 907 |
| 5 | `price_minus_28d_min` | lag | 1,437,200 | 2.45 | 1,327 |
| 6 | `upstream_brent_lag_0` | upstream | 1,372,723 | 2.34 | 609 |
| 7 | `price_minus_28d_max` | lag | 1,236,605 | 2.11 | 1,150 |
| 8 | `upstream_brent_aud_lag_0` | upstream | 1,180,237 | 2.02 | 753 |
| 9 | `cal_day_of_month` | cal | 1,058,660 | 1.81 | 1,367 |
| 10 | `upstream_brent_lag_1` | upstream | 947,947 | 1.62 | 379 |
| 11 | `lag_price_28` | lag | 752,621 | 1.29 | 1,025 |
| 12 | `cal_year` | cal | 607,994 | 1.04 | 409 |
| 13 | `roll_price_mean_28` | lag | 574,768 | 0.98 | 962 |
| 14 | `roll_price_mean_7` | lag | 536,216 | 0.92 | 545 |
| 15 | `upstream_brent_lag_14` | upstream | 513,269 | 0.88 | 635 |
| 16 | `upstream_brent_aud_lag_14` | upstream | 507,704 | 0.87 | 549 |
| 17 | `upstream_brent_lag_7` | upstream | 400,892 | 0.68 | 419 |
| 18 | `upstream_brent_lag_3` | upstream | 368,475 | 0.63 | 462 |
| 19 | `stn_nearest_venue_km` | venue | 354,279 | 0.61 | 564 |
| 20 | `upstream_audusd_lag_0` | upstream | 330,619 | 0.56 | 644 |

### Where SA2 features rank in Model B

| SA2 feature | Rank in B | Gain | Gain % |
|-------------|----------:|-----:|-------:|
| `sa2_pct_drive_to_work` | 34 | 80,328 | 0.13 |
| `sa2_dss_parenting_payment_partnered_recipients` | 45 | 24,796 | 0.04 |
| `sa2_seifa_ieo_score` | 48 | 17,428 | 0.03 |
| `sa2_dss_carer_allowance_recipients` | 49 | 16,932 | 0.03 |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | 51 | 10,380 | 0.02 |
| `sa2_dss_carer_payment_recipients` | 53 | 10,260 | 0.02 |
| `sa2_seifa_irsd_score` | 55 | 9,733 | 0.02 |
| `sa2_median_age` | 56 | 9,665 | 0.02 |
| `sa2_pct_aged_65_plus` | 57 | 9,469 | 0.02 |
| `sa2_pct_employed_full_time` | 59 | 6,906 | 0.01 |
| `sa2_pct_renters` | 61 | 6,241 | 0.01 |
| `sa2_median_household_income_weekly` | 63 | 5,968 | 0.01 |
| `sa2_motor_vehicles_per_dwelling` | 65 | 4,199 | 0.01 |
| `sa2_pct_one_parent_family` | 66 | 3,153 | 0.01 |
| `sa2_total_population` | 88 | 0 | 0.00 |

### Where VENUE-block features rank in Model B' (spec §13.6 Phase 1)

| Venue feature | Rank in B' | Gain | Gain % |
|---------------|-----------:|-----:|-------:|
| `stn_nearest_venue_km` | 19 | 354,279 | 0.61 |
| `stn_nearest_venue_capacity` | 55 | 6,441 | 0.01 |
| `stn_n_venues_within_5km` | 72 | 255 | 0.00 |
| `cal_is_pre_long_weekend` | 76 | 59 | 0.00 |
| `stn_nearest_venue_type` | 88 | 0 | 0.00 |

## SA2 ↔ non-SA2 feature correlation

_Pearson `r` between each SA2 feature and the most-correlated non-SA2 numeric feature, computed on a sample of 100,000 rows. Categoricals are excluded (Pearson is numeric-only). High correlation (`|r| > 0.5`) flags features the model could already infer from existing inputs._

### Top 3 correlated non-SA2 features per SA2 feature

| SA2 feature | #1 (|r|) | #2 (|r|) | #3 (|r|) |
|-------------|----------|----------|----------|
| `sa2_dss_parenting_payment_partnered_recipients` | `stn_competitors_within_5km` (+0.455) | `stn_competitors_within_2km` (+0.328) | `wx_wind_speed_max_kmh_t3` (-0.302) |
| `sa2_dss_carer_payment_recipients` | `wx_wind_speed_max_kmh_t3` (-0.385) | `stn_competitors_within_5km` (+0.381) | `wx_wind_speed_max_kmh_t7` (-0.365) |
| `sa2_dss_carer_allowance_recipients` | `wx_wind_speed_max_kmh_t3` (-0.464) | `wx_wind_speed_max_kmh_t2` (-0.447) | `wx_wind_speed_max_kmh_t7` (-0.369) |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | `stn_competitors_within_5km` (+0.656) ⚠️ | `stn_competitors_within_2km` (+0.481) | `ctx_traffic_5km_radius_count` (+0.473) |
| `sa2_total_population` | `wx_wind_speed_max_kmh_t6` (-0.509) ⚠️ | `wx_wind_speed_max_kmh_t4` (-0.467) | `wx_wind_speed_max_kmh_t5` (-0.438) |
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
| `sa2_total_population` | `wx_wind_speed_max_kmh_t6` | -0.509 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t3` | -0.507 | wx |
| `sa2_pct_employed_full_time` | `wx_temp_max_c_t2` | -0.506 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t6` | -0.504 | wx |

---

_Generated by `python -m fuel_pred.evaluate.compare`. Re-run after
`make train` to refresh; predictions are read from
`models/predictions_*.parquet` rather than re-loading the pickles
for speed._
