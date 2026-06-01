# Model A vs Model B vs Model B' — comparison report

        Generated: 2026-05-31 00:51:19 UTC
        Features: `C:\repos\cauldnz\aus-fuel-forecaster\data\processed\features_e1_dss_temporal.parquet`
        Models:   `C:\repos\cauldnz\aus-fuel-forecaster\models_e1_dss_temporal/model_a.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e1_dss_temporal/model_b.pkl`, `C:\repos\cauldnz\aus-fuel-forecaster\models_e1_dss_temporal/model_b_prime.pkl`

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
| test_normal | 849,334 | 6.373 | 6.049 | -0.324 | 10.953 | 10.705 | 3.352 | 3.170 | -0.182 |
| test_crisis | 172,858 | 13.616 | 13.446 | -0.170 | 19.054 | 18.926 | 6.181 | 6.077 | -0.104 |

## Headline (overall) — B vs B' (venue-block additive sanity check)

| Fold | n | MAE B | MAE B' | Δ MAE (B'−B) | RMSE B' | MAPE B' | Δ MAE (B'−A) |
|------|--:|------:|-------:|-------------:|--------:|--------:|-------------:|
| test_normal | 849,334 | 6.049 | 6.523 | +0.473 | 10.975 | 3.435 | +0.150 |
| test_crisis | 172,858 | 13.446 | 13.378 | -0.068 | 18.611 | 6.081 | -0.238 |

## Segmented by Metro / regional

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 846,336 | 6.377 | 6.054 | 6.527 | -0.323 | +0.474 | 3.438 |
| True | 2,998 | 5.209 | 4.814 | 5.156 | -0.394 | +0.341 | 2.750 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| False | 172,336 | 13.618 | 13.449 | 13.381 | -0.169 | -0.068 | 6.083 |
| True | 522 | 12.811 | 12.288 | 12.208 | -0.523 | -0.080 | 5.468 |

## Segmented by Brand (top 8 + Other)

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 203,257 | 6.787 | 6.488 | 6.953 | -0.299 | +0.465 | 3.601 |
| Other | 163,936 | 5.401 | 4.975 | 5.583 | -0.426 | +0.609 | 3.007 |
| 7-Eleven | 120,966 | 8.585 | 8.298 | 8.630 | -0.287 | +0.332 | 4.471 |
| Metro | 94,517 | 5.739 | 5.348 | 5.809 | -0.391 | +0.461 | 3.204 |
| BP | 88,670 | 6.652 | 6.366 | 6.884 | -0.286 | +0.518 | 3.562 |
| Independent | 68,994 | 4.802 | 4.495 | 5.050 | -0.307 | +0.555 | 2.717 |
| Coles Express | 50,686 | 6.176 | 6.068 | 6.302 | -0.108 | +0.233 | 3.170 |
| Speedway | 29,238 | 6.087 | 5.704 | 6.132 | -0.383 | +0.428 | 3.389 |
| United | 29,070 | 5.327 | 5.003 | 5.526 | -0.324 | +0.522 | 2.962 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Ampol | 39,875 | 13.894 | 13.746 | 13.642 | -0.148 | -0.104 | 6.173 |
| Other | 35,702 | 13.155 | 12.875 | 12.914 | -0.281 | +0.039 | 5.801 |
| 7-Eleven | 21,900 | 13.494 | 13.151 | 12.923 | -0.343 | -0.228 | 6.250 |
| BP | 18,770 | 14.011 | 14.027 | 13.916 | +0.016 | -0.111 | 6.275 |
| Metro | 17,740 | 11.737 | 11.564 | 11.509 | -0.174 | -0.055 | 5.375 |
| Independent | 16,184 | 15.707 | 15.784 | 15.685 | +0.077 | -0.099 | 6.744 |
| Shell | 9,321 | 14.706 | 14.644 | 14.606 | -0.062 | -0.039 | 6.514 |
| Reddy Express | 8,212 | 12.915 | 12.599 | 12.696 | -0.316 | +0.097 | 5.848 |
| United | 5,154 | 12.782 | 12.523 | 12.577 | -0.259 | +0.054 | 5.820 |

## Segmented by Fuel type

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 849,334 | 6.373 | 6.049 | 6.523 | -0.324 | +0.473 | 3.435 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| U91 | 172,858 | 13.616 | 13.446 | 13.378 | -0.170 | -0.068 | 6.081 |

## Segmented by SEIFA quintile

### test_normal

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 159,692 | 5.979 | 5.673 | 6.158 | -0.306 | +0.484 | 3.313 |
| Q4 | 156,121 | 6.412 | 6.064 | 6.503 | -0.348 | +0.439 | 3.429 |
| Q3 | 154,732 | 5.839 | 5.497 | 6.001 | -0.342 | +0.504 | 3.175 |
| Q5 | 153,650 | 7.983 | 7.588 | 7.955 | -0.395 | +0.367 | 4.101 |
| Q2 | 151,464 | 5.717 | 5.393 | 5.908 | -0.324 | +0.515 | 3.122 |
| Unknown | 73,675 | 6.255 | 6.134 | 6.726 | -0.121 | +0.592 | 3.516 |

### test_crisis

| Segment | n | MAE A | MAE B | MAE B' | Δ MAE (B−A) | Δ MAE (B'−B) | MAPE B' |
|---------|--:|------:|------:|-------:|------------:|------------:|--------:|
| Q1 | 32,137 | 13.576 | 13.353 | 13.326 | -0.223 | -0.026 | 6.051 |
| Q4 | 31,839 | 13.172 | 12.912 | 12.841 | -0.260 | -0.071 | 5.896 |
| Q5 | 31,650 | 13.296 | 13.006 | 12.869 | -0.291 | -0.137 | 6.041 |
| Q3 | 31,643 | 13.797 | 13.682 | 13.583 | -0.115 | -0.099 | 6.077 |
| Q2 | 31,555 | 14.492 | 14.406 | 14.309 | -0.086 | -0.097 | 6.335 |
| Unknown | 14,034 | 13.059 | 13.168 | 13.306 | +0.109 | +0.138 | 6.098 |

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
| 1 | `lag_price_1` | lag | 34,615,477 | 59.72 | 2,953 |
| 2 | `lag_price_2` | lag | 5,116,587 | 8.83 | 1,397 |
| 3 | `days_since_last_price_change` | lag | 3,573,645 | 6.17 | 2,202 |
| 4 | `lag_price_3` | lag | 1,338,774 | 2.31 | 739 |
| 5 | `price_minus_28d_min` | lag | 1,210,490 | 2.09 | 941 |
| 6 | `upstream_brent_lag_0` | upstream | 1,194,516 | 2.06 | 417 |
| 7 | `price_minus_28d_max` | lag | 1,026,340 | 1.77 | 846 |
| 8 | `cal_day_of_month` | cal | 942,539 | 1.63 | 990 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 840,709 | 1.45 | 426 |
| 10 | `upstream_brent_lag_1` | upstream | 802,703 | 1.38 | 256 |
| 11 | `lag_price_28` | lag | 699,251 | 1.21 | 707 |
| 12 | `upstream_brent_lag_7` | upstream | 663,602 | 1.14 | 267 |
| 13 | `roll_price_mean_7` | lag | 625,872 | 1.08 | 430 |
| 14 | `upstream_brent_lag_14` | upstream | 559,855 | 0.97 | 403 |
| 15 | `cal_year` | cal | 550,656 | 0.95 | 309 |
| 16 | `upstream_brent_aud_lag_14` | upstream | 542,344 | 0.94 | 401 |
| 17 | `roll_price_mean_28` | lag | 478,895 | 0.83 | 646 |
| 18 | `upstream_brent_lag_3` | upstream | 459,929 | 0.79 | 242 |
| 19 | `upstream_audusd_lag_0` | upstream | 256,013 | 0.44 | 380 |
| 20 | `xfuel_dl_price_lag_0` | lag | 236,380 | 0.41 | 292 |

### Model B' — top 20 by gain importance

| Rank | Feature | Block | Gain | Gain % | Splits |
|-----:|---------|-------|-----:|-------:|-------:|
| 1 | `lag_price_1` | lag | 30,953,745 | 52.40 | 4,802 |
| 2 | `lag_price_2` | lag | 8,144,796 | 13.79 | 2,459 |
| 3 | `days_since_last_price_change` | lag | 3,318,463 | 5.62 | 3,432 |
| 4 | `upstream_brent_lag_0` | upstream | 1,644,171 | 2.78 | 782 |
| 5 | `lag_price_3` | lag | 1,640,034 | 2.78 | 1,144 |
| 6 | `price_minus_28d_min` | lag | 1,431,196 | 2.42 | 1,608 |
| 7 | `price_minus_28d_max` | lag | 1,231,700 | 2.09 | 1,404 |
| 8 | `cal_day_of_month` | cal | 1,096,830 | 1.86 | 1,753 |
| 9 | `upstream_brent_aud_lag_0` | upstream | 961,211 | 1.63 | 782 |
| 10 | `lag_price_28` | lag | 803,115 | 1.36 | 1,249 |
| 11 | `upstream_brent_lag_1` | upstream | 760,311 | 1.29 | 479 |
| 12 | `roll_price_mean_28` | lag | 604,430 | 1.02 | 1,164 |
| 13 | `roll_price_mean_7` | lag | 559,426 | 0.95 | 702 |
| 14 | `upstream_brent_lag_14` | upstream | 533,706 | 0.90 | 821 |
| 15 | `cal_year` | cal | 516,673 | 0.87 | 464 |
| 16 | `upstream_brent_lag_3` | upstream | 494,097 | 0.84 | 485 |
| 17 | `upstream_brent_aud_lag_14` | upstream | 407,670 | 0.69 | 624 |
| 18 | `upstream_brent_lag_7` | upstream | 349,259 | 0.59 | 447 |
| 19 | `upstream_audusd_lag_0` | upstream | 323,637 | 0.55 | 829 |
| 20 | `stn_nearest_venue_km` | venue | 320,122 | 0.54 | 630 |

### Where SA2 features rank in Model B

| SA2 feature | Rank in B | Gain | Gain % |
|-------------|----------:|-----:|-------:|
| `sa2_pct_drive_to_work` | 31 | 76,160 | 0.13 |
| `sa2_dss_parenting_payment_partnered_recipients` | 42 | 25,643 | 0.04 |
| `sa2_seifa_ieo_score` | 46 | 10,756 | 0.02 |
| `sa2_median_age` | 48 | 9,585 | 0.02 |
| `sa2_pct_aged_65_plus` | 49 | 9,398 | 0.02 |
| `sa2_seifa_irsd_score` | 54 | 6,530 | 0.01 |
| `sa2_dss_carer_allowance_recipients` | 56 | 6,025 | 0.01 |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | 57 | 5,451 | 0.01 |
| `sa2_dss_carer_payment_recipients` | 58 | 4,687 | 0.01 |
| `sa2_pct_renters` | 60 | 2,935 | 0.01 |
| `sa2_pct_employed_full_time` | 62 | 2,723 | 0.00 |
| `sa2_motor_vehicles_per_dwelling` | 66 | 1,755 | 0.00 |
| `sa2_pct_one_parent_family` | 67 | 1,318 | 0.00 |
| `sa2_median_household_income_weekly` | 68 | 668 | 0.00 |
| `sa2_total_population` | 69 | 327 | 0.00 |

### Where VENUE-block features rank in Model B' (spec §13.6 Phase 1)

| Venue feature | Rank in B' | Gain | Gain % |
|---------------|-----------:|-----:|-------:|
| `stn_nearest_venue_km` | 20 | 320,122 | 0.54 |
| `stn_nearest_venue_capacity` | 53 | 6,981 | 0.01 |
| `stn_nearest_venue_type` | 70 | 635 | 0.00 |
| `stn_n_venues_within_5km` | 75 | 364 | 0.00 |
| `cal_is_pre_long_weekend` | 79 | 180 | 0.00 |

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
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t2` | -0.540 | wx |
| `sa2_motor_vehicles_per_dwelling` | `stn_competitors_within_5km` | -0.537 | stn |
| `sa2_motor_vehicles_per_dwelling` | `ctx_traffic_5km_radius_count` | -0.535 | ctx |
| `sa2_motor_vehicles_per_dwelling` | `stn_competitors_within_2km` | -0.530 | stn |
| `sa2_seifa_ieo_score` | `wx_temp_min_c_t4` | +0.525 | wx |
| `sa2_median_household_income_weekly` | `wx_temp_min_c_t2` | +0.521 | wx |
| `sa2_median_age` | `stn_competitors_within_5km` | -0.516 | stn |
| `sa2_pct_aged_65_plus` | `wx_temp_min_c_t3` | -0.515 | wx |
| `sa2_median_age` | `wx_temp_min_c_t2` | -0.514 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t3` | -0.507 | wx |
| `sa2_pct_employed_full_time` | `wx_temp_max_c_t2` | -0.506 | wx |
| `sa2_pct_drive_to_work` | `wx_temp_max_c_t6` | -0.504 | wx |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` | `ctx_traffic_5km_radius_count` | +0.502 | ctx |

---

_Generated by `python -m fuel_pred.evaluate.compare`. Re-run after
`make train` to refresh; predictions are read from
`models/predictions_*.parquet` rather than re-loading the pickles
for speed._
