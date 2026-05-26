# 7-day forecast horizon — architectural planning

**Date:** 2026-05
**Status:** research complete, queued as v2 architectural change
**Spec section:** `spec.md` §13.8
**Related:** `docs/research/2026-05_weather_leakage_fix.md` (the weather block is the most affected by multi-horizon; the leakage fix should land first)

## TL;DR

Move from a 1-day target (`y_t1`) to a per-day 7-day forecast (`y_t1`..`y_t7`). Recommended architecture: **A — one LightGBM model per horizon**, trained independently with horizon-specific weather and calendar features. Recommended sequencing: **land the §13.7 weather leakage fix first** under the existing 1-day horizon (v2.0), then build the 7-day horizon as v2.1 on top of a leakage-clean baseline. Total effort: **5–8 sessions**, dominated by training wall-clock (7× the v1 fits) and per-horizon evaluation/reporting.

## The change

**Current state (v1).** `add_targets()` in `src/fuel_pred/build/make_features.py` emits two columns per U91 row:

- `y_t1` — tomorrow's price (`groupby(['station_id','fuel_code'])['price_mean'].shift(-1)`).
- `y_t1_t7` — the *mean* of `price_mean[t+1..t+7]`. Exists in the schema but is not consumed by training; `feature_blocks.EXCLUDE_FROM_FEATURES` lists both targets and `train_models.train()` (today a TODO) is designed around `y_t1`.

**Target state (v2.1).** The pipeline emits a vector of 7 per-day targets (`y_t1`..`y_t7`) and the training stage produces 7 trained models that, together, deliver a per-day next-week price forecast. The result is a per-horizon error curve (MAE rises monotonically with lead time) — much more informative than the headline single number, and the right shape for any downstream "what will fuel cost on my Friday drive" question.

This is a substantial architectural change. It touches the target schema, the weather block, the calendar block, the train module, the evaluation module, and all three notebooks. It does *not* touch any fetcher or cleaner, the augmentor integration, the spatial joins, the lag block, the upstream block, or the station/SA2 enrichment. Most of the pipeline by line count is unaffected.

## What stays the same

Be explicit about this — the surface area of the change is smaller than it looks:

- **All fetchers** (`fetch.fuelcheck`, `brent`, `audusd`, `traffic`, `cash_rate`, `asx200`, `inflation_expectations`, `aip_tgp`). Their outputs are horizon-agnostic — they fetch raw external time series; lead-time selection happens downstream.
- **All cleaners** (`clean.fuelcheck`, `clean.traffic`).
- **Spatial joins** (`spatial.resolve_addrs`, `spatial.nearest`). Station↔counter geometry is invariant to horizon.
- **The augmentor integration** (`build.enrich_census`). SA2 demographics are time-invariant features.
- **The lag block** (`add_lag_features`, `LAG_COLUMNS`, the cross-fuel `xfuel_dl_*` joins). All lags reference past prices via `shift(n)` with `n ≥ 1`; they are independent of which future date is being predicted. See §"Lag and upstream features" below for the formal argument.
- **The upstream block** (`add_upstream_features`, `UPSTREAM_COLUMNS`). Brent / AUD-USD / TGP are not knowable at horizon t+k > t; the model can only consume what it knew at time t. Same lags work for predicting t+1 or t+7.
- **The static station block** (`add_station_features`, `STN_COLUMNS`). Brand, competitor counts, terminal distance — all time-invariant.
- **The SA2 block** (`add_sa2_features`, `SA2_COLUMNS`).
- **The traffic context block** (`ctx_traffic_*`). Yesterday's traffic is the most recent count available regardless of horizon.
- **The macro context columns** (`ctx_cash_rate`, `ctx_asx200_lag_1`, `ctx_inflation_expectations_lag_7`). Already lagged; not future-knowable.
- **Training row filter** — U91 only, target non-null, identical-rows guard against SA2 nulls (spec §8.4). Mechanics generalise straightforwardly to "every target horizon non-null".
- **Time-based folds** (spec §8.3). The train ≤ 2022-12-31 / val 2023 / test_normal 2024-25 / test_crisis 2026 boundaries hold. A row's eligibility for a fold is determined by the prediction date `t`, not by the target date `t+k`.
- **Evaluation primitives** in `evaluate.metrics` (MAE / RMSE / MAPE / median / p90). The `compare` module is the place that needs structural extension.
- **The Centrelink/SEIFA EDA chart in `01_eda.ipynb` §6.** It uses observed prices, not predictions.

## What changes

### Target schema

**Recommendation: seven explicit columns `y_t1, y_t2, y_t3, y_t4, y_t5, y_t6, y_t7` in `features.parquet`.**

```python
for k in range(1, 8):
    out[f"y_t{k}"] = grouped["price_mean"].shift(-k)
out.loc[~is_u91, [f"y_t{k}" for k in range(1, 8)]] = pd.NA
```

Rationale:

- **Stays in pandas wide-form.** No reshape to a long `(station_id, fuel_code, date, horizon, y)` panel — that would inflate `features.parquet` by 7× on disk and force every downstream consumer (notebooks, evaluators, ablation scripts) to learn a new shape.
- **Backward compatible.** `y_t1` keeps its existing meaning and existing semantics. The v1 1-day model can be retrained from the same features.parquet without code changes for direct comparison.
- **One row per `(station_id, fuel_code, date)`.** Preserves the grain established by spec §6.3, which means the existing identical-rows guard still works without refactoring.
- **Aligns with model architecture A** (one model per horizon — see below): the per-horizon model just selects its `y_tk` column.

Decision on the existing **`y_t1_t7` (7-day mean)** column: **drop it.** It was a v1 hedge — useful if v1 had ever shipped a "next-week average" forecast, but it didn't, and `mean(y_t1..y_t7)` is trivially recoverable from the per-day columns by a downstream consumer. Keeping it around as a derived sanity-check column adds maintenance burden (which `add_targets` recomputation logic owns it?) without value. **Note** for `EXCLUDE_FROM_FEATURES`: it must list `y_t1` *through* `y_t7` once renamed.

### Model architecture

**Recommendation: Architecture A — one LightGBM model per horizon (7 separate models).**

The candidates considered:

**A) One model per horizon.** Train 7 independent LightGBM regressors against `y_t1`..`y_t7`. Each model sees the same X matrix at training time (or near-same, if weather features are horizon-specific) and a different y vector. Inference is 7 `predict()` calls.

**B) One model with horizon as a categorical feature.** Replicate the panel 7 times, adding a `horizon ∈ {1..7}` column. Single LightGBM fit against a single stacked `y` column. Inference is 7 predicts against rows with different `horizon` values.

**C) Recursive forecasting.** Use the t+1 prediction as input for t+2, etc. Not pursued — recursion accumulates error fast in tree models, doesn't fit how `lag_price_*` are structured, and breaks the time-based fold guarantee.

| Dimension | A (per-horizon) | B (stacked + horizon feature) |
|---|---|---|
| Training time | 7× the v1 fit; trivially parallelisable across CPU cores | 1 fit on 7× the rows (training time ~5–7× v1 in practice, less parallel) |
| Memory | Same as v1 per model | Peak 7× v1 (the stacked panel) — non-trivial on a 32 GB box once features.parquet hits ~5–10 GB |
| Interpretability | Per-horizon SHAP for free; per-horizon importance is the natural cut | Single SHAP run, but the horizon column dominates and obscures the per-horizon picture |
| Feature handling | Each model gets horizon-specific weather / calendar columns trivially | Needs feature *values* that vary by horizon, joined per replicated row — adds a join step |
| `comparison.md` shape | Per-horizon MAE table is the natural output | Same per-horizon table, but computed from the single model's per-horizon-segment predictions |
| Identical-rows guard | Applies per model independently; semantics unchanged | Applies to the stacked panel; row count is 7× but logic is the same |
| A/B (SA2 ablation) | Cleanly doubles to 14 models (2 × 7); fold structure unchanged | One Model A + one Model B; the SA2 lift is averaged across horizons by default, needs explicit per-horizon segmentation in `compare` |
| Risk of cross-horizon contamination | None | Tree splits *could* memorise per-station-date-horizon combinations if `min_data_in_leaf` (200) is now relative to a 7× larger dataset |
| Code change shape | Loop in `train.train_models`; no make_features change beyond the target schema | Both `train` and `make_features` change (replication step); evaluation segmenter needs a horizon dimension |

The cleanest argument for A is that **the model's bias-variance trade-off is fundamentally different at different horizons** — the t+1 model should weight `lag_price_1` very heavily; the t+7 model should weight Brent / cycle features more. Forcing one parameter set to handle both via a categorical feature throws away that signal. With LightGBM's `feature_fraction=0.8`, the t+1 model will *learn* to pick up `lag_price_1` near-deterministically; the t+7 model will *learn* to discount it. Architecture B has to learn both within a single decision-tree ensemble, which is strictly harder.

Architecture A also makes per-horizon ablation trivial: "what if we drop weather for the t+7 model only?" is a code one-liner, vs a feature-conditioned masking layer in B.

Architecture B's strongest argument — shared representation across horizons — assumes there's a horizon-invariant latent structure for the model to exploit. For trees, this is a weaker assumption than for neural nets. The 7× training-time multiplier in A is fine because the fits are independent and parallelisable (`joblib.Parallel(n_jobs=-1)`); B's stacked fit doesn't parallelise as cleanly.

**Pick A.** Document B in the spec as the considered-and-rejected alternative.

### Weather features

This is the largest single design question and the most direct interaction with §13.7.

**The shape.** For Architecture A with the leakage fix already landed: each per-horizon model needs weather features *at its own horizon's valid date*. The t+1 model wants weather forecast for t+1 (issued on t). The t+7 model wants weather forecast for t+7 (issued on t).

**Recommendation: per-horizon weather columns, one set in `features.parquet`, materialised at make_features time.** Schema additions:

```
wx_temp_max_c_t1, wx_temp_max_c_t2, ..., wx_temp_max_c_t7
wx_temp_min_c_t1, ..., wx_temp_min_c_t7
wx_precipitation_mm_t1, ..., wx_precipitation_mm_t7
wx_wind_speed_max_kmh_t1, ..., wx_wind_speed_max_kmh_t7
wx_weather_code_t1, ..., wx_weather_code_t7
```

That's 5 × 7 = 35 weather columns where v1 has 5. Each per-horizon model picks its corresponding suffix block via `feature_blocks.WX_COLUMNS_T<k>`. The legacy `WX_COLUMNS` is dropped or aliased to `WX_COLUMNS_T1` for backward compatibility.

**Why per-horizon-explicit columns, not a join-by-horizon long form.** Architecture A keeps the panel in wide form; the t+1 and t+7 models share an X matrix in everything except weather. Materialising the per-horizon weather as suffix columns is the simplest way to express that. The model never sees its sibling horizons' weather; `feature_blocks.feature_columns()` slices the right subset.

**Fetcher impact (interacts with §13.7).** The Open-Meteo Historical Forecast API can return multi-day-ahead forecasts as part of the same daily-aggregate response — the API stitches whichever NWP run was operationally valid at lead time k into the daily row. Concretely: a request for `start_date=2024-01-15, end_date=2024-01-15` returns a single daily row; to get t+7 weather as known on t, the fetcher requests t+7's valid-date row and joins it onto the panel row at date t. Same API, same call signature, same variables. The fetcher therefore stays unchanged from the §13.7 plan, but `add_weather_features()` does 7 joins instead of 1, with shifts of −1 through −7 days.

**Forecast skill degrades.** NWP day-ahead RMSE for daily max temp at Sydney is ~1–2°C; day-7 RMSE is ~3–5°C. That degradation is the right thing for the model to see — `wx_temp_max_c_t7` carrying more noise than `wx_temp_max_c_t1` accurately reflects what an operational forecaster would have, and the model's learned dependence on `wx_*_t7` should be correspondingly weaker. No special handling needed; the noise is in the data.

**Dimensionality cost.** 35 weather columns out of ~95 total feature columns is non-trivial but not extreme. LightGBM's `feature_fraction=0.8` already samples 80% of columns per tree; with the 7-fold-larger weather block the per-horizon model still has plenty of non-weather features to split on. The 2.2% ERA5-fallback contamination window (2016) applies identically — the fallback proxy is "yesterday's ERA5 actuals shifted by k days", consistent across all 7 horizons.

### Calendar features

**Most calendar features are deterministic and horizon-invariant in the sense that they can be computed on the fly from `(date + k days)`.** No need to materialise 7 copies of `cal_day_of_week` — `(date + k) % 7` is one line.

The cleanest implementation: extend `add_calendar_features()` to emit per-horizon copies for the columns where the horizon date matters (`cal_day_of_week_tk`, `cal_day_of_month_tk`, `cal_day_of_fortnight_tk`, `cal_is_public_holiday_tk`, `cal_days_to_next_public_holiday_tk`, `cal_is_school_holiday_nsw_tk`). The columns that *don't* change meaningfully across a 7-day window (`cal_month`, `cal_year`, `cal_week_of_year`) are computed once at the prediction date `t` and shared across horizons.

Column count: roughly `7 horizon-dependent calendar fields × 7 horizons + 4 horizon-invariant fields = 53`. That's an 11-column block growing to ~53. Manageable.

`cal_is_first_business_day_after_break` is special — it's the first business day at `t+k`, so it needs to be computed per horizon. Easy enough; the existing helper takes a list of dates.

### Lag and upstream features

**Confirm they're invariant to horizon. They are.**

Lag features reference only past prices through `shift(n)` with `n ≥ 1`, and rolling windows use `shift(1).rolling(w)` — so the most recent observation any lag/rolling feature uses is `price_mean[t-1]`. This is the same regardless of whether we're predicting t+1 or t+7. The model's t+7 prediction is allowed to depend on `lag_price_1` (yesterday's price) because that's information genuinely available at prediction time t.

Upstream features have the same property — `upstream_brent_lag_0` is "today's Brent close" which is known at t for all horizons. There's no `upstream_brent_lag_-7` (tomorrow's Brent); we cannot use future Brent because we don't know it.

**The model legitimately uses identical X-feature values for the lag and upstream blocks across all 7 horizons.** Only the targets and the horizon-keyed weather/calendar features change.

**Crucial: re-audit `EXCLUDE_FROM_FEATURES`.** Today's `price_mean` is in it (correct — it would leak the target). `lag_price_1` is in features (correct — it's yesterday's price). All the rolling stats use `shift(1)`. Nothing flagged on a close read — but a unit test that simulates a leaky lag column at all 7 horizons would be valuable insurance.

### Evaluation

**Per-horizon MAE table** as the headline replacement for the current single-row table in `results/comparison.md`:

| Horizon | n | MAE A | MAE B | Δ MAE | rel. | RMSE A | RMSE B | MAPE A | MAPE B |
|---|---|---|---|---|---|---|---|---|---|
| t+1 | … | … | … | … | … | … | … | … | … |
| t+2 | … | … | … | … | … | … | … | … | … |
| … | | | | | | | | | |
| t+7 | … | … | … | … | … | … | … | … | … |

The t+1 row should reproduce v1's headline number (modulo the weather leakage fix). The t+7 row's MAE should be materially larger — we should see clear monotonic growth across horizons. If it's not monotonic, something is wrong.

**Error growth curve** as a chart in `02_modeling.ipynb` and `results/comparison.md`: MAE vs horizon, one line per model (A and B), faceted by SEIFA quintile to check whether the SA2 lift decays with horizon.

**Segmentation matrix.** Today's four segmentations (metro/regional, brand top-8, fuel, SEIFA quintile) become 4 × 7 = 28 tables. Either render them as long-form (recommended — one big "segmentation_by_horizon.parquet" companion artefact) or pick the t+1 and t+7 endpoints for the headline `comparison.md` and link to the full per-horizon tables.

**The crisis fold.** Still 2026-01-01 → end of data, fold structure unchanged. Per-horizon metrics computed identically. Expect crisis errors to grow faster with horizon than normal errors (the model trained on pre-crisis data has nothing to anchor 7-day forecasts against during a regime change).

**SA2 lift retention check.** Open question: does Model B's lift hold at all horizons or decay? Hypothesis: lift should be relatively *stable* with horizon, because the SA2 features (income, demographics, SEIFA) capture station-static characteristics that matter equally for any-day prediction. If the lift collapses to zero at t+7, it's evidence the SA2 block was only helping at t+1 through some interaction with `lag_price_1` we missed. Either result is a publishable finding for the project's "story".

### Notebooks

All three need updates:

- **`01_eda.ipynb`** — minor. Add a per-station autocorrelation-at-7-day-lags cell to motivate the 7-day target's coverage of the petrol cycle. The existing sections (geographic, cycle, crisis, Centrelink-day chart) are unchanged.
- **`02_modeling.ipynb`** — substantial. The "fit Model A / fit Model B" sections become loops over horizons. Per-horizon metrics tables, error-growth curve plot, per-horizon residual diagnostics. The headline write-up cell needs the per-horizon framing.
- **`03_explainability.ipynb`** — substantial. SHAP per horizon (or at minimum t+1, t+3, t+7) for both models. Per-horizon top-20 feature importance is the most interesting figure to add — does `lag_price_1` dominate t+1 SHAP but fade to mid-pack by t+7, as expected? Does `upstream_brent_lag_7` matter more at longer horizons? Case-study stations get a 7-day prediction plot.

Estimated notebook re-execution time grows 7× for model fits — this is mostly wall-clock during full pipeline runs and doesn't block development.

## Recommendation: sequencing vs §13.7

**Land §13.7 first.** Build the 7-day horizon as v2.1 on top of v2.0 (the leakage-corrected 1-day model). Reasoning:

1. **The leakage fix is a smaller, lower-risk change.** ~2.5 sessions, no schema changes beyond cache invalidation, no model architecture change. It can land, be measured (the expected 0.05–0.15 c/L MAE rise on the headline), and ship as v2.0 within a week.
2. **The 7-day work needs a clean weather baseline to interpret correctly.** If we bundle, we'd need to compare "v1 1-day leaky" against "v2.1 7-day leakage-corrected" — two changes at once, attribution of any MAE delta becomes ambiguous. With sequencing we get a clean v1→v2.0 (leakage cost) attribution and a separate v2.0→v2.1 (horizon expansion cost) attribution.
3. **The leakage fix is a prerequisite for the multi-horizon weather treatment.** The Historical Forecast API is the source of the multi-horizon weather data. Building 7-day weather joining without first migrating off ERA5 actuals means rewriting `add_weather_features()` twice.
4. **The leakage fix unblocks the §7.6 doc rewrite** that the 7-day plan would also touch. Cleaner diff if they're sequenced.
5. **v2.0 has independent value.** The leakage fix improves project honesty without depending on the horizon expansion landing.

If the 7-day work were the bigger user-facing payoff, the case for bundling would be stronger. But "7-day forecasts at the same headline quality as v1's 1-day, with leakage already corrected" is a much cleaner story to ship than "everything at once, can't tell what helped what".

**Bundle the documentation rewrites though.** Both changes touch `spec.md` §7.6 (weather) and `results/README.md`. When v2.1 lands, the docs only need one final rewrite rather than two interim states.

## Estimated effort

This is honestly a multi-session change. Don't undersell.

| Phase | Scope | Effort |
|---|---|---|
| **0** | Pre-req: §13.7 weather leakage fix lands as v2.0. (Separate work — counted under that backlog entry, not this one.) | (out of scope here) |
| **1** | Target schema: extend `add_targets()` to emit `y_t1`..`y_t7`; drop `y_t1_t7`; update `EXCLUDE_FROM_FEATURES`; rebuild `features.parquet`. Unit tests on the shift semantics. | ½ session |
| **2** | Multi-horizon weather: modify `add_weather_features()` to do 7 joins; add 35 columns; add `WX_COLUMNS_T1`..`T7` to `feature_blocks`. Fetcher unchanged — same Historical Forecast API. Tests on join correctness at each horizon. | 1 session |
| **3** | Multi-horizon calendar: extend `add_calendar_features()` to emit per-horizon copies of the horizon-dependent fields; add `CAL_COLUMNS_T1`..`T7`. Tests. | ½ session |
| **4** | `feature_blocks.feature_columns()` rework: take a `horizon: int` parameter, select the right per-horizon WX + CAL columns, share the rest. Tests. | ½ session |
| **5** | `train.train_models`: loop over horizons, train 7 × {A, B} = 14 models, persist each with horizon-suffixed filenames. Parallel via `joblib.Parallel`. Update `feature_lists.json` schema. | 1 session (mostly wall-clock for the 14 fits) |
| **6** | `evaluate.compare`: per-horizon segmentation; error growth curve generation; per-horizon `predictions_*.parquet` artefacts; rewrite `results/comparison.md` template. | 1 session |
| **7** | Notebooks: update `02_modeling.ipynb` for per-horizon analysis (loop over horizons, error-growth chart). Update `03_explainability.ipynb` for per-horizon SHAP at t+1/t+3/t+7. Minor `01_eda.ipynb` autocorrelation addition. | 1.5 sessions |
| **8** | Documentation: rewrite spec §3 (target definition), §7.8 (target generation), §8.3-8.5 (training/eval), `results/README.md`. Combine with §13.7's pending doc rewrites. | ½ session |

**Total: 6 sessions plus ~½ session of wall-clock for training the 14 models.** Round to **5–8 sessions** acknowledging that integration issues, the inevitable cross-horizon evaluation bug, and the notebook re-runs cumulatively slip schedule. If anything goes sideways with weather forecast skill at long horizons or LightGBM overfitting per-horizon, add another session for diagnostics.

## Backward compatibility

**Keep `y_t1` semantics unchanged.** A "v2.0 model" (1-day, leakage-corrected) and a "v2.1 t+1 model" should be exactly comparable — same training data, same fold, same target column, same metrics. `evaluate.compare` should emit a "v1 vs v2.0 vs v2.1 t+1" three-way comparison in the first iteration of the new `comparison.md`, so the project has a clean attribution of "what did the leakage fix cost" and "what did the horizon expansion cost (at the t+1 row)".

The v1 model artefacts (`models/model_a.pkl`, `models/model_b.pkl`) should be preserved in a `models/v1/` subdirectory before being overwritten — the comparison value is high and the disk cost is trivial.

## Risks and open questions

### R1. Weather forecast skill at t+7 may degrade enough that the column is useless

Day-7 NWP temperature RMSE around 3–5°C is high relative to v1's day-1 ERA5 (essentially zero noise). LightGBM may simply ignore `wx_*_t7` columns. If so, the per-horizon weather block is wasted complexity. **Mitigation:** measure feature importance per horizon; if `wx_*_t7` ranks below position 80, drop horizons 5–7 weather features explicitly and document the asymmetric treatment.

### R2. Training time and memory on 32 GB

14 LightGBM fits with current row counts (~5M for train) is ~30–60 min wall-clock if serialised, faster with `joblib.Parallel(n_jobs=-1)` on an 8-core machine. Memory is the bigger constraint: holding 7 model artefacts plus the X matrix at peak is non-trivial. **Mitigation:** train horizons sequentially, persist each model immediately, free LightGBM booster between fits. Architecture A's independence makes this trivial.

### R3. The identical-rows guard becomes stricter

Spec §8.4: training uses rows where every SA2 column is non-null. With 7 targets, training uses rows where every SA2 column *and* every `y_tk` column is non-null. The latter constraint drops a *7-day* tail off the end of every station's history rather than v1's 1-day tail. The effective training-row count drops by approximately 6 days × ~2,500 stations ≈ 15,000 rows — negligible relative to the ~5M total. Worth confirming empirically before committing.

### R4. Cross-horizon evaluation comparability

The current v1 `comparison.md` is one table. The v2.1 `comparison.md` is at minimum 7× larger. Risk: information overload obscures the key headline (Δ MAE A vs B). **Mitigation:** keep the headline as a 7-row table (per horizon), put per-horizon segmentation tables in a separate auto-generated `comparison_by_horizon.md` linked from the headline.

### R5. Per-horizon SHAP cost

SHAP TreeExplainer is fast but not free. Computing SHAP for 14 models on a 100k-row evaluation sample is ~10× the v1 work. **Mitigation:** compute full SHAP for t+1, t+3, t+7 only — the interesting cross-sections — and report importance-only for the other horizons.

### R6. The `y_t1_t7` mean column may have value as a "stability" predictor

Dropping `y_t1_t7` is technically reversible (any consumer can recompute it). But the column has been in the schema since v1 and might be relied on by ad-hoc analysis scripts in the repo. **Decision needed:** drop entirely vs keep as a derived sanity column emitted by `add_targets()`. **Recommendation: drop.** Recompute is trivial; the schema is cleaner without it.

### R7. Architecture A vs B ambiguity on small-sample SA2 segments

In sparse segments (e.g. SEIFA Q1 metro stations, few rows), 14 independent fits may overfit per-horizon vs B's 1 shared fit. The H1 hypothesis "A is uniformly better" assumes enough training data per horizon to support the per-horizon decoupling. If a segment has < 10k training rows, B's shared fit might generalise better there. **Mitigation:** measure A's per-horizon-per-segment confidence intervals; if any segment shows wide intervals, document as an A-architecture limitation in `results/comparison.md`.

### R8. Operational deployment semantics

If the project ever exits "methodology demo" status, a 7-day daily forecast means generating predictions at time `t` that depend on Open-Meteo weather forecasts at times `t+1`..`t+7`. Open-Meteo's Forecast API (not Historical Forecast — different endpoint) serves these. The training-vs-deployment endpoint mismatch is intentional and well-handled by Open-Meteo's documentation, but worth flagging in `spec.md` §3 if v2.1 is the basis for any operational claim.

## See also

- `spec.md` §13.8 — this entry
- `docs/research/2026-05_weather_leakage_fix.md` — the §13.7 work; prerequisite
- `src/fuel_pred/build/make_features.py` — `add_targets()`, `add_weather_features()`, `add_calendar_features()`
- `src/fuel_pred/train/train_models.py` — model fit logic (currently `NotImplementedError`; the per-horizon loop lives here)
- `src/fuel_pred/train/feature_blocks.py` — `LAG_COLUMNS`, `UPSTREAM_COLUMNS`, `CALENDAR_COLUMNS`, `WX_COLUMNS`, `EXCLUDE_FROM_FEATURES`; will gain per-horizon variants
- `results/README.md` — v1 result baseline; needs the per-horizon table treatment
- Open-Meteo Historical Forecast API multi-day-ahead documentation: https://open-meteo.com/en/docs/historical-forecast-api
