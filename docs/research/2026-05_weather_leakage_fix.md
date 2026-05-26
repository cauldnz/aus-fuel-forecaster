# Weather leakage fix — implementation plan

**Date:** 2026-05
**Status:** research complete, ready to scope as v2 work
**Spec section:** `spec.md` §7.6 (current state — to be updated), proposed new §13.7 (this plan)
**Related:** `results/README.md` caveat #4 (the v1 known compromise)

## TL;DR

v1 joins ERA5 reanalysis weather *actuals* onto the panel at the same date as the prediction row — this is leakage because in real deployment you would have a *forecast* for tomorrow, not retrospective truth. The fix is to swap the Open-Meteo Archive API for the Historical Forecast API and shift the join key by one day. Coverage starts 2017-01-01 for Australia (empirically confirmed), creating a 4-month ERA5-fallback window for late 2016. A-vs-B comparison is unaffected; absolute MAE will rise modestly (estimated 0.05–0.15 c/L).

## The leakage problem (precise diagnosis)

In `add_weather_features()` (`src/fuel_pred/build/make_features.py`), weather is joined on `["station_id", "date"]`. The weather Parquet for station `s` contains one row per calendar date, where values are ERA5 reanalysis actuals for that date.

The panel row for `(station_id=s, date=t)` therefore receives:
- `wx_temp_max_c` = ERA5 actual max temperature on day `t`
- `wx_temp_min_c` = ERA5 actual min temperature on day `t`
- `wx_precipitation_mm` = ERA5 actual precipitation on day `t`
- `wx_wind_speed_max_kmh` = ERA5 actual max wind on day `t`
- `wx_weather_code` = ERA5 actual WMO code on day `t`

The target for the same row is `y_t1 = price_mean at t+1`.

**Why this is leakage:** on prediction day `t` in deployment, a forecaster knows weather actuals through `t-1` and only has a *forecast* for `t+1`. ERA5 actuals for `t` are not known until several days later (ERA5 has ~5-day publication lag, which the fetcher's `_clamp_end_to_yesterday()` already acknowledges). The values currently in the matrix are retrospectively accurate — closer to ground truth than any operational forecast could be. The model learns to use today's highly-accurate ERA5 as a proxy for tomorrow's weather.

**The correct join for t+1 horizon:** for a feature row at date `t` predicting `y_t1` (price at `t+1`), we want the weather *forecast* for `t+1` as it would have been issued at or before day `t`. The Open-Meteo Historical Forecast API serves exactly this — the actual NWP forecast that was operationally issued for that date, not a reanalysis.

## Historical Forecast API verdict

**Endpoint:** `https://historical-forecast-api.open-meteo.com/v1/forecast`

**Call signature:** Identical to the archive API — `latitude`, `longitude`, `start_date`, `end_date`, `daily`, `timezone`. No `forecast_days` or `past_days` needed for bulk historical fetches.

**Variables available (all 5 current `wx_*` confirmed):**
- `temperature_2m_max` → `wx_temp_max_c`
- `temperature_2m_min` → `wx_temp_min_c`
- `precipitation_sum` → `wx_precipitation_mm`
- `wind_speed_10m_max` → `wx_wind_speed_max_kmh`
- `weather_code` → `wx_weather_code`

**Coverage — empirically probed for Sydney (-33.87, 151.21):**

| Date | Result |
|---|---|
| 2016-09-01 | All null |
| 2016-12-01 | All null |
| **2017-01-01** | **Real values** |
| 2018-01-01 | Real values |
| 2019-01-01 | Real values |
| 2020-12-01 | Real values |
| 2021-01-01 | Real values |

**Conservative boundary: coverage starts 2017-01-01 for Australian coordinates** using the default "Best Match" model. The Open-Meteo docs mention "coverage starts around 2022" — that refers to premium high-resolution models (BOM ACCESS-G, available 2024+). The free-tier fallback resolves to a global model (likely ECMWF IFS or GFS) extending further back.

**What the API returns:** the actual NWP forecast as it was operationally issued for the requested date, with lead-time approximately 0–24h baked into the daily aggregates (the API stitches early hours of each day from the run initialised the prior day). This is exactly what a deployed forecaster would have had.

**Rate limits (free tier):** 600 calls/min, 5,000/hour, 10,000/day. Current `fetch.weather` uses `DEFAULT_INTER_CALL_SECONDS = 0.1s` (≤ 600/min). With ~1,500 NSW stations, a full refetch is ~150 seconds wall-clock. Well within limits.

**API key:** Empirically not required for the Best-Match default model at Australian coordinates. The pricing page suggests "Professional API Plan" is needed but probes returned data unauthenticated. Open-Meteo's free API key is registrable in 30 seconds if required later.

**Drop-in viability: high.** Identical call signature, identical output schema, identical variable names, free-tier accessible. Only the values change (forecast vs reanalysis) and the coverage start date (~2017 vs 1940).

## Coverage gap strategy

The gap: training data runs from `2016-09-01`. The Historical Forecast API returns null for September–December 2016 (4 months, ~2.2% of training rows).

### Option 1 — Hybrid (recommended)

Use ERA5 archive for 2016-09-01 → 2016-12-31; use Historical Forecast API from 2017-01-01 onwards. The 4 months of ERA5-contaminated data are entirely within the training fold (train ≤ 2022-12-31), so val and test metrics are unaffected. With `min_data_in_leaf=200`, the model will not over-fit on 2.2% contamination. The boundary is documented; the spec is honest.

### Option 2 — Drop 2016 weather (null fill)

Set `wx_*` to null for 2016 dates. LightGBM handles nulls natively. Wastes signal for those rows. **Not recommended** — Option 1 is strictly better.

### Option 3 — ERA5 throughout with shifted join

Use ERA5 yesterday's actuals as a persistence-forecast proxy. ERA5 day `t-1` joined onto panel row `t` is genuinely available on day `t`. **Not recommended** — a real NWP day-ahead forecast (~1–2°C RMSE at Sydney) is much cleaner signal than persistence (~3–5°C RMSE), and Option 1 makes both available.

### Option 4 — Narrow training to 2017+

Drop 4 months of training data. **Not recommended** — Option 1 preserves the rows without meaningfully contaminating test metrics.

**Recommendation: Option 1 (hybrid).** Documented in the spec, with the 2017-01-01 boundary as a config constant.

## Schema changes

**No column renames.** The five `wx_*` column names stay exactly as in `WX_COLUMNS`:
```python
"wx_temp_max_c", "wx_temp_min_c", "wx_precipitation_mm",
"wx_wind_speed_max_kmh", "wx_weather_code"
```

**Values change.** `data/raw/weather/<station_id>.parquet` files contain forecast data (2017+) or ERA5 (2016) instead of pure ERA5 throughout. Parquet schema is identical.

**Cache invalidation required.** All existing `data/raw/weather/*.parquet` files must be deleted and re-fetched. `make fetch-weather --force` covers this; the Makefile `clean-all` target also handles it.

**One new config constant:**
```python
# config.py
WEATHER_FORECAST_COVERAGE_START: str = "2017-01-01"
```

**Trained model artefacts (`models/model_a.pkl`, `models/model_b.pkl`, prediction parquets, `results/comparison.md`) are invalidated.** Full pipeline re-run required: weather re-fetch (~150s) + feature build (~20 min) + training (~30–60 min). G-NAF geocoding (~85 min, the longest step) is not affected — `stations.parquet` is unchanged.

## Pipeline changes

### `src/fuel_pred/fetch/weather.py`

Add a second API URL constant:
```python
FORECAST_URL: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"  # existing
```

New function `_request_daily_forecast(lat, lon, start, end)` — same signature as `_request_daily`, same `@retry`, calls `FORECAST_URL`.

Modify `fetch_one()` for hybrid logic:
1. If `end < WEATHER_FORECAST_COVERAGE_START`: archive only (existing behaviour).
2. If `start >= WEATHER_FORECAST_COVERAGE_START`: forecast only.
3. If range straddles: archive for `[start, 2016-12-31]`, forecast for `[2017-01-01, end]`, concatenate, deduplicate on `date`.
4. Safety net: any rows where all 5 `wx_*` are null after forecast call get backfilled from archive for those specific dates.

Module docstring updated to remove the v1 "methodological compromise" language; replaced with hybrid description and the 2016 boundary note.

### `src/fuel_pred/build/make_features.py` — `add_weather_features()`

Add a 1-day shift when merging. The forecast parquet's `date` column holds the *valid date* (the day the weather occurs). To get the day-ahead forecast onto the row predicting that day, shift the join key back by 1:

```python
# Before merging:
# weather row "valid on date d" → joins onto panel row "predicts date d"
# which lives at panel.date = d - 1
wx["date"] = pd.to_datetime(wx["date"]).dt.date
wx["date"] = [d - dt.timedelta(days=1) for d in wx["date"]]
return df.merge(weather, on=["station_id", "date"], how="left")
```

After the fix: `wx_*` columns on panel row `t` contain the day-ahead NWP forecast for `t+1`, issued on `t`. That's what the model will have in deployment.

For 2016 ERA5-fallback rows, the same shift applies — yesterday's actual weather as a persistence-forecast proxy. Documented as such in the spec.

### `src/fuel_pred/config.py`

```python
# Historical Forecast API coverage start for Australia (empirically probed).
# Below this date, ERA5 archive is used as a fallback in fetch.weather.
WEATHER_FORECAST_COVERAGE_START: str = "2017-01-01"
```

### `Makefile`

No structural changes — `fetch-weather` already passes `--start` / `--end`. Add a comment on the target:
```make
# Weather: hybrid fetch — ERA5 for 2016 (forecast API has no coverage),
# Historical Forecast API from 2017 onwards. See spec §7.6 + fetch.weather.
```

### Tests

`tests/test_fetch_weather.py`:
- Add test: `fetch_one()` calls `FORECAST_URL` for dates ≥ 2017-01-01.
- Add test: `fetch_one()` calls `ARCHIVE_URL` for dates < 2017-01-01.
- Add test: straddling case calls both, concatenates, deduplicates.
- Existing archive-only tests remain valid.

`tests/test_features.py`:
- Modify the `add_weather_features()` test to verify the 1-day shift: output `wx_temp_max_c` on row `date=t` equals the input weather fixture's value for `date=t+1`.

### Documentation

- `spec.md` §7.6 — rewrite to describe the hybrid approach, the 2016 boundary, the join-shift mechanic, and expected impact.
- `spec.md` §5.1 weather row — confirm the table reads accurately (both URLs).
- `results/README.md` caveat #4 — mark the v1 leakage as resolved in v2, with the new headline figures.
- `results/comparison.md` header — note v2 figures are leakage-corrected; show v1 vs v2 absolute MAE comparison table.

## Expected impact on metrics

**Absolute MAE: rises by ~0.05–0.15 c/L** (rough upper bound). Weather features rank below lag and upstream blocks in SHAP importance (top drivers: `lag_price_1`, Brent lags, `cal_day_of_month`, brand). The `wx_*` block captures extreme-weather demand shocks (heat waves, rain) rather than routine variation — relatively rare events. Order-of-magnitude estimate: `6.0 c/L × 0.03 SHAP fraction × 0.30 RMSE inflation ≈ 0.05 c/L`. Likely in the 0.05–0.15 range.

**A-vs-B comparison: unaffected.** Both Model A and Model B receive identical `wx_*` features. The leakage fix degrades both equally. The Δ MAE between models (−0.391 on test_normal, v1 figure) measures the SA2 block's contribution and is invariant to what values the `wx_*` columns carry — there's no asymmetry introduced.

**Honesty:** v1's absolute MAE figures (6.3 c/L Model A, 5.9 c/L Model B on test_normal) are slightly optimistic. v2's corrected figures will be marginally higher. The −6.2% lift from SA2 features is unbiased either way.

## Implementation order

| Phase | Scope | Effort |
|---|---|---|
| **A** | Fetcher: add forecast endpoint, hybrid logic, hermetic tests; re-fetch all stations | ½ session |
| **B** | Feature builder: 1-day shift in `add_weather_features()`, update tests, re-build features.parquet | ½ session |
| **C** | Retrain + evaluate: `make train && make evaluate`, regenerate notebooks | 1 session (mostly wall-clock for training) |
| **D** | Documentation: update spec §7.6, results/README.md, results/comparison.md headers | ½ session |

Total: ~2.5 sessions, of which ~1 session is wall-clock for model training.

## Risks and open questions

### R1. Exact coverage boundary needs confirmation for non-coastal NSW
The probe was Sydney coordinates. The coverage boundary is global (same NWP model worldwide), so it should be consistent — but recommend one extra probe at a western NSW coordinate (e.g. Broken Hill -31.95, 141.45) for 2016-12-31 and 2017-01-01 before locking the config constant.

### R2. What the Historical Forecast API actually provides at the daily aggregate level
The API stitches each NWP run's first few hours into a continuous timeseries, then aggregates. Daily values represent ~0–24h lead-time — the closest available approximation to a day-ahead forecast at no cost. Acceptable and honest representation, but the spec should describe it as "historical forecast archive" rather than "day-ahead forecast" to avoid overpromising.

### R3. Paid plan requirement ambiguity
Open-Meteo's pricing page is unclear on whether the free-tier fallback model is genuinely free for the historical-forecast endpoint or if rate limits / auth walls will appear in production use. Recommend a burst test of 100 sequential requests before committing to the architecture. If auth is required, the free API key takes 30 seconds to register.

### R4. The 2.2% ERA5-contaminated training rows
Even with the hybrid approach, 4 months of training data carries ERA5-derived persistence values rather than real forecasts. **Decision needed:** accept the 2.2% contamination (recommended — current spec accepts 100%) or drop those rows entirely (loses signal). Accept with documentation.

### R5. Whether to add a `wx_source` diagnostic column
Adding a categorical column (`era5_persistence` / `historical_forecast`) would let analysts audit treatment per row. Must be in `EXCLUDE_FROM_FEATURES` so the model doesn't see it. **Decision needed:** include for traceability or skip for simplicity. Recommend skip — the date boundary (< 2017-01-01) is the effective indicator.

### R6. Retraining invalidates v1 model artefacts
This is unavoidable. The PR landing the fix must either regenerate artefacts or include a clear "run `make all`" note. The 2026 crisis test fold uses 2026 data — full pipeline re-run is required to get fresh comparison numbers.

## Addendum (2026-05): historical multi-day forecast investigation (for v2.1 7-day horizon)

**Why this addendum exists.** The pre-flight (`2026-05_weather_leakage_preflight.md` Test 3) discovered that the Open-Meteo Historical Forecast API only exposes ~0–24h lead time, and the sibling Previous Runs API — which does expose `_previous_dayN` lead-time suffixes — only has Australian coverage from January 2024. For the v2.1 7-day horizon work (`2026-05_7day_forecast_horizon.md` §"Fetcher impact"), that leaves ~7 of the 9.5-year training span without real multi-day-ahead forecast values. This addendum investigates alternative sources and recommends a path forward.

### Sources investigated

All probes empirical except where flagged "doc-only".

| # | Source | Endpoint / access | AU coverage start | Max lead | Resolution | Auth / rate | Format | Complexity | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Open-Meteo Previous Runs** (incumbent option for 2024+) | `https://previous-runs-api.open-meteo.com/v1/forecast` | **2024-01-01** (GFS 2m temp to 2021-03, JMA to 2018) | 7 days (hourly only; daily endpoint rejects `_previous_dayN`) | Picks best model; ~9–25 km native | Free tier sufficient | JSON | low | USE for ≥2024 |
| 2 | **Open-Meteo Ensemble API** | `https://ensemble-api.open-meteo.com/v1/ensemble` | none (≈3 months `past_days` only) | 35 d forward | as model | Free | JSON | low | SKIP — not historical |
| 3 | **NOAA GFS on AWS Open Data** (`s3://noaa-gfs-bdp-pds/`) | Anonymous S3, `pgrb2.0p25.fNNN` GRIB2 files, 4 cycles/day | **2021-04-01** for full 0.25° to f168 (earlier dates only have WAFS aviation files) | 16 d (384 h); 0.25° to f120, 0.5° to f384 | 0.25° (~28 km) | None; AWS egress free in us-east-1 | GRIB2 (~540 MB per f168 file) | **high** (GRIB parsing, ~50 GB/year for AU subset) | DEFER — only 2021-04+ and heavy lift |
| 4 | **NOAA NCEI GFS archive** (older history) | NCEI THREDDS / HAS request | 1° from 2004-02; 0.5° from 2007-01; 0.25° from 2021-02 | as model | 0.5° / 1° (~50–100 km) for the long history | Free, but "approx. six months online"; older data requires a HAS request order | GRIB2 | high (order-then-wait pipeline) | DEFER — too coarse + offline retrieval |
| 5 | **ECMWF Open Data** (`data.ecmwf.int/forecasts/`) | HTTP listing of recent runs | **~3 days rolling only** (12 most recent runs) | 10–15 d forward | 0.25° / 0.4° | Free | GRIB2 | n/a | SKIP — no historical retention |
| 6 | **Microsoft Planetary Computer `ecmwf-forecast`** | STAC + Azure Blob | **previous 30 days only** | 10–15 d forward | as ECMWF | Free, anon | GRIB2 | medium | SKIP — no historical retention |
| 7 | **BOM ACCESS-G archive** | doc-only (`bom.gov.au/nwp` request timed out from probe) | No public archive identified; available via NCI to institutional users only | n/a | 12 km regional / 25 km global | Institutional credentials | NetCDF | very high (auth + infra) | SKIP for v2.1 |
| 8 | **Visual Crossing Historical Forecast API** (commercial) | REST JSON | "50+ years" claimed; not regionally verified | model-dependent | as model | API key; free tier 1,000 records/day; $0.0001/record metered | JSON | low | DEFER — fall-back only |
| 9 | **Day-1-proxy (controlled-leakage workaround)** | Use Open-Meteo HFA day-1 forecast as a proxy for `wx_*_tk` (k=2..7) on rows where Previous Runs API has no coverage | 2017-01-01 (HFA coverage) | n/a (proxy) | as HFA | as HFA | JSON | trivial | USE as the pre-2024 backfill — see below |

Notes on the GFS S3 probe: I confirmed empirically that `gfs.20210101/12/atmos/` is empty and that `gfs.t12z.pgrb2.0p25.f168` returns 404 for dates before 2021-04-01 but succeeds from 2021-04-01 onwards (single representative file size = 543,799,141 bytes for 2024-06-01 12Z f168). Bracketed at month boundaries 2021-03-01 (no), 2021-03-15 (no), 2021-04-01 (yes) — strict bound, no earlier surface 0.25° data on this bucket.

### The "day-1 proxy" worst case

**Concretely.** For a panel row at date `t` with target `y_tk` (k ∈ {2..7}) and no Previous Runs API coverage, populate `wx_*_tk` from the **same** Historical Forecast API value the t+1 model is using for that day — i.e. the day-ahead forecast for date `t+1` valid on day `t`, re-used as the "best-effort proxy" for what the k-day-ahead forecast on day `t` would have said. The same value populates `wx_*_t2` through `wx_*_t7`. (Variant: use the HFA value for valid-date `t+k`, which is the day-ahead-of-`t+k` forecast — i.e. issued on day `t+k-1`. That's better for the model's signal but is leakage from the row's prediction time — pick one and document it.)

**Leakage implications.** The day-1 forecast is roughly RMSE 1–2 °C for daily max temp in mid-latitudes; the day-7 forecast is ~3–5 °C (WeatherBench-class scorecards, IFS HRES is the gold standard, GFS slightly worse). Re-using day-1 as day-7 therefore feeds the model a *too-confident* weather signal — about half the noise it would see in deployment. The model will weight `wx_*_t7` more heavily than it should, mildly inflating headline metrics for the longer horizons. This is bounded — weather is a low-rank feature block (SHAP fraction ~3% per the main plan's "Expected impact" section) — so the headline drift is likely <0.1 c/L MAE at t+7, well below the noise floor of the A-vs-B Δ.

**Why it's strictly better than ERA5 actuals.** ERA5 day-`t+7` actuals are zero-noise truth; the leakage is unbounded. The day-1 forecast proxy has the noise structure of an operational forecast at the *short* end of the curve, just optimistic on the long end. ERA5 actuals would teach the model "tomorrow's price moves with future actual rainfall"; the proxy teaches it "tomorrow's price moves with what the bureau thinks will happen tomorrow". The second is much closer to deployment reality.

**When to use it.** As a stopgap to cover 2017-01-01 → 2024-01-01 (~7 years, ~74% of training rows for the 2017+ window after the existing ERA5-2016 carve-out). Not as a permanent design — the moment Previous Runs API coverage extends backwards, the pre-2024 rows should be re-fetched. Tagging the proxy rows with a `wx_lead_proxy` boolean (in `EXCLUDE_FROM_FEATURES`) preserves traceability for the eventual replacement.

### Recommendation for v2.1 (7-day horizon work)

**Use the day-1 proxy for pre-2024 training rows; use the Previous Runs API for ≥2024 rows.** Hybrid in the same shape as the v2.0 ERA5/HFA hybrid this doc already designs.

Why not the alternatives:
- **NOAA GFS on AWS Open Data** is the only free multi-day source with multi-year history, but (a) it starts 2021-04-01 — three years short of the 2017 goal anyway, (b) the GRIB2 pipeline is a category jump from JSON in operational complexity (cfgrib + xarray + per-cycle iteration + lat/lon subsetting across ~5,000 files/year), and (c) for the 2021-04 → 2024-01 window it would cover (~33% of pre-Previous-Runs training rows), the engineering cost dominates the marginal value. Defer to a v2.2 if the proxy's weather skill turns out to matter.
- **Visual Crossing** ($0.0001/record) would cost ~$5–15k for the full historical multi-day load (5 vars × 7 horizons × ~1,500 stations × ~2,500 pre-2024 days). Out of proportion to a hobby/research project budget for a feature block worth ~3% SHAP.
- **BOM ACCESS-G** has no public free archive identified. Not worth the institutional-access fight for a v2.1.

**Pipeline integration shape.** `fetch.weather` gains a third URL constant (`PREVIOUS_RUNS_URL`) and a fourth code path in `fetch_one()`:

1. `[2016-09-01, 2016-12-31]`: ERA5 (existing v2.0 fallback). All 7 lead-day columns get ERA5 value at `t+k` (same logic as the existing 2016 fallback, extended to 7 horizons).
2. `[2017-01-01, 2023-12-31]`: HFA for valid date `t+1` only. `wx_*_t2..t7` populated from that same value (day-1 proxy). `wx_lead_proxy = True` for these rows.
3. `[2024-01-01, end]`: Previous Runs API with `hourly=<var>_previous_day{1..7}`, client-side daily aggregation. `wx_lead_proxy = False`. `weather_code_previous_day7` is documented as nulls — accept.

That preserves the v2.0 1-day fetcher path unchanged and slots the multi-day work in as additive complexity. The proxy code path is ~10 lines of column duplication on top of the existing HFA call — no extra network round-trips for the 2017–2023 window.

### Impact on v2.0 weather leakage fix (this doc's main plan)

**None.** v2.0 targets the 1-day horizon only. The Historical Forecast API serves day-1 valid-date forecasts cleanly back to 2017-01-01 (preflight Test 1) — that's the variable v2.0 needs and the column it materialises. This addendum's findings only constrain v2.1 (multi-horizon), which is gated behind v2.0 in the recommended sequencing (see `2026-05_7day_forecast_horizon.md` §"Recommendation: sequencing vs §13.7").

The one cross-doc edit recommended: `2026-05_7day_forecast_horizon.md` R1 should be updated to reference this addendum's finding that the pre-2024 weather block is a *proxy*, not a real multi-day forecast, and that the per-horizon weather feature importance per horizon should be interpreted with that proxy contamination in mind for any pre-2024 SHAP analysis.

## See also

- `spec.md` §7.6 — current weather block spec (to be updated)
- `results/README.md` caveat #4 — the v1 acknowledged compromise
- `src/fuel_pred/fetch/weather.py` — current fetcher (archive only)
- `src/fuel_pred/build/make_features.py` — current `add_weather_features()` (unshifted join)
- `docs/research/2026-05_weather_leakage_preflight.md` — Test 3 finding that motivates this addendum
- `docs/research/2026-05_7day_forecast_horizon.md` — v2.1 plan affected by this addendum
- Open-Meteo Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api
- Open-Meteo Previous Runs API: https://open-meteo.com/en/docs/previous-runs-api
- NOAA GFS on AWS Open Data: https://registry.opendata.aws/noaa-gfs-bdp-pds/
- NCEI GFS product page (coverage tiers): https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
- ECMWF Open Data retention: https://www.ecmwf.int/en/forecasts/datasets/open-data
- Microsoft Planetary Computer `ecmwf-forecast` (30-day rolling, not historical): https://planetarycomputer.microsoft.com/api/stac/v1/collections/ecmwf-forecast
