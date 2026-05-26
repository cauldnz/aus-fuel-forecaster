# Weather leakage fix — pre-flight verification

**Date:** 2026-05-26
**Source doc:** [`2026-05_weather_leakage_fix.md`](2026-05_weather_leakage_fix.md)
**Status:** PASS with caveats

## Why this exists

Three empirical claims from the source research doc needed real-world verification before committing to the v2 implementation PR. This is a small (~30 min) low-commitment probe to lock the design.

## Test 1 — Coverage boundary

Probed the Historical Forecast API (`https://historical-forecast-api.open-meteo.com/v1/forecast`) at four NSW coordinates spanning the geographic range. Daily aggregates: `temperature_2m_max, temperature_2m_min, precipitation_sum, wind_speed_10m_max, weather_code`. Timezone `Australia/Sydney`.

Each cell counts non-null values out of the 5 requested variables on that date (single-day range request, which is how the bracketing was done):

| Location | Coordinate | 2016-12-31 | 2017-01-01 | 2017-01-02 | 2017-01-07 | 2017-01-15 | Verdict |
|---|---|---|---|---|---|---|---|
| Sydney | -33.87, 151.21 | null (range fetch) / data (single-day) | 4/5 (precip null) | 5/5 | 5/5 | 5/5 | start 2017-01-01 |
| Broken Hill | -31.95, 141.45 | null (range fetch) / data (single-day) | 4/5 (precip null) | 5/5 | 5/5 | 5/5 | start 2017-01-01 |
| Tweed Heads | -28.18, 153.55 | null (range fetch) / data (single-day) | 4/5 (precip null) | 5/5 | 5/5 | 5/5 | start 2017-01-01 |
| Eden | -37.07, 149.90 | null (range fetch) / data (single-day) | 4/5 (precip null) | 5/5 | 5/5 | 5/5 | start 2017-01-01 |

I also probed 2016-09-01, 2016-10-01, 2016-11-01, 2016-12-01, 2016-12-15 for all four coordinates — every cell was `0/5 non-null`.

Two interesting empirical wrinkles (neither blocks the design, but the spec should know):

1. **Single-day vs range fetch inconsistency on the boundary.** Requesting `start_date=2016-12-31&end_date=2016-12-31` returns 4/5 populated (precip null). Requesting `start_date=2016-12-15&end_date=2017-01-05` returns `None` for 2016-12-31 in the same response. This is almost certainly a timezone-boundary stitch artefact — when the request range crosses 2017-01-01 in `Australia/Sydney`, 2016-12-31 falls outside the run's coverage. Treat **2017-01-01 as the reliable boundary** and don't try to salvage 2016-12-31.
2. **Precipitation specifically is null on 2017-01-01** at all four coordinates (other four variables populated). Reliable for all five variables from **2017-01-02 onwards**. Marginal — one day of partial coverage. The fetcher's hybrid logic should treat 2017-01-01 as the first forecast-API row and accept the precip-null cell as a single-day quirk (LightGBM handles nulls natively).

**Verdict:** The boundary is **global** at the level we care about (all four NSW coordinates behave identically). The source doc's claim of "2017-01-01" is correct as the design boundary.

**Recommendation for `WEATHER_FORECAST_COVERAGE_START` config constant:** `"2017-01-01"` (no change from the source doc's recommendation). The hybrid fetcher uses ERA5 for `[2016-09-01, 2016-12-31]` and Historical Forecast for `[2017-01-01, end]`. The one-day precip nibble on 2017-01-01 is tolerable.

## Test 2 — Auth and rate limit at scale

100 sequential requests, ~0.1s spacing, across 10 coordinates × 25 month-windows spanning 2017-Q1 through 2024-Q3. Total wall-clock 150.0s (matches the docstring's claim of ~600/min upper bound at this spacing).

- HTTP 200: **100**
- HTTP 401: 0
- HTTP 403: 0
- HTTP 429: 0
- Other: 0

No `User-Agent` rejection, no rate-limit kick-in, no auth challenge. Free tier handles the v1 station-fetch workload (~1,500 stations ≈ 150s) comfortably. The pricing-page ambiguity flagged in the source doc's R3 is resolved: free tier is sufficient.

**Verdict:** Free tier is sufficient. No API key registration required.

## Test 3 — Multi-day lead time for v2 7-day horizon

Read the Historical Forecast API docs end-to-end and probed both the Historical Forecast API and the sibling **Previous Runs API**. The picture is more nuanced than the v2 7-day horizon doc currently assumes, and this Test 3 is the most important finding of the pre-flight.

**Historical Forecast API (the source-doc's target for v2.0):**
- Returns a continuous timeseries stitched from each run's first few hours. Daily aggregates effectively represent ~0–24h lead-time only.
- The docs are explicit: *"Each run's first few hours are stitched into a continuous hourly timeseries. To access the full forecast horizon of individual runs, use the Single Runs API."*
- **It does NOT expose 2-day through 7-day-ahead daily aggregates as separate variables.** The `&models=` parameter selects NWP source (ECMWF IFS, BOM ACCESS, GFS, etc.) and the `start_date` / `end_date` parameters select valid dates — but lead-time is not a query dimension.
- Suitable for v2.0 (the 1-day leakage fix). The values it returns are exactly what a deployed forecaster would have had for the day-ahead.

**Previous Runs API** (`https://previous-runs-api.open-meteo.com/v1/forecast`):
- *Same NWP models, same parameter list*, but exposes each variable at fixed lead-time offsets via `_previous_day1` through `_previous_day7` suffixes.
- Critically: the lead-time suffixes work on the **hourly** endpoint only, not the **daily** endpoint. A request for `daily=temperature_2m_max_previous_day1` returns HTTP 400; the same suffix on `hourly=temperature_2m_previous_day1` succeeds. Daily aggregates at lead-time k must be computed client-side from the hourly response.
- Coverage starts **January 2024** for most models (GFS 2m temp extends back to March 2021; JMA from 2018). This is much shallower than the Historical Forecast API.
- Empirically verified for `(-33.87, 151.21)`, `start_date=2024-06-01`, `end_date=2024-06-02`: all four weather variables × all 7 lead-day offsets populate cleanly (48/48 hourly slots non-null), with one exception — `weather_code_previous_day7` returns all nulls (weather code only available at lead-times 1–6 for the Best Match model).

**Implications for the v2 7-day horizon doc (`2026-05_7day_forecast_horizon.md` §"Fetcher impact"):**

The current 7-day horizon doc (line 115) says: *"a request for `start_date=2024-01-15, end_date=2024-01-15` returns a single daily row; to get t+7 weather as known on t, the fetcher requests t+7's valid-date row and joins it onto the panel row at date t. Same API, same call signature, same variables."*

**This is incorrect.** Requesting `valid_date = t+7` from the Historical Forecast API returns the ~0–24h-lead-time forecast that was valid for t+7, *not* the 7-day-ahead forecast issued on t. The HFA's stitched timeseries doesn't preserve longer lead times. To get the t+7 forecast as known on t, the 7-day horizon work needs the **Previous Runs API** with `hourly=<var>_previous_day7`, aggregated client-side to daily.

**Verdict for v2 §13.8 (7-day horizon):** Needs a different API for the multi-day lead-time weather. Specifically:
- **Lead-time 1 (v2.0 weather leakage fix):** Historical Forecast API, 2017-01-01 → present. As planned in §13.7.
- **Lead-times 2–7 (v2.1 horizon expansion):** Previous Runs API, January 2024 → present. Hourly endpoint, client-side daily aggregation. `weather_code` only available at lead-times 1–6.
- **Training-data implication:** Lead-times 2–7 weather features will only be available for ~2 years of training data (2024–2026). Pre-2024 training rows must either (a) drop wx_t2..t7 features entirely (LightGBM null-handling), (b) backfill with ERA5 persistence as a coarse stand-in, or (c) accept that the longer-horizon models train on a much smaller weather window. **This is a substantive constraint on the 7-day horizon plan** and should be added as a risk in `2026-05_7day_forecast_horizon.md` before v2.1 starts.

The v2.0 weather leakage fix (this preflight's primary scope) is unaffected — the Historical Forecast API serves the 1-day case correctly.

## Recommendation

**PROCEED with the v2 implementation PR as specified** for the §13.7 (1-day leakage fix) work, with one minor tolerance:

- Treat the first-day partial coverage (precip null on 2017-01-01) as expected; do not gate fetcher acceptance on full 5/5 coverage for the boundary row.
- Lock `WEATHER_FORECAST_COVERAGE_START = "2017-01-01"` (matches source doc; verified globally consistent across NSW).
- No API key registration required.
- Burst test confirms free tier is sufficient.

**Separately, flag to whoever picks up §13.8 (7-day horizon):** the assumption in `2026-05_7day_forecast_horizon.md` §"Fetcher impact" that the Historical Forecast API can serve multi-day-ahead forecasts is wrong. The 7-day work needs the Previous Runs API for lead-times 2–7, with the consequence that those features are only available for training rows from January 2024 onwards.

## See also

- `docs/research/2026-05_weather_leakage_fix.md` — the full implementation plan this probe verifies
- `docs/research/2026-05_7day_forecast_horizon.md` — the parallel v2 horizon-extension planning (Fetcher impact section needs revision per Test 3)
- `src/fuel_pred/fetch/weather.py` — the current fetcher that will be modified
- Open-Meteo Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api
- Open-Meteo Previous Runs API: https://open-meteo.com/en/docs/previous-runs-api
