# NSW Fuel Price Prediction — Specification

**Status**: v1 design — pre-implementation
**Source of truth**: this document. Code that disagrees with `spec.md` is a bug; design changes are made by editing `spec.md` first.

---

## 1. Purpose

Build a regression model that predicts daily retail fuel prices at NSW service stations and use it to demonstrate that augmenting per-station features with SA2-level Australian Census demographic variables (via the [`abs-census-augmentor`](https://github.com/cauldnz/abs-census-augmentor) library) measurably improves predictive performance.

The project trains two LightGBM models with identical pipelines except for one feature block, and reports the lift on held-out future data. The "story" of the project is the comparison.

This is a methodology demonstration, not a production forecasting system.

## 2. Acceptance Criteria

The project is "done" when all of the following hold:

1. A single command (`make all` or `uv run -- inv all`) reproduces the entire pipeline end-to-end on a clean checkout, given access to the network for raw fetches (or raw cache pre-populated).
2. A processed feature matrix exists at `data/processed/features.parquet` with the schema documented in §6, covering at minimum 2016-09 → most recent complete month, U91 + Diesel, all NSW stations with a successful G-NAF resolution.
3. Two trained models (A: no SA2 features, B: with SA2 features) are saved to `models/` with identical hyperparameters and identical training rows.
4. Three notebooks exist and run cleanly top-to-bottom against `data/processed/features.parquet`:
   - `notebooks/01_eda.ipynb`
   - `notebooks/02_modeling.ipynb`
   - `notebooks/03_explainability.ipynb`
5. A summary report at `results/comparison.md` reports MAE / RMSE / MAPE for both models on the holdout, segmented by metro / regional, brand, and fuel type.
6. SHAP outputs (summary plot, top-feature dependence plots, interaction plots for `day_of_fortnight × seifa`) saved to `results/shap/`.
7. Test suite passes (`pytest`) with hermetic tests for all pipeline modules — no real network calls in CI.

## 3. Scope

**In scope (v1)**
- NSW only (FuelCheck)
- Daily granularity per `(station_id, fuel_code)`
- Two prediction horizons: `t+1` (primary), `t+1..t+7` (secondary, optional in v1)
- **Forecast target: U91 only** (Unleaded 91). Diesel (`DL`) data is still cleaned and persisted in `fuel_daily.parquet` because cross-fuel lags / co-movement at the same station may be predictive features for U91; see §7.1. The headline A/B comparison and `results/comparison.md` report U91 only.
- Historical span: **2016-09-01 → most recent complete month**
- Train on local machine (≤ 32 GB RAM, no GPU required)

**Out of scope (v1)**
- Other states (VIC, WA FuelWatch)
- Forecasting fuels other than U91 (E10, U95, U98, Diesel, LPG). Diesel is still ingested as a candidate feature.
- Hourly granularity
- Probabilistic / uncertainty estimates
- Real-time inference, deployment, web/mobile UI
- Cross-validation of the SA2 input variables themselves (forward selection on the augmentor's variable catalogue)
- Counterfactual / causal claims about pass-through dynamics

## 4. Architecture

The pipeline is a DAG of small, single-purpose modules. Each module is a Python file under `src/` with a CLI entrypoint and is independently invokable. Intermediate artefacts are written to `data/interim/` as Parquet so any node can be re-run without re-running its predecessors.

```
                       ┌──────────────────┐
                       │   src/fetch/*    │
                       └────────┬─────────┘
                                ↓
                       ┌──────────────────┐
                       │   src/clean/*    │
                       └────────┬─────────┘
                                ↓
                ┌───────────────┴────────────────┐
                ↓                                ↓
       ┌──────────────────┐            ┌──────────────────┐
       │ src/spatial/     │            │ src/build/       │
       │ resolve_addrs    │            │ panel_grid       │
       │ (G-NAF → SA2)    │            │ (station × day)  │
       └────────┬─────────┘            └────────┬─────────┘
                ↓                                ↓
       ┌──────────────────┐            ┌──────────────────┐
       │ src/build/       │            │ src/build/       │
       │ enrich_census    │ ─────────→ │ make_features    │
       │ (augmentor)      │            └────────┬─────────┘
       └──────────────────┘                     ↓
                                       ┌──────────────────┐
                                       │ features.parquet │
                                       └────────┬─────────┘
                                                ↓
            ┌───────────────────────────────────┼───────────────────────────────────┐
            ↓                                   ↓                                   ↓
   ┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
   │ notebooks/       │               │ src/train/       │               │ notebooks/       │
   │ 01_eda.ipynb     │               │ train_models.py  │               │ 03_explain.ipynb │
   └──────────────────┘               └────────┬─────────┘               └──────────────────┘
                                               ↓
                                      ┌──────────────────┐
                                      │ models/{a,b}.pkl │
                                      │ results/*.md     │
                                      └──────────────────┘
```

Caching philosophy: every fetcher writes to `data/raw/<source>/` with a deterministic filename (date-stamped or content-hashed). Re-runs are cheap. Cleaners read from `data/raw/`, write to `data/interim/`. Feature builder reads from `data/interim/`, writes `data/processed/features.parquet`.

## 5. Data Sources

### 5.1 Tier 1 — required

| Source | URL / API | Format | Granularity | Coverage |
|---|---|---|---|---|
| **NSW FuelCheck Price History** | https://data.nsw.gov.au/data/dataset/fuel-check (CKAN package; one resource per month) | mostly XLSX, some CSV — see §5.1.1 | per-update events | 2016-09 → present |
| **Brent crude (futures continuous)** | `yfinance` ticker `BZ=F` | OHLCV | daily | 2000-01 → present |
| **AUD/USD** | RBA F11.1 historical, https://www.rba.gov.au/statistics/historical-data.html#exchange-rates | per-period XLS + one current CSV — see §5.1.2 | daily | 1983 → present (XLS); 2023 → present (CSV) |
| **NSW Roads Traffic Volume Counts** | https://opendata.transport.nsw.gov.au/data/dataset/nsw-roads-traffic-volume-counts-api | CKAN datastore (stations) + ZIP of CSVs (hourly) — see §5.1.3 | hourly per station | 2006 → present |
| **Australian public holidays** | `python-holidays` package | code | daily | unbounded |
| **NSW school terms** | manual `data/static/nsw_school_terms.csv`, sourced from NSW Education term-dates page | CSV | term-boundary dates | 2016 → present |
| **Open-Meteo weather** | `https://archive-api.open-meteo.com/v1/archive` (Historical Weather, ERA5) and `https://historical-forecast-api.open-meteo.com/v1/forecast` (Historical Forecast) | JSON | daily aggregates per lat/lon | 1940 → present (archive); 2021 → present (forecast) |

#### 5.1.1 NSW FuelCheck — actual resource layout (verified May 2026)

The CKAN package `fuel-check` lists ~115 monthly archives. Names follow `Service Station Price History - <Month> <Year>` or `FuelCheck Price History <MonYYYY>`; URL filenames follow `fuelcheck_pricehistory_<mon><yyyy>.xlsx` or `price_history_checks_<mon><yyyy>.csv`. Format breakdown across the 113 data resources (excluding the FAQ + DQS resources):

- ~94 are `xlsx`
- 8 are `csv`
- A handful have an empty `format` field — trust the URL extension (`.csv` or `.xlsx`).

`fetch.fuelcheck` downloads each monthly resource verbatim and writes one Parquet per month (`<YYYY-MM>.parquet`). Schema normalisation is the cleaner's job — column renames have happened over the years (e.g. `ServiceStationName` ↔ `service_station_name`, `PriceUpdatedDate` in `YYYY/MM/DD HH:MM:SS` vs ISO 8601).

#### 5.1.2 RBA F11.1 — actual resource layout (verified May 2026)

The historical-data page lists 11 legacy `.xls` files (one per ~3-year period from 1983-1986 through 2018-2022) plus one rolling `.csv` for 2023-current at `https://www.rba.gov.au/statistics/tables/csv/f11.1-data.csv`. Both formats share the same logical layout — a multi-row preamble (Title / Description / Frequency / Type / Units / blank / Source / Publication date / Series ID) followed by data rows. The "Series ID" row identifies which column carries each series; `FXRUSD` is AUD/USD.

For the project span (2016-09 onwards) we fetch only the three files that overlap: `2014-2017.xls`, `2018-2022.xls`, and the current `.csv`. Older periods are out of scope and intentionally skipped. Reading XLS requires `xlrd>=2.0`; CSV requires the stdlib `csv` module (pandas' C and Python parsers both reject the title row's variable column count).

#### 5.1.3 TfNSW Traffic Volume Counts — actual resource layout (verified May 2026)

The `nsw-roads-traffic-volume-counts-api` package contains:

- **Road Traffic Counts Station Reference (API Generated CSV)** — `datastore_active=true`. Fetch via paginated `datastore_search`. ~1,800 stations with WGS84 lat/lon, road metadata, `quality_rating` (1-5), `permanent_station` flag.
- **Road Traffic Counts Hourly Permanent (API Generated CSVs)** — `format=ZIP`, *not* a datastore. The single ZIP download contains one or more CSVs with daily-row format (`date`, `daily_total`, `hour_00`..`hour_23`).
- Plus a yearly summary, a small hourly sample, an API description, and a PDF doc — all ignored by the fetcher.

`fetch.traffic` handles both shapes: datastore pagination for stations, ZIP-extract for hourly. Date-column timestamps in the ZIP are tz-aware (UTC) and must be normalised to naive before range filtering.

### 5.2 Tier 2 — get if cheap

| Source | URL / API | Notes |
|---|---|---|
| **AIP Terminal Gate Prices** | https://www.aip.com.au/historical-ulp-and-diesel-tgp-data | The "weekly" XLSX is misnamed — it ships **the full daily TGP back to 2004-01-01** for all 7 capital cities + national avg. We scrape the index page for the latest dated `AIP_TGP_Data_<DD-MMM-YYYY>.xlsx` link, parse the Petrol + Diesel sheets, lift Sydney columns. Forward-only/Wayback backfill from the original spec hint isn't needed — no data gap. |
| **RBA cash rate** | RBA F1.1 historical CSV (`csv/f1.1-data.csv`, series ID `FIRMMCRT`) | Monthly average; forward-fill to daily in the feature builder. |
| **ASX 200** | `yfinance` ticker `^AXJO` | daily close |
| **~~ANZ-Roy Morgan Consumer Confidence~~** → **RBA Inflation Expectations** | RBA G3 (`csv/g3-data.csv`, series ID `GCONEXP`) | Roy Morgan publishes only HTML tables (no API/CSV/XLS) and gates the historical series behind a commercial offering at `store.roymorgan.com`. Substituting RBA G3 *Consumer Inflation Expectations* (Melbourne Institute survey, quarterly back to 1985) — same signal-direction (consumer macro mood) with a clean, free, machine-readable feed. Forward-fill to daily in the feature builder. |
| **Singapore Mogas 95** | EIA International Petroleum Weekly | weekly; only add if Brent residuals indicate Singapore-shaped error |

### 5.3 Tier 3 — explicitly skipped in v1
- ABS Monthly Household Spending Indicator (monthly granularity, weak daily signal)
- NAB Business Survey (monthly, weak signal)
- CommBank HSI (proprietary)
- BOM operational forecast archives (Open-Meteo wraps ECMWF cleanly enough)

### 5.4 SA2 demographic features (via `abs-census-augmentor`)

The following 10 SA2-level variables form the "augmentation block." All come from the 2021 ABS Census GCP DataPack:

| Augmentor key | Variable | Rationale |
|---|---|---|
| `median_age` | `G02.Median_age_persons` | Age structure → driving / commuting patterns |
| `median_household_income_weekly` | `G02.Median_tot_hhd_inc_weekly` | Price sensitivity proxy |
| `total_population` | `G01.Tot_P_P` | Catchment size |
| `pct_drive_to_work` | derived from G46 | Direct fuel-demand proxy |
| `motor_vehicles_per_dwelling` | `G31` family | Vehicle ownership rate |
| `pct_renters` | derived from G33 | Tenure / wealth proxy |
| `pct_employed_full_time` | derived from G43 | Employment intensity |
| `pct_aged_65_plus` | derived from G04 | Age-pension recipient density proxy |
| `seifa_irsd_score` | external SEIFA dataset, joined on SA2 code | Disadvantage index, key for Centrelink-day interaction |
| `pct_one_parent_family` | derived from G25 | Welfare-recipient density proxy |

If SEIFA isn't supported by `abs-census-augmentor` directly, fetch the SA2 SEIFA table separately from ABS and merge in `src/build/enrich_census.py` after the augmentor pass.

## 6. Data Schemas

### 6.1 `data/interim/stations.parquet`

One row per unique service station ever observed in FuelCheck.

| Column | Type | Description |
|---|---|---|
| `station_id` | string | Stable hash of `(name, address, suburb, postcode)` |
| `name` | string | ServiceStationName (latest) |
| `address` | string | Address (latest) |
| `suburb` | string | |
| `postcode` | string | |
| `brand_raw` | string | Original `Brand` string from FuelCheck — preserved verbatim because franchisee-vs-corporate distinctions (e.g. `EG Ampol` vs `Ampol Foodary`) carry pricing signal. See §7.5. |
| `brand_canonical` | string | Standardised brand after `data/static/brand_aliases.csv` mapping (see §7.5). |
| `brand_is_major` | bool | True for the five "major" brand families: Ampol/Caltex, BP, Shell, 7-Eleven, Coles Express + Reddy Express. Looked up by `brand_raw` in the alias CSV; identity-mapped raws default to False. |
| `lat` | float64 | From G-NAF (preferred) or Nominatim (fallback) |
| `lon` | float64 | |
| `geocoder` | string | `'gnaf'` or `'nominatim'` |
| `mb_code` | string | Mesh Block code from G-NAF (when available); enables the augmentor's MB→SA2 fast-path. Null for Nominatim hits. |
| `sa2_code` | string | 2021 ASGS SA2 code from spatial join (added in Phase 3). |
| `sa2_name` | string | (added in Phase 3) |
| `first_seen` | date | First date in FuelCheck data |
| `last_seen` | date | Last date in FuelCheck data |

### 6.2 `data/interim/fuel_daily.parquet`

| Column | Type | Description |
|---|---|---|
| `station_id` | string | FK to stations |
| `fuel_code` | string | `'U91'`, `'DL'`, etc. |
| `date` | date | |
| `price_mean` | float64 | Mean of intraday price observations (cents/L) |
| `price_min` | float64 | |
| `price_max` | float64 | |
| `n_obs` | int | Number of price submissions that day |

Days with zero observations at a station are *not* present (i.e., the panel is unbalanced; rows are inserted only when a price was submitted). The feature builder forward-fills within station up to `max_forward_fill_days` (default 7) before computing lags.

`fuel_daily.parquet` retains both U91 and Diesel rows. Only the U91 rows feed the target (§7.8); the Diesel rows are kept so feature-engineering can construct cross-fuel signals at the same station (e.g. same-day Diesel price as a feature for U91, or U91-minus-Diesel spread). Cross-fuel feature columns live in the lag block — see §7.1.

### 6.3 `data/processed/features.parquet`

The training-ready matrix. Grain: `(station_id, fuel_code, date)`. Schema documented exhaustively in §7.

## 7. Feature Engineering Catalogue

All features are computed in `src/build/make_features.py`, organised into named blocks. Each block is a pure function `add_<block>_features(df, **kwargs) -> df` so blocks can be ablated individually for experimentation. Feature names use `snake_case` and a consistent prefix per block.

### 7.1 Lag block (`lag_*`)

Per `(station_id, fuel_code)` — for U91 rows only (the target rows):

```
lag_price_1, lag_price_2, lag_price_3, lag_price_7, lag_price_14, lag_price_28
roll_price_mean_7, roll_price_mean_14, roll_price_mean_28
roll_price_std_7, roll_price_std_14
days_since_last_price_change
price_minus_28d_min                 # captures cycle phase implicitly
price_minus_28d_max
```

Cross-fuel features (Diesel data joined onto U91 rows by `(station_id, date)`):

```
xfuel_dl_price_lag_0                 # same-day Diesel price at this station
xfuel_dl_price_lag_1
xfuel_u91_minus_dl_lag_1             # spread, often more stable than levels
xfuel_dl_roll_mean_7
```

If the station has no Diesel observation on a given day, the cross-fuel
columns forward-fill up to `max_forward_fill_days`, then null. LightGBM
handles nulls natively.

All rolling windows use `min_periods=window` to avoid early-life leakage.

### 7.2 Upstream block (`upstream_*`)

```
upstream_brent_lag_0, upstream_brent_lag_1, upstream_brent_lag_3, upstream_brent_lag_7, upstream_brent_lag_14
upstream_audusd_lag_0, upstream_audusd_lag_1, upstream_audusd_lag_3, upstream_audusd_lag_7
upstream_brent_aud_lag_0, upstream_brent_aud_lag_7, upstream_brent_aud_lag_14   # = brent / audusd
upstream_brent_change_7d, upstream_brent_change_14d
upstream_audusd_change_7d
```

If AIP TGP data is available for the relevant date, also:

```
upstream_tgp_sydney_lag_0, upstream_tgp_sydney_lag_3, upstream_tgp_sydney_lag_7
upstream_tgp_minus_brent_aud_lag_7   # margin proxy
```

Otherwise `upstream_tgp_*` columns are present and entirely null in the feature matrix; LightGBM handles nulls natively.

### 7.3 Calendar block (`cal_*`)

```
cal_day_of_week                       # 0-6
cal_day_of_month                      # 1-31
cal_month                             # 1-12
cal_week_of_year                      # 1-53
cal_year                              # int
cal_day_of_fortnight                  # 0-13, anchored at 2016-07-04 (a Monday)
cal_is_public_holiday                 # bool, NSW
cal_days_to_next_public_holiday       # int
cal_days_since_last_public_holiday    # int
cal_is_school_holiday_nsw             # bool
cal_is_first_business_day_after_break # bool, captures post-weekend/holiday Centrelink catch-up
```

The petrol cycle is *not* explicitly encoded — it should emerge from the lag block + day-of-week.

### 7.4 Demand context block (`ctx_*`)

Traffic features come from the **top-N nearest TfNSW counters** to the station — not just the single nearest. This captures the local demand environment: a station near a freight corridor + a school-bus route + a residential street has a different demand profile than a station near three suburban arterials of similar volume.

```
ctx_traffic_top1_distance_km          # haversine distance to closest counter
ctx_traffic_top2_distance_km
ctx_traffic_top3_distance_km
ctx_traffic_top1_lag_1                # daily count from closest counter
ctx_traffic_top1_lag_7
ctx_traffic_top2_lag_1
ctx_traffic_top2_lag_7
ctx_traffic_top3_lag_1
ctx_traffic_top3_lag_7
ctx_traffic_5km_radius_count          # number of counters within 5 km
```

`spatial.nearest` (Phase 2) builds a `(station_id, counter_rank, counter_id, distance_km)` table for ranks 1..N (default N=3). `build.make_features` joins counters' daily totals on `(counter_id, date)`.

If the *closest* counter is > 50 km away, all `ctx_traffic_top*` columns are null for that station.

```
ctx_inflation_expectations_lag_7      # RBA G3 Consumer (GCONEXP), forward-filled — see §5.2
ctx_asx200_lag_1                      # close
ctx_cash_rate                         # current value, forward-filled (slow-moving)
```

### 7.5 Static station block (`stn_*`)

Computed once per station, broadcast across the time index. Brand is exposed at multiple levels of granularity so the model can learn franchisee-vs-corporate pricing differences (which a single canonical column would erase):

```
stn_brand_raw                         # categorical, original FuelCheck Brand string (high cardinality)
stn_brand_canonical                   # categorical, post-alias (e.g. "Ampol")
stn_brand_is_major                    # bool: Coles Express, Reddy Express, 7-Eleven, BP, Caltex/Ampol, Shell
stn_is_franchisee                     # bool, see §13 Q3 — derived from brand_raw via a static rules file
stn_competitors_within_2km            # int, count of distinct station_ids within 2 km
stn_competitors_within_5km            # int
stn_distance_to_sydney_terminal_km    # haversine to Botany terminal
stn_is_metro                          # bool, derived from SA2 urbanisation classification
```

Brand standardisation lives in `data/static/brand_aliases.csv` — a manually maintained mapping from raw `Brand` strings to canonical names + an `is_major` flag. The CSV must be kept up to date when new brands appear; `clean.fuelcheck` logs a WARNING for any unmapped brand seen in the data.

`stn_is_franchisee` is derived per `brand_raw` from a separate static rules file (`data/static/brand_franchisee_rules.csv`) that lists known franchisee patterns (e.g. `EG Ampol`, `EBM Ampol` are EG Group / EBM franchisees of Ampol; `Ampol Foodary` is corporate). The rules file is research-derived and starts small — see §13 Q3.

### 7.6 Weather block (`wx_*`)

Daily aggregates from Open-Meteo, joined on `(station_lat, station_lon, date)`. Cached per station in `data/raw/weather/<station_id>.parquet`:

```
wx_temp_max_c
wx_temp_min_c
wx_precipitation_mm
wx_wind_speed_max_kmh
wx_weather_code                        # categorical, WMO code
```

Note on leakage: v1 uses Historical Weather (ERA5 reanalysis) across the full span. The README must call this out as a methodological compromise. v2 should switch to Previous Runs API at lead-time = 1 day for the 2024+ portion of the data.

### 7.7 Demographic block (`sa2_*`) — the augmentor block

```
# Census 2021 GCP — direct fields
sa2_median_age                                    # G02.Median_age_persons
sa2_median_household_income_weekly                # G02.Median_tot_hhd_inc_weekly
sa2_total_population                              # G01.Tot_P_P

# Census 2021 PRESET ratios (six derived percentages, augmentor v1.4.2+)
sa2_pct_drive_to_work                             # PRESET.pct_drive_to_work
sa2_motor_vehicles_per_dwelling                   # PRESET.motor_vehicles_per_dwelling
sa2_pct_renters                                   # PRESET.pct_renters
sa2_pct_employed_full_time                        # PRESET.pct_employed_full_time
sa2_pct_aged_65_plus                              # PRESET.pct_aged_65_plus
sa2_pct_one_parent_family                         # PRESET.pct_one_parent_family

# SEIFA 2021 — four indexes (one-shot per Census)
sa2_seifa_irsd_score                              # SEIFA.irsd_score   — disadvantage continuum
sa2_seifa_irsad_score                             # SEIFA.irsad_score  — advantage + disadvantage two-direction
sa2_seifa_ier_score                               # SEIFA.ier_score    — economic resources (income, assets, dwelling)
sa2_seifa_ieo_score                               # SEIFA.ieo_score    — education + occupation

# ABS Estimated Resident Population (latest annual release, currently 2024).
# v1.5 fetcher only emits `population_total` — the dataset spec markdown's
# promised age bands / density / median age aren't wired up. See §7.7.3.
sa2_erp_population_total                          # ERP.population_total — current vs Census 2021 snapshot

# ABS Personal Income in Australia (latest annual release, currently 2022-23).
# LEED-derived from ATO data; complements Census household income with a
# different bias profile (no top-coding, but excludes non-filers).
sa2_pia_median_total_income                       # ABS_PIA.median_total_income
sa2_pia_mean_total_income                         # ABS_PIA.mean_total_income       — mean−median spread captures distribution skew
sa2_pia_income_earners_count                      # ABS_PIA.income_earners_count    — proxy for employment level
sa2_pia_median_age_of_earners                     # ABS_PIA.median_age_of_earners

# DSS Payment Demographic Data — quarterly welfare-recipient counts.
# Pinned to latest available release (currently 2025-Q3) for v1; temporal
# per-row resolution is a deferred follow-up — see §7.7.2.
# Selected from the ~21 columns DSS publishes — the omitted ones (ABSTUDY,
# special benefit, austudy, low-income card, etc.) have very small recipient
# pops that suppress to null in most NSW SA2s.
sa2_dss_age_pension_recipients                              # DSS.age_pension_recipients
sa2_dss_jobseeker_payment_recipients                        # DSS.jobseeker_payment_recipients
sa2_dss_disability_support_pension_recipients               # DSS.disability_support_pension_recipients
sa2_dss_parenting_payment_single_recipients                 # DSS.parenting_payment_single_recipients
sa2_dss_parenting_payment_partnered_recipients              # DSS.parenting_payment_partnered_recipients
sa2_dss_carer_payment_recipients                            # DSS.carer_payment_recipients
sa2_dss_carer_allowance_recipients                          # DSS.carer_allowance_recipients
sa2_dss_youth_allowance_other_recipients                    # DSS.youth_allowance_other_recipients
sa2_dss_youth_allowance_student_and_apprentice_recipients   # DSS.youth_allowance_student_and_apprentice_recipients
sa2_dss_commonwealth_rent_assistance_recipients             # DSS.commonwealth_rent_assistance_recipients
sa2_dss_commonwealth_seniors_health_card_recipients         # DSS.commonwealth_seniors_health_card_recipients
sa2_dss_family_tax_benefit_a_recipients                     # DSS.family_tax_benefit_a_recipients — kid-count proxy
sa2_dss_family_tax_benefit_b_recipients                     # DSS.family_tax_benefit_b_recipients — single-parent / single-income family proxy
```

This block is the *only* difference between Model A and Model B.

**Coverage / acceptance.** The v1 acceptance gate (≥ 95% non-null) applies to the GCP-derived columns (the 9 Census + 6 PRESET ones) and SEIFA — those are dense ABS publications. ERP / ABS_PIA / DSS columns may be null for SA2s that fall outside the publication's coverage (e.g. "Migratory / offshore / shipping" pseudo-SA2s, or SA2s under DSS small-cell suppression where < ~20 recipients of a given payment type are reported as null). Coverage is logged per-column on every `make enrich` run; columns persistently below 95% on substantive NSW SA2s are investigated, not silenced.

#### 7.7.1 Derived variables — RESOLVED

Originally Phase 3 v1 stubbed the 6 derived percentages with nulls because (a) the right denominator per ratio is non-obvious, (b) the 200-column GCP tables make field-code archaeology a non-trivial spike, and (c) augmentor PRESETs were not yet exposed as first-class pipeline variables. All three blockers have since cleared:

- [abs-census-augmentor#11](https://github.com/cauldnz/abs-census-augmentor/issues/11) → v1.3 shipped curated PRESET specs.
- [abs-census-augmentor#19](https://github.com/cauldnz/abs-census-augmentor/issues/19) → v1.4.1 ships the spec markdown in the wheel so registries populate on a fresh install.
- [abs-census-augmentor#18](https://github.com/cauldnz/abs-census-augmentor/pull/18) → v1.4.0 makes `PRESET.<id>` a first-class variable namespace alongside `G\d+.<col>` / `SEIFA.*` / `ERP.*` / `DSS.*` / `ABS_PIA.*`.
- [abs-census-augmentor#23](https://github.com/cauldnz/abs-census-augmentor/issues/23) → v1.4.2 rewrites the PRESETs against the **real** GCP DataPack (the v1.3 PRESETs referenced columns that didn't actually exist; tests passed because synthetic fixtures encoded the same broken names).

`build.enrich_census` now passes all 6 PRESETs as variables to `Pipeline.augment(...)`. All Census-derived columns are populated. Acceptance threshold (≥ 95% non-null on the GCP / SEIFA columns) applies as spec'd. No null-stub framework remains.

#### 7.7.2 Temporal augmentation — landed (PR B)

The DSS Payment Demographic Data feed publishes one snapshot per calendar quarter going back to 2022-Q4. The augmentor's Temporal mode (`Pipeline.augment(df, date_column=...)`) resolves each row to the closest snapshot independently, giving per-(station, date) values rather than a single static snapshot. This is the natural fit for the augmentor-narrative story — fortnightly Centrelink-day pricing cycles depend on *current* welfare populations, not a 2025-Q3 snapshot held constant across the panel. Per-row SEIFA similarly differentiates 2016 vs 2021 Census release values across the panel's 2016-09 → 2026-04 span.

**v1.5 status (historical):** Temporal mode existed but was single-edition (ASGS Edition 3 only); pre-2023-Q2 DSS releases on ASGS Edition 2 would fail or null out. This was the original blocker.

**v2.0 status (2026-05-27 release):** Cross-edition orchestration landed (Phases F.1–F.4). Our PR A spike (see [`docs/research/2026-05_abs_census_augmentor_v2.0_review.md`](docs/research/2026-05_abs_census_augmentor_v2.0_review.md)) found two implementation gaps and filed them upstream as cauldnz/abs-census-augmentor#91 (GCP cross-edition NaN) and #92 (ERP single-publication).

**v2.0 post-fix status (2026-05-29, current commit `65fd3fa6`):**

- **#92 fully resolved** via PR #95 — `ErpDataSource` now serves any historical year ≤ latest via column projection at `load()` time. `ERP.population_total` works for any panel row from 2017+.
- **#91 Stage 1 resolved** via PR #94 — silent-NaN replaced with a loud `ValueError`. Stage 2 (the proper per-release `DataPacksDataSource` routing) remains on upstream backlog.

**PR B architecture (landed in this PR):** split the augmentor surface into two passes, one cross-sectional and one temporal, with each variable belonging to exactly one pass.

- **Cross-sectional pass** (existing `build.enrich_census` against `stations.parquet`) — GCP direct + GCP-internal PRESETs + ERP age/sex + ABS_PIA + cross-dataset PRESETs + **DSS welfare** (latest quarter). Four reasons a variable stays cross-sectional:
  1. GCP-routed: upstream #91 Stage 2 pending — temporal mode raises a loud `ValueError`.
  2. ERP age/sex: ABS 3235.0 cube ships these only for the latest publication year (documented in upstream #92 resolution); historical rows would return null.
  3. Cross-dataset PRESETs depend on ERP age/sex denominators → inherit (2).
  4. **DSS welfare**: the augmentor's DSS XLSX parser fails on the 2022-Q4 release (the earliest available) — `RuntimeError: No SA2 data rows`. Filed upstream as cauldnz/abs-census-augmentor#99. Until that lands, DSS stays cross-sectional (latest quarter only). Moving DSS to the temporal pass when #99 closes is a one-line config change.
- **Temporal pass** (new `build.enrich_panel_temporal` against the panel, deduped to unique (station_id, date)) — SEIFA + ERP `population_total`. Output joins back to `features.parquet` on (station_id, date) at make_features time.

The split is exhaustive — `config.AUGMENTOR_VARIABLES_TEMPORAL` and `config.AUGMENTOR_VARIABLES_CROSS_SECTIONAL` are disjoint, guarded by a unit test. Each variable's `sa2_*` column name is the same regardless of source; the model code (`feature_blocks.SA2_COLUMNS`) doesn't care which pass populated which column.

**Coverage on train fold:** Train rows (≤ 2022-12-31) get genuine per-row SEIFA variation across the 2016/2021 release transition (~50/50 split). ERP `population_total` projects back to 2017 via column projection on the 2024 publication cube (ABS internal concordance, see upstream #92 docs); pre-2017 rows clamp to release 2017 via `temporal.out_of_range: nearest`. DSS adds no per-quarter variation in this PR (pending #99); it remains frozen at the latest published quarter via the cross-sectional pass — same behaviour as PR A.

**Empirical outcome (PR B headline retrain):** Per-row temporal SEIFA + ERP `population_total` **regressed** test_normal Δ MAE from −0.353 → −0.239 and test_crisis Δ MAE from −0.398 → −0.321 vs. the PR A cross-sectional baseline. The architecture is correct and the columns flow through correctly (verified by the temporal-block merge log + 99.2-100% coverage on the new panel parquet), but per-row variation in 2016-vs-2021 SEIFA + 2017-2024 ERP appears to introduce noise the model doesn't pay back — possibly because the panel-skew toward post-2020 dates makes the older releases low-value, or because the model was already extracting whatever temporal-demographics signal exists via `date`/year features. The architecture remains in place as a no-regret platform for future column moves (especially DSS once upstream #99 lands), but the **temporal-demographics hypothesis from this section is not empirically supported** on the v2.0+weather-fix problem as configured. See [`results/README.md`](results/README.md) iteration table for the full row.

#### 7.7.3 Augmentor schema vs spec drift — narrowed surface for ERP / ABS_PIA

PR #45 was drafted against the dataset spec markdown files in `cauldnz/abs-census-augmentor` (`datasets/erp_by_sa2.md`, `datasets/abs_personal_income.md`), which document a richer schema than the v1.5 fetchers actually emit:

- **ERP**: spec promises `population_density_per_km2`, `population_0_14`, `population_15_64`, `population_65_plus`, `median_age`. Fetcher emits only `population_total` + 25 historical-year columns (`population_history_YYYY`).
- **ABS_PIA**: spec promises `gini_coefficient`, plus `median_employee_income`, `median_investment_income`, `median_super_income`, `median_own_business_income`. Fetcher emits 5 summary stats only (`income_earners_count`, `median_age_of_earners`, `sum_total_income`, `median_total_income`, `mean_total_income`).
- **DSS**: spec lists 9 named payment columns. Fetcher emits 21 (everything DSS publishes per quarter), and one of the spec'd names (`youth_allowance_student_recipients`) is actually `youth_allowance_student_and_apprentice_recipients` post-snake-casing.

PR #46 fixed our `AUGMENTOR_VARIABLES` dict to match what's actually emitted: ERP shrank from 5 to 1 column, ABS_PIA grew from 1 to 4 columns (the gini was the one promised-but-missing entry), DSS grew from 9 to 13 (using real names + 4 bonus payment categories — FTB-A, FTB-B, carer allowance, seniors health card). Net SA2 surface: 28 → 29 columns.

**Augmentor v2.0 update (§7.7.5):** PR #82 upstream closed the ERP age/sex gap — `population_male/female`, `population_0_14/15_64/65_plus`, and `median_age` are now emitted by the v2.0 fetcher. We picked up `population_65_plus` + `median_age` in PR A; the gender split is fetched-but-not-modeled for v1. The ABS_PIA `gini_coefficient` + income-by-source medians remain on the spec-only side.

This is the second occurrence of the same root-cause pattern — [augmentor #23](https://github.com/cauldnz/abs-census-augmentor/issues/23) was the first, where v1.3 PRESETs referenced GCP columns that didn't exist and tests passed against synthetic fixtures encoding the same broken names. Upstream issue [#65](https://github.com/cauldnz/abs-census-augmentor/issues/65) was filed against the augmentor to (a) trim the dataset spec markdowns to reality or (b) implement the spec'd columns, *and* to add a `test_spec_matches_fetcher_columns` test rung that locks the door against this category of bug recurring. Local copy of the issue body is in `tools/upstream_issue_dataset_spec_drift.md`.

Lesson on our side: when integrating a 3rd-party data library, don't trust documentation as schema. Probe-fetch one record of each registered dataset and call `.columns` before writing any variable list. Construction-time validation (`Pipeline.create(variables=...)` succeeding) only proves variable refs *parse*, not that they'll *resolve to columns the fetcher returns*. A 5-second probe would have made this a single PR instead of two.

#### 7.7.4 Block curation — 31 columns broadened, then trimmed to 15

PR #45/#46 broadened the SA2 block from 10 → 31 columns by adding the new DSS welfare, ERP, ABS_PIA, and broader-SEIFA features the augmentor v1.5 surface exposed. The first training run with the broadened block produced:

| Iteration | SA2 cols | Val MAE | best_iter | Test_normal Δ MAE | Test_crisis Δ MAE |
|---|--:|--:|--:|--:|--:|
| v1.0 (original 10) | 10 | 4.85 | 696 | **−0.059** | **−0.396** |
| v1.1 (broadened to 31) | 31 | 4.78 | 585 | −0.025 | −0.191 |

Classic overfitting signature: **better val, worse test**. The 31-col model fit val-fold (2023) patterns that didn't generalise into test_normal (2024-25) or test_crisis (2026).

Feature-importance analysis attributes the regression to noise: of the 21 added features, only 5 ranked at ≥ 0.02% gain (ranks 45-51 in Model B); the bottom 16 had gain ≤ 0.01% (effectively noise floor) but still consumed parameter budget that LightGBM could have spent on better features. The high-correlation additions in particular — e.g. `sa2_dss_youth_allowance_student_and_apprentice_recipients` ↔ `stn_competitors_within_5km` at +0.66 — were *partially re-encoding* urban density that the model already gets from the `stn_competitors_*` and `ctx_traffic_*` blocks.

The v1.2 curation keeps the original 10 + the 5 top-by-gain new features:

| Kept | Reason |
|------|--------|
| All 10 original (Census GCP + SEIFA IRSD) | Proven baseline across 3 iterations |
| `sa2_seifa_ieo_score` (rank 51) | Only Education + Occupation SEIFA we'd have; distinct from IRSD |
| `sa2_dss_parenting_payment_partnered_recipients` (rank 45) | Highest-impact new feature |
| `sa2_dss_carer_payment_recipients` (rank 48) | Care-giver demographic proxy |
| `sa2_dss_carer_allowance_recipients` (rank 50) | Broader complement to carer_payment |
| `sa2_dss_youth_allowance_student_and_apprentice_recipients` (rank 49) | Young-cohort proxy; distinct generational signal despite collinearity with competitor count |

`AUGMENTOR_VARIABLES` in `config.py` still requests all 31 columns so they remain available in `stations.parquet` for future ablation studies. The model just doesn't consume them.

**Implication for temporal-DSS (§7.7.2):** if *static* DSS recipient counts contribute ≤ 0.04% gain at the top end (and most at noise floor), the marginal value of per-quarter temporal resolution is questionable — the temporal hypothesis would need to live entirely in quarter-to-quarter variation that the static snapshot misses. Temporal-DSS is **further deprioritised** until a separate experiment demonstrates the static signal floor isn't the ceiling.

#### 7.7.5 Augmentor v2.0 upgrade — static-surface bump

PR A (this phase) bumps the pin to augmentor v2.0.0 and broadens `AUGMENTOR_VARIABLES` by 5 columns — all in **cross-sectional** mode (temporal mode remains deferred per §7.7.2):

- `sa2_erp_population_65_plus` (ERP age-cohort split, new in v2.0 PR #82)
- `sa2_erp_median_age` (ERP median age, new in v2.0 PR #82)
- `sa2_pct_age_pension_recipients` (cross-dataset PRESET, DSS ÷ ERP 65+; new in v2.0 PR #86)
- `sa2_pct_jobseeker_recipients` (cross-dataset PRESET, DSS ÷ ERP working-age; new in v2.0 PR #86)
- `sa2_welfare_density_index` (cross-dataset PRESET, 9 DSS payments ÷ ERP total; new in v2.0 PR #86)

These land in `stations.parquet` but **do not enter `SA2_COLUMNS` (the 15-col model block)** automatically — per the §7.7.4 curation pattern, the model consumes a curated subset and gain-based ranking decides what's worth keeping. A follow-up curation experiment (similar to v1.5's 31→15 trim) can re-rank with the new 5 candidates in the pool. The PR A headline experiment is "same 15-col block, augmentor v2.0 vs v1.5 — does the upstream version bump alone move the test fold?" — analogous to the v1.4.2→v1.5 swing documented in [`docs/research/2026-05_abs_census_augmentor_v1.5_review.md`](docs/research/2026-05_abs_census_augmentor_v1.5_review.md).

For the temporal-mode (PR B) work that follows once the upstream blockers in §7.7.2 close, the 3 cross-dataset PRESETs inherit the ERP single-publication limitation (they use ERP denominators); they freeze to the latest ERP release regardless of `date_column`. Not a behavioural difference for PR A but worth flagging for PR B planning.

**PR B follow-up (2026-05-30):** With upstream cauldnz/abs-census-augmentor#92 fully resolved and #91 Stage-1 fixed (loud error), the split documented in §7.7.2 landed. The 5 columns above remain cross-sectional per the routing decisions there; PR B added SEIFA + DSS + `ERP.population_total` to a new temporal pass (`build.enrich_panel_temporal`).

### 7.8 Target

Built from U91 rows only:

```
y_t1     # price_mean at t+1, shifted within (station_id, 'U91')
y_t1_t7  # mean(price_mean[t+1..t+7]), shifted within (station_id, 'U91')
```

Diesel rows in `fuel_daily.parquet` carry no target — they exist solely
as feature inputs for the U91 cross-fuel block (§7.1). Rows where the
target is null (end-of-series) are dropped before training.

## 8. Modeling Specification

### 8.1 Algorithm

LightGBM regressor (`lightgbm.LGBMRegressor`), tabular tree-based model. Sufficient for the data size and handles nulls + categoricals natively.

### 8.2 Hyperparameters (v1, fixed)

```python
LGBM_PARAMS = dict(
    objective="regression_l1",       # MAE-aligned loss
    metric="mae",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    n_estimators=2000,
    early_stopping_rounds=100,
    verbose=-1,
    random_state=42,
)
```

These are deliberately reasonable defaults. **Hyperparameter tuning is out of scope for v1** — the experiment compares Model A vs Model B at fixed hyperparameters.

### 8.3 Validation strategy

Time-based, no shuffling. Splits:

| Fold | Date range | Use |
|---|---|---|
| Train | 2016-09-01 → 2022-12-31 | Fit |
| Validation | 2023-01-01 → 2023-12-31 | Early stopping |
| Test (normal) | 2024-01-01 → 2025-12-31 | Headline metrics |
| Test (crisis) | 2026-01-01 → end of data | Reported separately as out-of-distribution |

No k-fold CV in v1 — the time-based holdout is the validation. Group-aware splitting is unnecessary because we never train on a station-day's future and predict its past; targets are strictly forward-shifted.

### 8.4 The A/B comparison

Two models, identical except for one feature block:

| | Feature blocks |
|---|---|
| **Model A** | lag, upstream, calendar, ctx, stn, wx |
| **Model B** | lag, upstream, calendar, ctx, stn, wx, **sa2** |

Both trained on the *same* training rows (rows where every **SA2** column is non-null — so the comparison isn't biased by Model B having fewer/easier rows). Other naturally-sparse columns (`xfuel_dl_*`, `upstream_tgp_*`, occasional Tier-2 macros) are in both models' feature sets and LightGBM handles their nulls natively; filtering on every Model B column would be over-strict and on real corpora has been observed to leave zero training rows because rare-coverage columns combine multiplicatively. The §8.4 "apples-to-apples" intent is that the SA2 join shouldn't bias the comparison — exactly what filtering on the SA2 block isolates.

### 8.5 Metrics

For each model, on the test fold(s), report:

- MAE (cents/L)
- RMSE (cents/L)
- MAPE (%)
- Median absolute error
- 90th-percentile absolute error

Also compute these segmented by:
- Metro / regional (`stn_is_metro`)
- Brand (top 8 brands + "Other")
- Fuel type (U91 / Diesel)
- SA2 SEIFA quintile

The headline result is **Model B's MAE / MAPE minus Model A's**, segmented as above. The augmentor's value is the size and direction of this delta.

## 9. Notebooks

All notebooks read from `data/processed/features.parquet`. None of them refit data or re-call APIs.

### 9.1 `notebooks/01_eda.ipynb`

Sections:
1. Dataset overview — station count over time, fuel-code coverage, observation density
2. Geographic distribution — map of stations coloured by SA2 SEIFA, brand mix by region
3. Price level and dispersion — by fuel, by brand, over time
4. The petrol cycle — autocorrelation by station, FFT on a sample station to demonstrate the ~3-week period
5. The 2026 crisis — visible regime change in Brent + retail prices
6. Centrelink-day check — average price residual (vs 28-day rolling mean) by `cal_day_of_fortnight`, segmented by SEIFA quintile. **This is the augmentor-story chart and must be in the notebook.**
7. Cross-correlations — Brent (lagged) vs retail at Sydney metro vs regional, to motivate lag features
8. Missingness map for SA2 features (% rows that lack each SA2 variable)

### 9.2 `notebooks/02_modeling.ipynb`

Sections:
1. Load features, define folds
2. Fit Model A (no SA2)
3. Fit Model B (with SA2)
4. Print headline metrics for both, side by side
5. Segmented metrics tables
6. Residual diagnostics — plot residuals over time, check for crisis-period blowup
7. Save models, write `results/comparison.md`

### 9.3 `notebooks/03_explainability.ipynb`

Sections:
1. SHAP summary plot for Model B (top 30 features)
2. SHAP dependence plots for top SA2 features
3. SHAP interaction plot for `cal_day_of_fortnight × sa2_seifa_irsd_score` — the demonstration of the augmentor's interaction value
4. Comparison of top-20 feature importances between Model A and Model B
5. Per-station case studies — pick 3 stations across the SEIFA spectrum, show predictions vs actuals + waterfall for one prediction

## 10. Repository Layout

```
fuel-prediction/
├── README.md                    # human-facing intro, quickstart
├── CLAUDE.md                    # conventions for AI-agent contributors
├── spec.md                      # this document
├── pyproject.toml               # uv-managed deps
├── Makefile                     # `make all`, `make fetch`, `make features`, ...
├── .gitignore
├── data/
│   ├── raw/                     # gitignored; cached fetches
│   │   ├── fuelcheck/
│   │   ├── traffic/
│   │   ├── weather/
│   │   ├── brent.parquet
│   │   ├── audusd.parquet
│   │   ├── cash_rate.parquet
│   │   ├── asx200.parquet
│   │   └── consumer_confidence.parquet
│   ├── interim/                 # gitignored; cleaned intermediates
│   │   ├── stations.parquet
│   │   ├── fuel_daily.parquet
│   │   └── ...
│   ├── processed/               # gitignored; the final matrix
│   │   └── features.parquet
│   └── static/                  # checked in
│       ├── brand_aliases.csv
│       ├── nsw_school_terms.csv
│       └── crisis_events.csv    # event annotations (informational, not in features)
├── src/
│   └── fuel_pred/
│       ├── __init__.py
│       ├── config.py            # paths, constants
│       ├── fetch/
│       │   ├── fuelcheck.py
│       │   ├── traffic.py
│       │   ├── brent.py
│       │   ├── audusd.py
│       │   ├── weather.py
│       │   ├── cash_rate.py
│       │   ├── asx200.py
│       │   └── consumer_confidence.py
│       ├── clean/
│       │   ├── fuelcheck.py     # dedupe, standardise, daily aggregate
│       │   └── traffic.py
│       ├── spatial/
│       │   ├── resolve_addrs.py # G-NAF → Nominatim fallback
│       │   └── nearest.py       # nearest-traffic-counter, terminal distances
│       ├── build/
│       │   ├── panel_grid.py    # build (station, fuel, date) grid
│       │   ├── enrich_census.py # call abs-census-augmentor
│       │   └── make_features.py # all feature blocks
│       ├── train/
│       │   └── train_models.py  # fit Models A and B
│       └── evaluate/
│           ├── metrics.py
│           └── compare.py       # write results/comparison.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_explainability.ipynb
├── tests/
│   ├── test_fetch_*.py          # mocked
│   ├── test_clean_*.py
│   ├── test_features.py
│   └── test_metrics.py
├── models/
│   ├── model_a.pkl
│   └── model_b.pkl
└── results/
    ├── comparison.md
    └── shap/
        ├── summary_b.png
        ├── dependence_<feature>.png
        └── interaction_dof_seifa.png
```

## 11. Tech Stack & Conventions

- **Python**: 3.11+
- **Package manager**: `uv` (matches the `abs-census-augmentor` pattern)
- **Data**: `pandas` (compatible with `abs-census-augmentor`'s API), `pyarrow` for Parquet IO
- **ML**: `lightgbm`, `scikit-learn` (utilities only), `shap`
- **Geospatial**: `shapely`, `geopandas` (only as needed for spatial joins; `abs-census-augmentor` handles SA2)
- **Network**: `requests`, `tenacity` (retry), `yfinance`
- **Notebooks**: `jupyterlab`
- **Tests**: `pytest`, `pytest-mock`, `responses` (HTTP mocking)
- **Lint / format**: `ruff` (check + format), `mypy` (strict on `src/`)
- **Plotting**: `matplotlib` (notebooks), `plotly` (optional for interactive maps)
- **Task runner**: GNU `make` (Makefile)

Conventions:
- Every `src/` module has a `__main__` block: `python -m fuel_pred.fetch.brent --start 2016-09-01 --end 2026-04-30 --out data/raw/brent.parquet`
- All paths come from `fuel_pred.config` — no hard-coded paths in modules.
- All public functions have type hints, validated by mypy strict mode.
- Tests are hermetic. Real-network integration tests live in `tools/` (opt-in), mirroring the `abs-census-augmentor` pattern.
- Logging via `logging` stdlib, not print. Each fetcher logs source URL, row counts, and cache hits at INFO.

### Devcontainer & container engine

- The devcontainer (`.devcontainer/`) targets the Microsoft Python 3.11 base image. Both **Docker Desktop** and **Podman Desktop** are tested and work without changes — the Dev Containers extension auto-detects whichever engine is running.
- **No Docker-in-Docker / `docker-outside-of-docker`.** The `ghcr.io/devcontainers/features/docker-outside-of-docker` feature is intentionally **not** included. Audit (May 2026): zero references to the Docker socket, `DOCKER_HOST`, `docker-py`, `testcontainers`, or any subprocess invocation of `docker` exist anywhere in `src/`, `tests/`, `tools/`, lifecycle scripts, or the resolved dependency tree (including `abs-census-augmentor`). If a future component genuinely needs Docker access, prefer adding it as an explicit `mounts:` + `containerEnv: DOCKER_HOST=…` pair (so Podman users only need to point the mount at `/run/podman/podman.sock` rather than re-add a feature that assumes a socket path).
- Verify the audit any time with: `git grep -niE 'docker(-py|_host)|testcontainers|/var/run/docker' -- ':^.devcontainer' ':^.claude'` from the repo root — should return zero matches.

## 12. Implementation Phases

Each phase produces a runnable artefact and a testable outcome. Designed for sequential overnight Claude Code sessions.

### Phase 0 — Skeleton (≤ 1 session)
- `pyproject.toml`, `uv` lockfile, repo layout, empty modules with TODO docstrings
- `Makefile` with target stubs that print "not implemented"
- CI config (GitHub Actions: ruff, mypy, pytest)
- This `spec.md` checked in

### Phase 1 — Tier 1 fetchers ✅ (PR #1, claude/upbeat-wu-8bc435)
- `fetch.fuelcheck` — download monthly archives from data.nsw.gov.au, write **one Parquet per month** as `data/raw/fuelcheck/<YYYY-MM>.parquet` (concatenation deferred to the cleaner; preserves drift-affected raw schema)
- `fetch.brent`, `fetch.audusd`, `fetch.traffic` — implemented per §5.1.1-5.1.3
- `fetch.weather` — split out from Phase 1 because it needs station lat/lons from `clean.fuelcheck`. Lands as a Phase-2-rider after Phase 2's roster is in place; uses Open-Meteo's archive (ERA5) per §7.6 with the documented leakage caveat.
- Hermetic tests for each fetcher (responses-mocked)
- Acceptance: `make fetch-tier1` populates `data/raw/` end-to-end on a fresh machine. `make fetch-weather` runs separately after `make clean-data`.

### Phase 2 — Cleaning + station roster (1 session) ✅
- `clean.fuelcheck` — read all monthly Parquets, normalise brand strings via `data/static/brand_aliases.csv`, hash `(name, address, suburb, postcode)` into `station_id`, aggregate per `(station_id, fuel_code, date)` for both U91 and Diesel
- `spatial.resolve_addrs` — uses `abs-census-augmentor` (now `census-augment` import) with `GnafConfig(mode='remote')` + Nominatim fallback. One geocode per `station_id` (not per unique address — see §13 resolved). Idempotent: rows that already have `(lat, lon, geocoder)` populated are skipped unless `--force`. Nominatim responses cached on disk under `data/raw/geocode_cache/` to keep usage polite (Nominatim usage policy: 1 req/sec, no bulk).
- `clean.traffic` — daily aggregation from hourly. Drop rows from non-permanent stations and `quality_rating < 3` (TfNSW's data-quality scale runs 1-5; ratings 1-2 indicate sparse coverage that produces unreliable daily totals — see the dataset's Data Quality Statement)
- Acceptance: `data/interim/stations.parquet` and `data/interim/fuel_daily.parquet` exist with the schemas in §6

### Phase 3 — Census enrichment (1 session) ✅
- `build.enrich_census` — wraps `census_augment.Pipeline.augment(...)` (uses pre-resolved lat/lon from Phase 2). All §7.7 `sa2_*` columns are populated through a single unified augmentor call:
  - 3 direct GCP fields (`G01.Tot_P_P`, `G02.Median_age_persons`, `G02.Median_tot_hhd_inc_weekly`).
  - 6 PRESET derivations (`PRESET.pct_drive_to_work`, `motor_vehicles_per_dwelling`, `pct_renters`, `pct_employed_full_time`, `pct_aged_65_plus`, `pct_one_parent_family`) via augmentor v1.4.2+ — see §7.7.1 for the resolution history.
  - 4 SEIFA scores (`SEIFA.irsd_score`, `irsad_score`, `ier_score`, `ieo_score`).
  - 1 ERP variable (`ERP.population_total`) — pinned to the latest annual release. (PR #46 narrowed this from 5 to 1 — spec drift, see §7.7.3.)
  - 4 ABS_PIA variables (`ABS_PIA.median_total_income`, `mean_total_income`, `income_earners_count`, `median_age_of_earners`) — pinned to the latest financial-year release. (PR #46 grew this from 1 to 4 — spec drift, see §7.7.3.)
  - 13 DSS welfare-payment recipient counts (age pension, jobseeker, DSP, parenting × 2, carer × 2, youth allowance × 2, CRA, seniors health card, FTB-A, FTB-B) — pinned to the latest quarterly release; per-row temporal resolution deferred to a follow-up (§7.7.2).
- Acceptance: `data/interim/stations.parquet` has all `sa2_*` columns + `sa2_code` / `sa2_name` populated for ≥ 95% of stations on the GCP / SEIFA columns. ERP / ABS_PIA / DSS columns may legitimately be null on the small handful of NSW SA2s outside their publication coverage (or under DSS small-cell suppression); coverage is logged per column.

### Phase 4 — Feature build (1 session) ✅
- `build.panel_grid` — assemble the (station, fuel, date) grid
- `build.make_features` — implement all blocks from §7
- Forward-fill, lag, rolling, calendar features, weather join, traffic join
- Acceptance: `data/processed/features.parquet` exists, schema matches §7, no rows where every feature is null

### Phase 5 — Tier 2 fetchers + features (1 session) ✅
- `fetch.cash_rate`, `fetch.asx200`, `fetch.inflation_expectations` (replaces `consumer_confidence` per §5.2 — Roy Morgan unavailable as a clean feed), `fetch.aip_tgp`
- AIP TGP scraper (start collecting forward; no historical backfill required)
- Add corresponding feature columns
- Acceptance: feature matrix has the new `ctx_*` columns

### Phase 6 — Modeling (1 session) ✅
- `train.train_models` — fit Models A and B with the spec §8.2 hyperparameters. Implemented via:
  - `train.feature_blocks` — explicit per-block column lists (§7) + `MODEL_A_BLOCKS` / `MODEL_B_BLOCKS` per §8.4 + categorical / exclude lists.
  - `train.folds` — time-based splitter producing the four §8.3 folds (`train`, `val`, `test_normal`, `test_crisis`).
  - `train._fit` — inner `fit_lgbm()` wrapping `lightgbm.LGBMRegressor` with the early-stopping callback + `lgb.log_evaluation` periodic output (PR #37, #38).
  - `train.train_models.train()` — orchestrator: load → filter to U91 + non-null target → split → identical-rows guard (§8.4: rows where every SA2 column is non-null) → defensive object→numeric coercion (PR #36) → fit A → fit B → persist `model_a.pkl` / `model_b.pkl` / `feature_lists.json` / `predictions_test_normal.parquet` / `predictions_test_crisis.parquet`. CLI knobs: `--n-estimators`, `--log-period` (PR #37, #39).
- `evaluate.metrics` — implemented in PR #28 (the five §8.5 metrics + `all_metrics()` convenience).
- `evaluate.compare` — implemented; consumes the prediction parquets and writes `results/comparison.md` with overall + four segmented tables (metro/regional, brand top-8 + Other, fuel, SEIFA quintile) per spec §8.5/§9.2.
- Acceptance: both models saved, prediction parquets persisted, `results/comparison.md` generated.

### Phase 7 — Notebooks (1-2 sessions) ✅
- Implement `01_eda`, `02_modeling`, `03_explainability` per §9
- Acceptance: all three run top-to-bottom without errors against the saved feature matrix

### Phase 8 — Polish (1 session) ✅
- README with quickstart
- CLAUDE.md with contributor conventions
- Test coverage check (310 passed, 5 previously-skipped feature-engineering tests now implemented)
- One end-to-end run from a clean checkout to confirm reproducibility

## 13. Open Questions

To be resolved during implementation, not blocking spec sign-off:

1. **AIP TGP historical backfill** — is there any retrievable archive, or only forward scraping? If forward only, the `upstream_tgp_*` features will be heavily null in early years. Acceptable given tier-2 status.
2. **SEIFA join key** — does `abs-census-augmentor` expose SEIFA, or do we join independently after the augmentor pass? Resolved in Phase 3.
3. **Brand canonicalisation** ✅ resolved Phase 2: `data/static/brand_aliases.csv` is the canonical mapping. Unmapped brand strings produce a WARNING log and pass through verbatim — never fail. Initial seed built from Aug 2024 + Dec 2025 + Feb 2026 monthly archives. **Both** `brand_raw` and `brand_canonical` are persisted to `stations.parquet` so the model can pick up franchisee-vs-corporate pricing signal that would otherwise be erased.

   **Sub-question (open):** how do we identify franchisees vs corporate sites? Patterns like `EG Ampol` (Euro Garages franchisee) vs `Ampol Foodary` (corporate sub-brand) carry plausible pricing signal — and the *cross-brand* hypothesis is more interesting still: a franchisee's pricing behaviour may resemble other franchisees more than it resembles their own brand's corporate sites. So a single `stn_is_franchisee` boolean (and possibly a `stn_franchisee_operator` categorical, e.g. "EG", "EBM") could be a stronger signal than `stn_brand_raw` alone.

   **Research path** (Phase 4-ish, not blocking earlier work):
   - Build `data/static/brand_franchisee_rules.csv` with `raw_brand → is_franchisee, operator`. Sources:
     - Press releases / annual reports of major franchisee operators (EG Group, EBM, Reddy Express's history with Shell)
     - Australian Franchise Council registry (if accessible)
     - Brand-name pattern matching as a fallback: `^EG ` / `^EBM ` / `... Mobil 1 ...` etc. as proxies
     - ABN lookups against operator names if FuelCheck ever exposes operator metadata (it doesn't currently)
   - Schema: `raw_brand,is_franchisee,operator,confidence` — confidence in {`confirmed`, `pattern_match`, `inferred`} so analysts can filter to high-confidence only.
   - The cross-brand `operator` column lets feature engineering build aggregates like "median price among EG-operated sites within 10 km".

   For Phase 2 / 3, ship `brand_raw` + `brand_canonical` only and defer `is_franchisee` to a dedicated research pass before Phase 4 feature build.
4. **Petrol cycle as a sanity check** — should `01_eda.ipynb` verify the cycle is endogenously captured by lag features (e.g., by training a tiny model on lag features alone and inspecting predictions on a held-out station)? Nice-to-have.
5. **Crisis-period reporting** — confirm whether the test (crisis) fold is reported in the headline `comparison.md` or only as a sub-section. Suggest sub-section to keep the headline numbers comparable to a "normal world" baseline.
6. **Major events and spatially-granular holiday features** — v1 has two statewide calendar signals: `cal_is_public_holiday` (NSW-wide, from `python-holidays`) and `cal_is_school_holiday_nsw`. These are coarse — a station 500m from ANZ Stadium on State of Origin night is in a completely different demand environment from one in Dubbo on the same date. The model can't see that distinction today.

   **Status: research complete, awaiting EDA gate before implementation.**

   Full implementation plan in [`docs/research/2026-05_major_events_features.md`](docs/research/2026-05_major_events_features.md). Headlines:

   - **Eventbrite API:** effectively closed to new integrations (developer portal redirects). Skip.
   - **AFL via Squiggle API** (`api.squiggle.com.au`): free, open, 2000–present coverage. Use.
   - **NRL fixtures:** no public API; Wikipedia season-page scrape produces `data/static/nrl_fixtures.csv`. Use.
   - **Regional NSW show days:** not gazetted as public holidays; not in `python-holidays`. Sydney Royal Easter Show handled via `data/static/major_events.csv` instead.
   - **`data/static/major_venues.csv`:** hand-curated 10-venue pilot list (Accor, Allianz, SCG, etc.) — enables the EDA gate with zero API dependency.

   Phase 0 EDA gate: residual-by-venue-distance-quintile chart in `01_eda.ipynb` §10 (new). Decision rule: Q1 vs Q5 residual gap ≥ 1 c/L on holiday/pre-holiday rows → proceed; < 0.5 c/L → stop and record null result.

   **Status: tested, STOP.** Phase 0 EDA gate passed (Q1 vs Q5 gaps of +1.53 / +2.71 / +4.11 c/L) but with a flagged caveat: a non-monotonic Q4 dip suggested metro/regional cycle-amplitude confounding rather than pure venue proximity. Phase 1 sanity-check additive test confirmed it — Model B' (Model B + 5 venue/long-weekend features) lost by **+0.681 c/L MAE** on test_normal (33× the decision threshold in the wrong direction). `stn_nearest_venue_km` ranked #20 by gain but bled splits from better-generalizing features; `cal_is_pre_long_weekend` got zero gain because LightGBM was already extracting that signal from `cal_day_of_week × cal_days_to_next_public_holiday` interactions in the existing CAL block. Phases 2-4 (AFL/NRL fixture integration, Easter Show, etc.) are not justified. See [`docs/research/2026-05_major_events_phase1_outcome.md`](docs/research/2026-05_major_events_phase1_outcome.md) for full details. Code (`spatial/venues.py`, VENUE block in `feature_blocks.py`, Model B' training) and the 10-venue static CSV remain in place for cheap re-experimentation with different feature designs.

7. **Weather leakage fix (v2.0)** — v1's `wx_*` columns join Open-Meteo ERA5 *reanalysis actuals* on the same date as the prediction row, which means the model sees retrospective truth rather than the forecast a real-time predictor would have. `results/README.md` caveat #4 acknowledges this; absolute MAE is optimistic, though the A-vs-B comparison is unbiased (both models leak equally).

   **Status: ✅ LANDED (2026-05-29).** Outcome doc: [`docs/research/2026-05_weather_leakage_fix_outcome.md`](docs/research/2026-05_weather_leakage_fix_outcome.md).

   **Result summary** (v2.0 GFS day-ahead vs v1 ERA5 leaky):
   - **Absolute MAE rose 0.07-0.15 c/L** — within predicted leakage-tax range
   - **test_normal Δ MAE (B vs A)**: −0.391 → **−0.353** (essentially unchanged — A-vs-B comparison was indeed unbiased w.r.t. leakage)
   - **test_crisis Δ MAE**: −0.183 → **−0.398** (more than doubled — SA2 lift is now bigger AND more robust on OOD data)
   - **Crisis-fold RMSE regression fixed**: v1 had Model B's RMSE *worse* than A (18.739 vs 18.628); v2 has B better (18.578 vs 19.054)
   - v1 results/README.md caveat #2 ("crisis-fold lift is real but smaller and noisier") is **invalidated** by v2

   Original plan: [`docs/research/2026-05_weather_leakage_fix.md`](docs/research/2026-05_weather_leakage_fix.md) (Open-Meteo Historical Forecast API). Pre-flight at [`docs/research/2026-05_weather_leakage_preflight.md`](docs/research/2026-05_weather_leakage_preflight.md). **Empirical reality**: 2026-05-26/27 attempts to refetch all 4,587 NSW stations hit Open-Meteo's per-minute/per-hour/per-day rate limits within minutes, with each 429 itself counting against the daily quota — three attempts burned through the daily 10,000-call cap before completing ~3% of the fetch. The free tier is unworkable at this volume.

   **Revised plan**: [`docs/research/2026-06_nwp_archive_alternative.md`](docs/research/2026-06_nwp_archive_alternative.md). Strict-free path via **NOAA GFS 0.25°** (2021-04+) + **NOAA GEFS 1°/0.5°** (2017-01 → 2021-03) on anonymous AWS S3 buckets — no key, no quota, no 429s. Trade-off: ~10-14 hour one-time backfill via GRIB byte-range subsetting (7 horizons × 9 years of data). Open-Meteo path retained as optional paid-tier upgrade via `WEATHER_SOURCE=openmeteo` config switch + `OPENMETEO_API_KEY` env var.

   **Code landed (implementation commits, all on `claude/weather-leakage-fix-v2` branch):**
   - Session 1 (`17f491c`): GFS scaffolding + `spatial/gfs_grid.py` station-grid mapping
   - Session 2 (`14c9e2e`): `fetch/gfs.py` multi-horizon fetcher + `tools/parallel_gfs_fetch.py` orchestrator
   - Session 3 (`86f2071`): `build/make_features.py` `add_weather_features_gfs()` with bilinear interp + 35-column wide schema
   - Session 4a (`8ae01b7`, `4a7f9bc`, `3d3c4e8`, `25c651e`, `cb91719`, `36df15d`, `c3b23db`, `92d2dc1`, `0f54604`): `WEATHER_SOURCE` config router, `MODEL_*_GFS_BLOCKS`, `train_models` dispatch, categorical fix, `make_features` orchestrator dispatch, Makefile targets, docs (this entry), self-hosted-Open-Meteo backlog item (§13.9)
   - Session 4b (production fetch + retrain): GFS fetch ran 2026-05-28 18:03 → 2026-05-29 11:07 (~17h wall-clock), then features regen + train + evaluate completed 12:43. 3,029 dates fetched (89% of in-range); 374 NOAA archive gaps + 122 pre-2017 fail-fast = ~20% rows with null `wx_*_t1` (handled natively by LightGBM)

   **`WEATHER_SOURCE` env var contract:**
   - `auto` (default): picks `openmeteo` if `OPENMETEO_API_KEY` is set, else `gfs`
   - `gfs`: strict-free NOAA GFS/GEFS path
   - `openmeteo`: optional paid-tier Open-Meteo path

   **2016-09 → 2016-12 gap decision: null-stub for v2.0.** Recommended in the research doc was a one-time Open-Meteo Archive backfill, but today's evidence (Open-Meteo free-tier rate limits make even small fetches finicky) makes the deferred approach cleaner. 2016 Sept-Dec is 2.2% of training rows, train fold only — val (2023) and test_normal (2024-25) and test_crisis (2026) all use post-2017 dates with 100% wx coverage. LightGBM handles nulls natively. If post-retrain SHAP shows the 2016 nulls cost more than expected, the backfill is a clean v2.0.1 follow-up.

   Headlines:

   - **Sources:** NOAA GFS 0.25° + GEFS 1° + GEFS 0.5° hybrid for 2017-01 → present.
   - **Spatial smartness:** grid-cell caching, not per-station. ~600 unique GFS cells cover the 4,587-station NSW roster (many stations resolve to the same cell, especially in Sydney metro). Per-(date, lead) grid parquet + pre-computed `station_grid_mapping.parquet` with bilinear interpolation weights. Adding new stations later requires no API calls.
   - **Join semantics:** GFS parquet `<date>_h<N>.parquet` carries the forecast *issued on `date`, valid on `date + N` days*. Panel row at `t` reads `<t>_h<N>.parquet` directly — **no date shift needed** (unlike the Open-Meteo path which stored valid-date and required a -1d shift).
   - **Expected absolute MAE rise:** 0.05-0.15 c/L. A-vs-B Δ MAE unchanged.
   - **Methodological compromises:** `wx_weather_code` becomes null-stub (GFS/GEFS doesn't emit WMO codes; low SHAP rank); UTC vs Sydney day-boundary aggregation shifts daily aggregates by ~10h (low-rank feature, acceptable). 2016-09 → 2016-12 wx columns are null (~2.2% of training rows, train fold only).

8. **7-day forecast horizon (v2.1)** — extend v1's 1-day prediction (`y_t1`) to per-day predictions for t+1 through t+7. Substantial architectural change: target schema, multi-horizon feature engineering, per-horizon model training, evaluation.

   **Status: data path unblocked by NOAA GFS pivot (§13.7); modelling work still queued.**

   The data blocker that originally shelved this is now solved. NOAA GFS/GEFS publishes forecast lead times out to 384 hours (16 days), so for each prediction date `t` the same files used for the 1-day case also serve t+2, t+3, …, t+7 (via `f048`, `f072`, …, `f168` lead-time slices of the same 00Z run on day `t`). User decision (2026-05-27): bundle multi-horizon data path with §13.7 v2.0 implementation (~1 extra session of fetcher work, ~10-14 hours wall-clock total backfill) so the data layer is 7-day-ready from day one and v2.1 modelling can land as a clean follow-up.

   Full modelling plan still in [`docs/research/2026-05_7day_forecast_horizon.md`](docs/research/2026-05_7day_forecast_horizon.md) — Architecture A (one LightGBM per horizon), 5-8 sessions estimated for the modelling work itself (target schema, per-horizon weather joins, 14-model training loop, per-horizon evaluation, notebook updates). This is **unchanged** by the data-source pivot; only the upstream weather pipeline changed.

   Total path to v2.1 from current state: **~9-12 sessions** in 3 logical phases: (Phase 1+2) NOAA GFS/GEFS data pipeline with 7-horizon support [4 sessions, this is §13.7 v2.0 plus the multi-horizon bundle]; (Phase 3) v2.1 modelling work [5-8 sessions, future].

   Multi-horizon data path design + grid-cell caching architecture documented in [`docs/research/2026-06_nwp_archive_alternative.md`](docs/research/2026-06_nwp_archive_alternative.md) ("Multi-horizon extension" and "Grid-cell caching architecture" sections).

9. **Self-hosted Open-Meteo (long-term weather-pipeline alternative)** — Open-Meteo publishes the server software open-source ([getting-started docs](https://github.com/open-meteo/open-meteo/blob/main/docs/getting-started.md)) and the underlying NWP data on a public S3 mirror. Running the Open-Meteo stack ourselves would give us:

   - **No rate limits** (our own instance, our own resources)
   - **Multi-day lead times** out of the box via their Previous Runs API — directly unblocks v2.1 §13.8 with the same Python client we already wrote (`src/fuel_pred/fetch/weather.py`), instead of the GFS GRIB pipeline (§13.7).
   - **Pre-built variable derivations** including `weather_code` (the WMO code we null-stubbed in §13.7), wind gusts, cloud cover, etc. — recovers the small SHAP signal we discarded.
   - **Same call signature** as the existing Open-Meteo fetcher, so the pivot is a config swap of `WEATHER_SOURCE=openmeteo` + a base-URL override pointing at our instance.

   **Trade-off:** infra ownership. The Open-Meteo server has non-trivial system requirements (containerised, ~30 GB disk per model per global run, regular sync from S3). Worth it if v2.1+ weather work intensifies; not worth it for v2.0 alone (the §13.7 GFS pipeline already solves the rate-limit problem).

   **Status: backlog, no current action.** Revisit when (a) v2.1 multi-horizon modelling work begins and the §13.7 GFS pipeline's parse-time cost becomes painful, OR (b) we want the richer Open-Meteo derived variables. Either trigger justifies the infra investment; v2.0 alone does not.

10. **Time-series k-fold cross-validation + remote training offload** — replace the current two-fold reporting (test_normal vs test_crisis) with a proper time-series k-fold CV across the full panel, so feature-engineering comparisons are robust to fold-specific noise instead of betting on a single 2024-25 / 2026 split. The PR C experiments (see [`results/pr_c_overnight_summary.md`](results/pr_c_overnight_summary.md)) made this gap concrete: E1 wins test_normal while losing test_crisis, E4 does the reverse, and E5's "combine the wins" hypothesis blew up — single-split deltas don't generalise predictably. A rolling-origin or expanding-window k-fold (e.g. 6-10 folds across 2018-2026) would give us a per-experiment mean ± stdev that says "this change is robustly an improvement" vs "this change traded one fold for another." Past studies (PR A v2.0 bump, PR B temporal-mode adoption, PR C ablations) should arguably be re-run under the new methodology — null results re-evaluated, surviving wins reconfirmed.

    **Compute side** — k-fold × multi-experiment retrain pattern is significantly more expensive than today's single-fit. PR C's 7 experiments at ~30-60 min each fit a single CPU-bound LightGBM; a 6-fold CV would multiply that by 6 per experiment. To keep iteration practical, consider offloading training (and possibly feature build) to a home AMD server with substantially more cores + memory than the dev laptop. Implementation sketch:
    - **Docker container** wrapping the train + evaluate stages with the project's `uv.lock` baked in, so the home server can pull + run without env setup
    - Container reads features from a mounted volume (or fetches from a bucket), writes models/predictions/comparison back the same way
    - Orchestrator on the laptop becomes "build features locally → ship features parquet to remote → trigger remote train → pull artefacts back" instead of running everything inline
    - Possibly: distribute the k-fold loop across remote workers if k-fold becomes the standard pattern

    **Status: backlog, plan-and-discuss.** Will need a dedicated session to:
    (a) pick a CV scheme (expanding-window vs rolling-origin; fold count; gap-fold behaviour);
    (b) decide whether to retrofit historical experiments or just start from now;
    (c) design the Docker/remote-training handoff (container build, mount strategy, artefact return path, security).

## 14. References

- `abs-census-augmentor`: https://github.com/cauldnz/abs-census-augmentor
- NSW FuelCheck dataset: https://data.nsw.gov.au/data/dataset/fuel-check
- TfNSW Traffic Volume Counts: https://opendata.transport.nsw.gov.au/data/dataset/nsw-roads-traffic-volume-counts-api
- RBA historical data: https://www.rba.gov.au/statistics/historical-data.html
- Open-Meteo: https://open-meteo.com/
- ABS 2021 Census GCP DataPack: https://www.abs.gov.au/census/find-census-data/datapacks
- ABS SEIFA 2021: https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/latest-release
