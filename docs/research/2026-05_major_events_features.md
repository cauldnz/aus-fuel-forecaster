# Major events + spatially-granular holiday features — implementation plan

**Date:** 2026-05
**Status:** research complete, awaiting EDA Phase 0 gate before implementation
**Spec section:** `spec.md` §13.6
**Related:** `docs/research/2026-05_centrelink_seifa_interaction.md` (the parallel "spatial behavioural-economic" hypothesis test that came back as a null result)

## TL;DR

v1's calendar features are statewide: `cal_is_public_holiday` (NSW-wide bool, from `python-holidays`) and `cal_is_school_holiday_nsw`. A station 500m from ANZ Stadium on State of Origin night is in a completely different demand environment from one in Dubbo on the same date — currently the model can't see that. This research scopes the data sources and feature design for closing that gap. **Before any code is written, an EDA gate test must confirm the signal is present in existing data.** If the gate fails, the work stops.

## The hypothesis

Fuel price behaviour at stations near major event venues differs from the statewide average — for example, on the Friday before a long weekend, stations near major travel arteries / venue precincts see different demand than rural stations. The current model has no spatial component to its holiday features.

## Phase 0 — EDA gate (run before any API integration)

Add a new section §10 to `notebooks/01_eda.ipynb` immediately after §9c. The cell pattern mirrors the Centrelink-day × SEIFA chart in §6 but stratifies by **distance to nearest major venue** instead of SEIFA quintile.

### Pre-condition

Requires a hand-curated `data/static/major_venues.csv` (10 pilot venues — see §C.5 below). No API needed; just the static CSV.

### What we measure

Three cells, in order:

1. **§10a — Venue distance join.** Per station, compute haversine distance to nearest major venue (BallTree, ~1 second for 1,500 stations × 10 venues). Bin into quintiles (Q1 = closest, ~≤2 km; Q5 = furthest, >15 km).

2. **§10b — Residual by venue-distance quintile × day type.** U91 price residual (= `price_mean` − `roll_price_mean_28`) grouped by `(day_type, venue_dist_q)`, where `day_type ∈ {normal, public_holiday, day_before_holiday}`. Plot as grouped bar chart.

3. **§10c — Long-weekend Friday probe.** Same residual measure restricted to Fridays before Monday public holidays (`cal_day_of_week == 4 AND cal_days_to_next_public_holiday == 3`). Travel-surge proxy.

### Decision rule

- **Q1 vs Q5 residual gap on holiday/pre-holiday rows ≥ 1 c/L** → signal is present, proceed to Phase 1
- **Gap < 0.5 c/L across all day types** → signal absent, record null result in this doc and stop. The Centrelink × SEIFA test went the same way; a null result here would be similar evidence that statewide flags already capture the variance.
- **Between 0.5 and 1.0 c/L** → ambiguous. Look at the §10c long-weekend Friday chart specifically; if that one shows clear Q1/Q2 elevation, proceed with the long-weekend feature only and defer the full event-calendar build.

## Data sources verdict

| Source | Status | Verdict |
|---|---|---|
| **Eventbrite API** | Developer portal at `developer.eventbrite.com` returns 307 redirect to homepage. The v3 REST API programme appears effectively withdrawn for new integrations since the 2019 Eventbrite layoffs. | **DEFER.** Even if access were obtained, the 2016–2020 historical gap would null-out 4+ years of training rows. Revisit only if the developer programme reopens. |
| **NRL fixtures** | `api.nrl.com` is not a public API (CORS + network blocks). No third-party API with 2016+ coverage found. Wikipedia season articles (`en.wikipedia.org/wiki/{year}_NRL_season`) are scrapable with `pandas.read_html`. | **USE — manual CSV path.** Build `data/static/nrl_fixtures.csv` via one-off Wikipedia scrape, committed to repo. Annual maintenance. |
| **AFL fixtures (Squiggle API)** | `api.squiggle.com.au` is free, open, no-auth, covers 2000–present. Returns venue *name* string only (no lat/lon), no attendance. | **USE.** Implement `fetch/afl.py`. Total requests: ~10 per pipeline run (one per year). Venue name → `venue_id` lookup via `major_venues.csv`. |
| **NSW regional show days** | Confirmed: NSW does **not** have statewide gazetted show holidays. `python-holidays` does not include NSW show days (it does include QLD's Ekka). | **DEFER regional shows.** Treat Sydney Royal Easter Show as a major recurring event in `data/static/major_events.csv` instead (12-day window at Sydney Showground, ~850–950k attendees). |
| **`major_venues.csv` static seed** | Does not exist. Can be hand-curated immediately from public information. | **USE immediately.** Enables EDA gate without any API dependency. |
| **Long-weekend surge** | Fully derivable from existing `cal_days_to_next_public_holiday` and `cal_day_of_week` columns. | **USE — derived only.** No new data source. Add `cal_is_pre_long_weekend` to `add_calendar_features()`. |

## Feature design

All new columns follow existing naming conventions: `cal_*` for derived calendar, `stn_*` for station-static, `evt_*` for new time-varying event block.

### New `cal_*` columns (in `add_calendar_features()`)

| Column | Type | Description |
|---|---|---|
| `cal_is_pre_long_weekend` | `bool` | Friday before a Monday public holiday. `cal_day_of_week == 4 AND cal_days_to_next_public_holiday == 3` |
| `cal_is_school_holiday_eve` | `bool` | Last school-term day before a holiday break begins |
| `cal_days_to_school_holiday_start` | `Float64` | Mirror of `cal_days_to_next_public_holiday` for school holidays |

### New `stn_*` static columns (in `add_station_features()`)

Computed once at spatial join time, stored in `data/interim/stations_venues.parquet`.

| Column | Type | Description |
|---|---|---|
| `stn_nearest_venue_km` | `float32` | Haversine distance to nearest major venue |
| `stn_nearest_venue_capacity` | `float32` | Capacity of nearest venue (proxy for event-day demand) |
| `stn_nearest_venue_type` | `category` | `stadium` / `showground` / `entertainment_centre` / `racecourse` |
| `stn_n_venues_within_5km` | `Int64` | Count of major venues within 5 km — captures precinct effects (Sydney Olympic Park has 3+ venues clustered) |

### New `evt_*` time-varying columns (new `add_event_features()` function)

Joined on `(station_id, date)` from `data/interim/events_calendar.parquet`.

| Column | Type | Description |
|---|---|---|
| `evt_is_event_day_nearest` | `bool` | Event at the station's nearest venue today |
| `evt_is_event_day_within_5km` | `bool` | Any major venue within 5 km has an event today |
| `evt_days_to_next_event_nearest` | `Float64` | Days until next event at nearest venue |
| `evt_days_since_last_event_nearest` | `Float64` | Days since last event at nearest venue |
| `evt_max_event_capacity_within_5km` | `float32` | Maximum event-venue capacity within 5 km on the current day (0 if no event) |

Block size: 5 columns. Slots between `cal_*` and `ctx_*` in the feature matrix.

### Intermediate asset — `data/interim/events_calendar.parquet`

Schema: `event_date, venue_id, event_type ∈ {nrl, afl, show, concert}, source`. No `station_id` — the join to stations is done at feature-build time using `stn_nearest_venue_km` / `stn_n_venues_within_5km`. This separation keeps the event calendar independent of the station roster.

## Pipeline additions

### New static files

- `data/static/major_venues.csv` — schema `venue_id,name,lat,lon,capacity,type`. Maintainer + agent owned (append-only, same policy as `brand_aliases.csv`).
- `data/static/nrl_fixtures.csv` — schema `date,venue_name,home_team,away_team,round,season`. Maintainer-updated annually after each NRL season ends.
- `data/static/major_events.csv` — schema `event_id,venue_id,event_name,start_date,end_date,annual_attendance_estimate,recurrence_note`. Multi-day recurring events (Easter Show, etc.).

### New fetch modules

- **`src/fuel_pred/fetch/afl.py`** — Squiggle API → `data/raw/afl/games_{year}.parquet` per year. Cache skip if `complete=100`. User-Agent: `fuel-pred/0.1 (https://github.com/cauldnz/fuel-prediction)` (required by Squiggle).
- **`tools/scrape_nrl_wikipedia.py`** — integration tool (per project convention), one-off Wikipedia scrape producing `data/static/nrl_fixtures.csv`.

### New build modules

- **`src/fuel_pred/build/build_event_calendar.py`** — reads AFL parquets + NRL CSV + major events CSV, normalises venue names → `venue_id`, emits `data/interim/events_calendar.parquet`. Log WARNING for unmatched venues; fail loudly if > 20% unmatched.
- **`src/fuel_pred/spatial/venues.py`** — station → venue distance join, emits `data/interim/stations_venues.parquet`.

### `make_features.py` changes

- New `add_event_features(df, events_calendar, stations)` function (new §7.9 block in spec).
- Add optional `events_calendar` and `stations_venues` parameters to `make_features()` (None-tolerant, following `aip_tgp` / `cash_rate` pattern).
- Add the 3 new `cal_*` columns to `add_calendar_features()`.
- Add the 4 new `stn_*` columns to `add_station_features()` when `stations_venues.parquet` present.

### New `config.py` paths

```python
RAW_AFL: Path = DATA_RAW / "afl"
INTERIM_EVENTS_CALENDAR: Path = DATA_INTERIM / "events_calendar.parquet"
INTERIM_STATIONS_VENUES: Path = DATA_INTERIM / "stations_venues.parquet"
STATIC_MAJOR_VENUES: Path = DATA_STATIC / "major_venues.csv"
STATIC_MAJOR_EVENTS: Path = DATA_STATIC / "major_events.csv"
STATIC_NRL_FIXTURES: Path = DATA_STATIC / "nrl_fixtures.csv"
```

## Implementation order

| Phase | Scope | Effort |
|---|---|---|
| **0** | EDA gate — hand-curate `major_venues.csv`, add §10 cells to `01_eda.ipynb`, run, decide go/no-go | ½ session |
| **1** | Static venue features (`stn_nearest_venue_*`) + derived calendar (`cal_is_pre_long_weekend`, etc.) | 1 session |
| **2** | AFL event calendar (Squiggle API → `events_calendar.parquet` → `evt_*` features) | 1 session |
| **3** | NRL fixture integration (Wikipedia scrape → static CSV → merge into calendar) | 1 session |
| **4** | Sydney Royal Easter Show + other multi-day recurring events | ½ session |

## Pilot venue list (Phase 0 seed for `major_venues.csv`)

| venue_id | name | lat | lon | capacity | type |
|---|---|---:|---:|---:|---|
| accor_stadium | Accor Stadium | -33.8473 | 151.0631 | 83,500 | stadium |
| allianz_stadium | Allianz Stadium | -33.8912 | 151.2233 | 45,500 | stadium |
| scg | Sydney Cricket Ground | -33.8918 | 151.2246 | 48,000 | stadium |
| sydney_showground | Sydney Showground | -33.8475 | 151.0631 | 70,000 | showground |
| commbank_stadium | CommBank Stadium | -33.8158 | 150.9942 | 30,000 | stadium |
| mcdonalds_jones_stadium | McDonald Jones Stadium | -32.9265 | 151.7817 | 33,000 | stadium |
| qudos_bank_arena | Qudos Bank Arena | -33.8467 | 151.0644 | 21,000 | entertainment_centre |
| engie_stadium | ENGIE Stadium | -33.8473 | 151.0631 | 20,000 | stadium |
| bluebet_stadium | BlueBet Stadium (Penrith) | -33.7510 | 150.6943 | 22,000 | stadium |
| netstrata_jubilee | Netstrata Jubilee Stadium | -33.9661 | 151.1285 | 24,000 | stadium |

## Risks and open questions

### R1. Signal may not be there
The Centrelink × SEIFA hypothesis was tested in v1 and came back as a partial null (main effect strong, interaction weak — see the companion research note). The same discipline applies here: gate on Phase 0 EDA. If the residual gap by venue-distance quintile is < 0.5 c/L, the entire feature set is dead-on-arrival.

### R2. Venue name matching is fragile
AFL Squiggle returns venue names as strings (`"S.C.G."`, `"Stadium Australia"`). Stadium Australia was renamed Accor Stadium / ANZ Stadium / Telstra Stadium across 2016–2024. The lookup dict in `build_event_calendar.py` is a source of silent data quality issues. **Mitigation:** WARNING per unmatched venue; fail loudly if > 20% unmatched.

### R3. NRL Wikipedia scrape is year-dependent
Table structure varies (some years use round-by-round subsections, others flat tables). The scraper may need year-specific table-index arguments for pre-2020 seasons. **Mitigation:** iterate the scraper against 2016, 2018, 2020, 2022 first; fall back to NRL Media PDFs via Wayback Machine for unreliable years.

### R4. Squiggle is a fan project, not official AFL
Single-developer maintenance, no SLA. Could go offline. **Mitigation:** cache all historical parquets permanently (`complete=100` games are immutable). Only re-fetch the current season.

### R5. Feature leakage in `evt_days_to_next_event_nearest`
The model knows when the next event is. Legitimate only if the event was publicly scheduled before the prediction date. Fixtures are usually announced months ahead so this is fine. **Mitigation:** spec requirement that the event calendar contains only originally-scheduled dates, not retrospective makeup dates for postponements.

### R6. Static venue features ignore venue lifecycle
Allianz Stadium was demolished in 2020 and reopened 2022. Static CSV bakes in 2024 capacity for all years. **Decision needed:** accept the approximation (geographic relationship is stable even during reconstruction) or add `valid_from`/`valid_to` columns. Recommend acceptance with a TODO; time-aware join is significant added complexity.

### R7. Non-sport events gap
AFL + NRL covers stadium events but misses Mardi Gras, Vivid, NYE fireworks, Field Day. **Decision needed:** accept gap for v1 (~5 events/year that the model can't see). Revisit if Phase 2/3 EDA shows residual unexplained variance near non-sport precincts (CBD, Darling Harbour).

## See also

- `spec.md` §13.6 — the spec entry pointing to this doc
- `docs/research/2026-05_centrelink_seifa_interaction.md` — companion null-result on the spatial-behavioural-economic hypothesis design
- `notebooks/01_eda.ipynb` §6 — the Centrelink × SEIFA chart that established the "residual by quintile × time" methodology this doc proposes for venue distance
