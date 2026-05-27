# NWP archive alternative — strict-free weather source for v2

**Date:** 2026-06 (research in late May 2026; multi-horizon + grid-cell-caching extensions added late May 2026)
**Status:** research complete, ready to plan implementation
**Spec section:** `spec.md` §13.7 (revised approach) and §13.8 (un-shelved — see "Multi-horizon extension" below)
**Related:** [`docs/research/2026-05_weather_leakage_fix.md`](2026-05_weather_leakage_fix.md) (the original plan; this supersedes its source-selection on the free path), [`docs/research/2026-05_weather_leakage_preflight.md`](2026-05_weather_leakage_preflight.md), [`docs/research/2026-05_7day_forecast_horizon.md`](2026-05_7day_forecast_horizon.md) (the modelling plan; data path now unblocked by this doc)

## TL;DR

Replace Open-Meteo free-tier (which rate-limited us into uselessness at 4,587-station scale) with a hybrid of two anonymously-accessible NOAA AWS buckets — **NOAA GEFS operational `pgrb2a` 1° control-member** for 2017-01-01 → 2020-09-22 and **NOAA GFS 0.25°** from 2021-04-01 onwards — bridged by a ~6-month gap that's covered by either NOAA GEFS 0.5° (2020-09-23 → 2021-03-31) or a one-time Open-Meteo Archive backfill at small volume. Keep `src/fuel_pred/fetch/weather.py` (Open-Meteo) untouched as an optional paid-tier path via a new `WEATHER_SOURCE` config switch. Estimated implementation effort: **3 sessions** — the GRIB pipeline and per-station spatial subset dominate.

## Why we're doing this

The v2 weather leakage fix ([`2026-05_weather_leakage_fix.md`](2026-05_weather_leakage_fix.md)) was scoped against the v1 fetcher's working assumption of ~1,500 NSW stations. Reality at v2 fetch time: 4,587 stations × one Open-Meteo call per station per refetch. The free tier's claimed 10,000 calls/day ceiling is hit on the first refetch (4,587 stations alone, no retries, no incremental coverage extension). The two refetch attempts on 2026-05-26 and 2026-05-27 both fell over to 429s within minutes of starting.

The user has decided: **strict-free is the primary path; commercial Open-Meteo (via `OPENMETEO_API_KEY`) stays as an optional flag-flip for users who have that plan**. The investigation below finds a free path that's slower and more complex than Open-Meteo would be at scale, but is unconditionally accessible without auth or quota.

## Recommended approach

**Hybrid two-source GFS/GEFS pipeline:**

| Window | Source | Bucket | Resolution | Per-day raw bytes (one cycle, 4 lead-times) |
|---|---|---|---|---|
| 2017-01-01 → 2020-09-22 | NOAA GEFS operational `gec00` (control member, `pgrb2a` 1° subdir) | `noaa-gefs-pds` | 1° (~111 km) | ~1 MB |
| 2020-09-23 → 2021-03-31 | NOAA GEFS operational `geavg` (ensemble mean, `pgrb2ap5` 0.5° subdir) | `noaa-gefs-pds` | 0.5° (~55 km) | ~5 MB |
| 2021-04-01 → present | NOAA GFS deterministic (`pgrb2.0p25`) | `noaa-gfs-bdp-pds` | 0.25° (~28 km) | ~10 MB (byte-range subset; full file is 522 MB) |
| 2016-09-01 → 2016-12-31 (gap) | Open-Meteo Archive (ERA5) at small volume — pre-existing v2 fallback per source doc | (existing fetcher) | ~28 km | n/a (low call volume) |

The 2016 gap is the same as the original v2 plan and uses the same ERA5 fallback (~4,587 calls one-off, doable at the existing rate-limit settings if run overnight with `--inter-call-seconds=1.0`). The 2017-2020 and 2020-2021 windows use the resolution that's natively available in the respective subfolder layouts on `noaa-gefs-pds` — see the empirical layout transition section below.

**Why not single-source GFS 0.25° throughout:** the `noaa-gfs-bdp-pds` bucket starts 2021-04-01 only. Empirically confirmed: HTTP 404 for `gfs.20210301/00/atmos/gfs.t00z.pgrb2.0p25.f024`, HTTP 200 for `gfs.20210401/...`. Confirmed at 2021-03-15 (no) and 2021-04-15 (yes); the 2021-04-01 boundary is sharp.

**Why not GEFS reforecast as the long-history backbone:** GEFSv12 reforecast (`noaa-gefs-retrospective` bucket) goes back to 2000-01-01 and would be appealing — but it's **only published through 2019-12-31** (a one-time retrospective dataset) and has a per-variable-per-cycle file structure that doesn't include `ugrd_10m`/`vgrd_10m` for 10m wind (only `gust_sfc` is available at the surface). The operational GEFS bucket is simpler and covers our 2017-2020 window directly with the same variables we ultimately need for 2021+.

## Source-by-source analysis

All probes empirical (HTTP-listed via the public AWS REST listing, no boto3 required), except items 5–8 marked "doc-only".

| # | Source | Access | AU coverage | Format | Verdict |
|---|---|---|---|---|---|
| 1 | **NOAA GFS 0.25° deterministic** (`s3://noaa-gfs-bdp-pds/`) | Anonymous HTTPS, S3 v2 listing API | **2021-04-01** strict | GRIB2 (~522 MB per file, ~10 MB byte-range subset for our vars/lead-times) | USE for 2021-04+ |
| 2 | **NOAA GEFS operational** (`s3://noaa-gefs-pds/`) — control member at 1° | Anonymous HTTPS | **2017-01-01** confirmed (probably earlier) | GRIB2 (~3.9 MB per full file) | USE for 2017-01 → 2020-09 |
| 3 | **NOAA GEFS operational** ensemble mean at 0.5° (post-2020-09-23 layout) | Anonymous HTTPS | 2020-09-23 → present | GRIB2 (~14 MB per file) | USE for 2020-09 → 2021-03 bridge |
| 4 | **NOAA GEFSv12 reforecast** (`s3://noaa-gefs-retrospective/`) | Anonymous HTTPS | 2000-01-01 → 2019-12-31 | GRIB2, per-variable files (~30 MB each) | SKIP — no 10m wind, ends 2019, file-per-variable makes per-day fetches more chatty |
| 5 | **NCEI GFS historical** (THREDDS / HAS request) | Order-then-wait | 1°: 2004-02+, 0.5°: 2007-01+, 0.25°: 2021-02+ | GRIB2 | SKIP — offline retrieval workflow incompatible with cache-first design |
| 6 | **ECMWF Open Data** (`data.ecmwf.int/forecasts/`) | HTTP listing | **~4 days rolling only** (re-confirmed 2026-05) | GRIB2 | SKIP — no historical retention |
| 7 | **Microsoft Planetary Computer `ecmwf-forecast`** | STAC + Azure Blob | 30-day rolling per STAC collection metadata | GRIB2 | SKIP — no historical retention |
| 8 | **Google Cloud `gcp-public-data-arco-era5`** | Anonymous GCS | ERA5 from 1940-01-01 | NetCDF/Zarr | SKIP for our use-case — this is reanalysis, same leakage as Open-Meteo Archive |
| 9 | **BOM ACCESS-G / data.gov.au** | Public-but-not-NWP-archive | No daily forecast archive identified; only "rainfall, temperature and wind forecast verification" datasets (2015-05 → 2018-04) covering NSW spatially but not at our resolution | n/a | SKIP — coverage too narrow, format unsuitable |
| 10 | **Commercial fallbacks** (Visual Crossing, OpenWeatherMap, Tomorrow.io) | API keys + metered cost | full | JSON | SKIP — strict-free brief |

The GEFS empirical layout transition is worth recording: on `noaa-gefs-pds`, the file path schema changes twice in the period of interest:

- 2017-01-01 → 2018-12-31: `gefs.YYYYMMDD/HH/gec00.t{HH}z.pgrb2af{NNN}` (control member only at top level, 3.9 MB)
- 2019-01-01 → 2020-09-22: `gefs.YYYYMMDD/HH/pgrb2a/gec00.t{HH}z.pgrb2af{NN}` (control member in `pgrb2a` subdir, 1° still — note lead-time digit count changes from 3 to 2)
- 2020-09-23 → present (GEFSv12): `gefs.YYYYMMDD/HH/atmos/pgrb2ap5/geavg.t{HH}z.pgrb2a.0p50.f{NNN}` (0.5° ensemble mean in `atmos/pgrb2ap5` subdir, 14 MB; control member also available)

The fetcher needs three path templates and a date-based selector — annoying but mechanical.

## Variable mapping

| Our col | Source variable (GRIB short name : level : fcst type) | Notes |
|---|---|---|
| `wx_temp_max_c` | `TMAX:2 m above ground:N-(N+6) hour max fcst` | GFS/GEFS emits 6-hour max blocks. Aggregate four 6h blocks (f006 max, f012 max, f018 max, f024 max from run 00Z on day `t`) to get day-`t+1` 24h max. Convert K→°C. |
| `wx_temp_min_c` | `TMIN:2 m above ground:N-(N+6) hour min fcst` | Same as above, with min reducer. |
| `wx_precipitation_mm` | `APCP:surface:N-(N+6) hour acc fcst` | 6-hour accumulation. Sum f006+f012+f018+f024 blocks for day `t+1` total. Already in mm. |
| `wx_wind_speed_max_kmh` | `UGRD:10 m above ground` + `VGRD:10 m above ground`, instantaneous at f006/f012/f018/f024 | Compute scalar wind speed `sqrt(U² + V²)` at each lead, take max across the 4 leads. Convert m/s → km/h (× 3.6). NB this is *instantaneous max-of-4-snapshots*, not true day-max — coarser than Open-Meteo's `wind_speed_10m_max`. Better proxy: GEFS also exposes `GUST:surface` if available; for GEFS pgrb2a 1° it is at f024 (record 14 in the sample idx) — recommend using GUST when available. |
| `wx_weather_code` | Not directly available | GFS/GEFS does not emit WMO weather codes. **Recommendation: drop this column from v2**, or stub it with a derived code from `APCP` (>1 mm → rain code 61) + `TCDC` (entire atmosphere total cloud cover, available in both buckets) using a simple lookup table. The original Open-Meteo `weather_code` column had low SHAP importance per v1 analysis; dropping it costs <0.01 c/L MAE. |

**Variable inventory confirmed empirically** on GEFS `gec00.t00z.pgrb2af024.idx` for 2018-06-01 (83 records total):
- Record 63: TMAX 2m 18-24h max
- Record 64: TMIN 2m 18-24h min
- Record 65: UGRD 10m
- Record 66: VGRD 10m
- Record 67: APCP surface 18-24h acc
- Record 75: TCDC entire atmosphere 18-24h ave (cloud cover, available for weather_code proxy)

And on GFS 0.25° `gfs.t00z.pgrb2.0p25.f024.idx` for 2024-06-01 (743 records total — much fatter than GEFS):
- Record 14: GUST surface (24h fcst) — useful for wind max
- Record 581: TMP 2m 24h (instantaneous)
- Record 586: TMAX 2m 18-24h max
- Record 587: TMIN 2m 18-24h min
- Record 588/589: UGRD/VGRD 10m
- Record 596: APCP surface 18-24h acc
- Record 597: APCP surface 0-1 day acc (single record for day-1 total — even simpler)

Note: GFS 0.25° exposes a single `APCP:surface:0-1 day acc fcst` record at f024 — one byte-range gets the full day-1 precipitation total. GEFS 1° only has the 6h-acc records and we'd sum four; not a real cost difference.

**Day boundary alignment.** The Open-Meteo pipeline uses `Australia/Sydney` daily boundaries. GFS/GEFS emits in UTC. To match: the day-`t+1` 24h window in Sydney time is 14:00 UTC on day `t` → 13:00 UTC on day `t+1` (in AEST, UTC+10) or 13:00→12:00 UTC for AEDT (UTC+11). A clean approximation is "take the 00Z run from day `t`, use lead times f014..f038" to cover the Sydney day. **Recommend** for v2: use `f006..f030` (UTC day boundary aligned to the run) and document that the daily aggregate is UTC-day, not Sydney-day. The 10-hour misalignment is below the noise floor for daily aggregates of temperature and rainfall; it only matters at the edge of unusual heat waves or storm fronts. (If exact Sydney-day alignment is needed, do client-side hourly resampling on a wider lead range — complexity not worth it for v2.)

## Spatial subsetting

**Recommended: AWS S3 byte-range requests using the `.idx` files.** Each GRIB file has a sidecar `.idx` text file with one line per record giving its byte offset and a description string (`record_no:byte_offset:date:variable:level:fcst_type:`). To fetch one variable for the whole NSW box:

1. `GET .idx` (~30–50 KB, very cheap).
2. Parse for our 5 target lines.
3. For each, issue a `GET` with `Range: bytes=offset-(next_offset-1)`. Empirically: HTTP 206 returns ~490 KB for one global TMAX 2m record at 0.25°, ~50 KB at 1°.
4. The downloaded bytes are a valid mini-GRIB starting with `GRIB` magic and ending with `7777` — empirically verified.
5. Parse with cfgrib → xarray DataArray → subset by `lat ∈ [-37.5, -28]`, `lon ∈ [140.5, 154]` → interpolate at each station lat/lon (bilinear via `xarray.interp`).

**Why not full-file download + cfgrib:** at 0.25° each file is 522 MB and we'd read ~1% of it. 522 MB × 4 leads × 1,710 days ≈ **3.4 TB** of GFS-only download for a one-time historical backfill. Byte-range subsetting cuts this to ~17 GB.

**Why not pre-subset community projects:** searched — no NSW or AU-only GFS extract that's actively maintained. NOAA's official `gfs_atm_da` is global; pre-cooked subsets exist for hurricane research (Atlantic) but not AU.

## Pre-2021 gap strategy

GFS 0.25° starts 2021-04-01. Our training data starts 2016-09-01, so ~4.5 years of pre-GFS need other treatment:

| Sub-window | Strategy |
|---|---|
| 2017-01-01 → 2020-09-22 | NOAA GEFS operational `gec00` 1° (3.9 MB/file). Same byte-range pipeline as GFS but with coarser grid — 1° ≈ 111 km, NSW spans ~10° lat × 14° lon = 140 grid points. Plenty of resolution for daily aggregates that get joined to per-station rows by nearest-neighbour. |
| 2020-09-23 → 2021-03-31 | NOAA GEFS operational `geavg` 0.5° (14 MB/file) in the v12 `pgrb2ap5` subfolder. Same variable inventory. Bridges to GFS 0.25° clean. |
| 2016-09-01 → 2016-12-31 | One-time Open-Meteo Archive (ERA5) fetch at low rate. 4,587 stations × ~120 days = ~550k row-equivalents, but Open-Meteo bills per station call not per row, so it's 4,587 calls one-off — well within free-tier daily ceiling if run overnight at 1s spacing. Same leakage caveat as v1; documented as `wx_source=era5_persistence` for this window only. |

**Alternative: skip 2016 entirely** (null-fill the `wx_*` for those rows). v1 had 100% leakage and shipped; v2's 2016-Sep window is 2.2% of training rows. LightGBM null-handling is robust. Recommend: include the one-time ERA5 backfill rather than dropping — it's a one-time cost and the rows carry useful non-weather signal anyway.

**Why not ERA5 persistence (yesterday's actual as today's "forecast") throughout 2017-2020:** the GEFS GRIB pipeline is being built anyway for the gap 2020-09 → 2021-03; extending it back to 2017-01 adds zero code and gives us real forecast data instead of persistence-proxy. The persistence option is strictly worse here.

## Integration plan

### Module layout

```
src/fuel_pred/fetch/
├── weather.py           # UNCHANGED — Open-Meteo Archive/HFA path (paid-tier optional)
├── gfs.py               # NEW — anonymous AWS GFS/GEFS byte-range fetcher + GRIB parser
└── weather_router.py    # NEW (or fold into make_features) — selects source per date+config
```

`gfs.py` public API:

```python
def fetch_one(station_id: str, lat: float, lon: float,
              start: str, end: str, out_dir: Path,
              *, force: bool = False) -> Path | None:
    """One station, full date range. Selects GFS/GEFS subwindow per date."""

def fetch(stations_path: Path, start: str, end: str, out_dir: Path, *,
          force: bool = False) -> None: ...

def _list_cycles(date: dt.date) -> list[str]: ...     # ["00", "06", "12", "18"] historically; subset for daily run
def _build_url(date: dt.date, cycle: str, lead_h: int) -> str: ...    # routes between three path schemes
def _fetch_idx(url: str) -> dict[str, tuple[int, int]]: ...           # var_key -> (start_byte, end_byte)
def _fetch_records(url: str, byte_ranges: list[tuple[int,int]]) -> bytes: ...
def _parse_grib_to_xarray(grib_bytes: bytes) -> xr.DataArray: ...     # via cfgrib (with temp file)
def _aggregate_day(per_lead_arrays: dict[int, xr.DataArray], reducer: str) -> xr.DataArray: ...
def _interp_at_stations(arr: xr.DataArray, stations: pd.DataFrame) -> pd.DataFrame: ...
```

Schema-compatible output: per-station `data/raw/weather/<station_id>.parquet` matching the existing 6 columns (`date, wx_temp_max_c, wx_temp_min_c, wx_precipitation_mm, wx_wind_speed_max_kmh, wx_weather_code`). The last column is null-filled if the weather-code-derivation is not implemented (recommendation: ship null first, derive in a follow-up if v1 SHAP says it matters).

CLI:

```
python -m fuel_pred.fetch.gfs \
    --stations data/interim/stations.parquet \
    --start 2017-01-01 --end 2026-04-30 \
    --out data/raw/weather \
    [--force] [--cycles 00] [--lead-hours 6,12,18,24]
```

### Hybrid router

`make_features.py` `add_weather_features()` stays unchanged: it reads `data/raw/weather/<station_id>.parquet` and joins with the 1-day shift (per the v2 leakage fix). The router lives at *fetch* time:

```python
# config.py additions
WEATHER_SOURCE: Literal["gfs", "openmeteo", "auto"] = os.environ.get("WEATHER_SOURCE", "auto")
GFS_S3_BUCKET: str = "noaa-gfs-bdp-pds"
GEFS_S3_BUCKET: str = "noaa-gefs-pds"
GFS_COVERAGE_START: str = "2021-04-01"          # GFS 0.25 strict start
GEFS_PGRB2A_END: str = "2020-09-22"             # last day before GEFSv12 layout change
GEFS_PGRB2AP5_START: str = "2020-09-23"
ERA5_FALLBACK_END: str = "2016-12-31"           # one-time Open-Meteo Archive backfill window
OPENMETEO_API_KEY: str | None = os.environ.get("OPENMETEO_API_KEY")  # only used in "openmeteo" mode
```

- `WEATHER_SOURCE=gfs` (or `auto` without key): GFS+GEFS hybrid via `gfs.py`, ERA5 backfill for 2016 via `weather.py` Open-Meteo Archive (run one-shot at slow pace).
- `WEATHER_SOURCE=openmeteo` (or `auto` with key): existing `weather.py` HFA path, no GFS code touched.
- `WEATHER_SOURCE=auto`: pick `openmeteo` if `OPENMETEO_API_KEY` is set, else `gfs`.

The Makefile gets a new target `fetch-weather-gfs` and a comment on the existing `fetch-weather` that it's the Open-Meteo path.

### New dependencies

| Dep | Purpose | Install pain |
|---|---|---|
| `boto3` | OPTIONAL — anonymous S3 ListBucket is available via plain HTTPS too (probed in this research with `requests`), so boto3 is **not strictly required**. Recommend skipping it; one less dep. | n/a (skipped) |
| `cfgrib` | GRIB2 parsing via xarray | Requires `eccodes`. Pre-built wheels ship from PyPI 2.37+ for Linux/macOS/Windows; install is `pip install eccodes cfgrib` — empirically straightforward on Python 3.11+. If wheels fail on a future Windows build, fall back to `conda install -c conda-forge cfgrib`. |
| `xarray` | DataArray manipulation + bilinear interp | Already in scientific Python stack |
| `pygrib` | ALTERNATIVE to cfgrib — direct GRIB1/2 binding | More mature on Windows historically, but cfgrib has caught up. Recommend cfgrib for the xarray integration. |
| `wgrib2` (CLI) | Reference subsetting tool | Not bundled in Python; skip — cfgrib path is enough |

Action item: when implementing, do a dry-run `pip install cfgrib eccodes` on the project's CI image (and the dev container in `.devcontainer/`) before writing code. If Windows wheels fail, document the fallback.

## Performance estimate

For 4,587 NSW stations, 9.5 years (3,529 days), single run cycle (00Z) per day:

| Phase | Action | Network | Wall-clock (single-machine, modest broadband) |
|---|---|---|---|
| 2017-01 → 2020-09 (1,360 days) | GEFS 1° byte-range subset | 4 leads × ~50 KB × 5 vars = 1 MB/day; total ~1.3 GB | ~30 min download + ~10 min parse |
| 2020-09 → 2021-03 (190 days) | GEFS 0.5° byte-range subset | 4 leads × ~150 KB × 5 vars = 3 MB/day; total ~570 MB | ~10 min |
| 2021-04 → present (1,860 days) | GFS 0.25° byte-range subset | 4 leads × ~490 KB × 5 vars = 10 MB/day; total ~18 GB | ~3-4 hours download + ~30 min parse |
| 2016-09 → 2016-12 (120 days) | Open-Meteo Archive (one-time backfill) | 4,587 calls × ~5 KB = ~25 MB | ~1.5 hours wall-clock at 1s spacing |
| **Total (one-time refetch)** | | **~20 GB raw GRIB cache** | **~5-6 hours** |
| **Per-station parquet output** | After GRIB parse + station-interp | 4,587 × ~30 KB Zstd parquet = ~140 MB | n/a |
| **Incremental daily refresh** | One new day's worth of GFS (0.25°) | 10 MB | ~30 seconds + interpolate to all stations (~2 min) |

Compared to Open-Meteo "ideal" (~150s for ~1,500 stations) this is **2-3 orders of magnitude slower as a one-time backfill**, but: it's wall-clock not call-budget, it's resumable, and (the critical thing) it actually completes without 429s. Incremental daily refresh is bounded by parse time, not network — fast enough for daily-cadence model retraining.

**Disk: ~20 GB raw + ~140 MB processed**, well within laptop budget. Recommend `.gitignore` covers `data/raw/weather_gfs/` and that the raw GRIB cache is treated as fully regenerable (delete-and-refetch is cheap once the pipeline is built).

## Estimated implementation effort

| Session | Scope |
|---|---|
| **1** | `gfs.py` skeleton + URL routing for the 3 path schemes + idx parser + byte-range fetch + cfgrib parse for one variable, one date, one station. Verify against an Open-Meteo Archive call for the same date/coord (should match within ~1°C / few mm). |
| **2** | Full 5-variable pipeline + 4-lead-time aggregation (max/min/sum reducer) + per-station bilinear interpolation. Add per-station parquet output matching the existing schema. Hermetic tests with synthetic GRIB bytes (or a small real cached fixture — these can be ~50 KB each, well within the "no big fixtures" rule). |
| **3** | Router config switch + Makefile target + spec §13.7 revision + 2016-09 one-shot Open-Meteo backfill driver + one-shot end-to-end run + write `weather_code` decision (drop or derive). Includes the wall-clock hours for the full refetch. |

Total: **3 sessions of code + 5-6 hours wall-clock for the one-time backfill** (the wall-clock is unattended and can run overnight).

If `weather_code` derivation turns out to materially affect SHAP, add a half-session for the lookup-table implementation (TCDC + APCP → WMO code). The v1 SHAP analysis suggests it won't.

## Risks and open questions

### R1. GRIB byte-range subset edge cases

`.idx` files reference byte offsets in the source GRIB. Empirically the byte range `offset-(next_offset-1)` returns a valid mini-GRIB with magic and trailing `7777`. **Verify**: that cfgrib opens the byte stream without complaint when given a partial file containing only some records. The pre-flight test 1 (a single TMAX byte-range fetch) returns 491,339 bytes starting with `GRIB`, ending with `7777` — looks well-formed but not yet parsed end-to-end. Plan: in session 1 verify cfgrib opens it cleanly.

### R2. Day-boundary alignment (UTC vs Sydney)

The recommended approach (use f006..f030 from 00Z run on day `t` to fill day `t+1`) covers a UTC day, not a Sydney local day. Mean temperature shifts ~0.1–0.5°C, daily max/min shift ~0.5–1°C in edge cases (heat-wave peaks late afternoon AEST = ~05Z, well within the f006–f030 window from prior 00Z run, OK; but rainfall front passages can fall on the wrong side of the UTC boundary). **Decision needed**: accept UTC-day aggregation as documented, or implement client-side hourly resampling on a wider lead range. Recommend accept — feature block is low-rank in SHAP, complexity not justified.

### R3. weather_code drop vs derive

GFS/GEFS doesn't emit WMO codes natively. Options: (a) drop the column entirely (one fewer feature, simpler), (b) stub null (LightGBM null-handles), (c) derive from APCP+TCDC. v1 SHAP ranks `wx_weather_code` very low. Recommend (b) stub null in the first ship of v2; revisit if A/B comparison shows model degradation tied to this column.

### R4. Wind: max-of-4-instants vs true daily max

GFS instantaneous wind at f006/f012/f018/f024 misses any sub-6h gust peaks. `GUST:surface` available in both buckets is a better proxy for true daily wind max. **Recommendation**: use `GUST:surface` (which is a 6h-ave gust forecast in GFS, instantaneous-at-step in some GEFS variants) and document the slight definitional shift from Open-Meteo's `wind_speed_10m_max`. The downstream feature is a coarse predictor of demand-suppression by storms; either is fine.

### R5. cfgrib + eccodes install on Windows

Pre-built wheels exist on PyPI from eccodes 2.37+. If they're broken on a future Windows Python release, fall back to conda-forge install in the dev container. Action: add an explicit `pip install cfgrib eccodes` smoke test to the dev-container build step.

### R6. NOAA bucket retention

NOAA's AWS Open Data Sponsorship doesn't guarantee indefinite retention. The GFS bucket has been stable since 2021 and the GEFS one since 2017, and NOAA has publicly committed to long-term archival, but if AWS sponsorship ends, the buckets could be deprecated with notice. **Mitigation**: the raw GRIB cache, once built (~20 GB), is portable and could be self-hosted if needed. The fetcher should log a warning if `--force` re-fetch attempts return 404s on previously-cached dates.

### R7. ECMWF Open Data evolution

Re-confirmed in this research: still only 4 days rolling retention. If ECMWF expands to multi-year retention (no announcement at time of writing), it would be a strong replacement for GFS — IFS HRES has better skill scores than GFS at all lead times. **Action**: schedule a re-check in 12 months.

## Multi-horizon extension (un-shelves spec §13.8)

The original v2 weather leakage fix targets only the 1-day horizon (`y_t1`). Spec §13.8 (the shelved 7-day forecast horizon plan) was blocked precisely because Open-Meteo's free Historical Forecast API doesn't expose multi-day lead times pre-2024. **NOAA GFS/GEFS unblocks this entirely** — operational runs publish lead times out to 384h (16 days). For each prediction date `t`, the 00Z run produces:

| File | Forecasts for | Serves horizon |
|---|---|---|
| `gfs.t00z.pgrb2.0p25.f024` | t+1 | day-1 (the v2.0 case) |
| `gfs.t00z.pgrb2.0p25.f048` | t+2 | day-2 |
| `gfs.t00z.pgrb2.0p25.f072` | t+3 | day-3 |
| `gfs.t00z.pgrb2.0p25.f096` | t+4 | day-4 |
| `gfs.t00z.pgrb2.0p25.f120` | t+5 | day-5 |
| `gfs.t00z.pgrb2.0p25.f144` | t+6 | day-6 |
| `gfs.t00z.pgrb2.0p25.f168` | t+7 | day-7 |

For *daily aggregates* at each lead, we need a 6h window of records — e.g., for the day-7 24h aggregate use `f150..f174`. Same pattern as the day-1 case (`f006..f030` from prior 00Z run), just shifted forward by 24h per horizon.

### What this changes for the implementation plan

The 3-session estimate above covers the **1-day horizon only**. Extending to 7 horizons adds **~1 session** of work — the fetcher logic is the same, just looped over a configurable list of horizons:

- `_build_url(date, cycle, lead_h)` already takes `lead_h` as a parameter; just expand the call set from `{6,12,18,24}` to `{6,12,...,174}` (28 files per day instead of 4).
- `_aggregate_day` becomes `_aggregate_day_at_horizon(per_lead_arrays, horizon_days, reducer)` — same reducer, different lead-time slice.
- Output schema grows: instead of 5 `wx_*` columns per row, **5 × 7 = 35 columns** named `wx_temp_max_c_t1`, `wx_temp_max_c_t2`, …, `wx_weather_code_t7`. Or — better — keep the parquet long-format with a `horizon` column (`(station_id, date, horizon, wx_*)`), and let the feature builder pivot to wide.

### Cost implications

| | Day-1 only (v2.0) | 7-day (v2.1 unblocked) |
|---|---|---|
| Files per day per run | 4 (f006, f012, f018, f024) | 28 (f006 → f174) |
| Network per day (GFS 0.25°, byte-range subset, NSW box) | ~10 MB | ~70 MB |
| One-time backfill (1,860 days × 7 horizons) | ~18 GB | ~130 GB |
| Wall-clock for one-time backfill | 5-6 hours | ~10-14 hours |
| Disk for parsed parquet | ~140 MB (4,587 stations × 1 horizon × 30 KB) | **~140 MB total** with grid-cell caching (see next section) |
| Incremental daily refresh | ~30s + 2 min interp | ~3 min + 5 min interp |

The disk savings on the parsed side come from grid-cell caching (next section), not horizon-aware tricks. **The 1-day pipeline is a strict subset of the 7-day pipeline**; building it 7-day-ready from day one costs ~1 extra session but eliminates a refactor when v2.1 lands. **Strong recommendation: bundle Phase 2 with Phase 1** (the user has now agreed to this — see "Implementation sequencing below").

### v2.1 modelling work (still queued)

The 7-day horizon also requires the modelling-side work documented in [`docs/research/2026-05_7day_forecast_horizon.md`](2026-05_7day_forecast_horizon.md) — Architecture A (one LightGBM per horizon), per-horizon evaluation, notebook updates. That's the 5-8 session estimate from that doc, **unchanged** by which data source provides the values. Sequencing recommendation: ship v2.0 (this doc) first with 7-day data path baked in; revisit v2.1 modelling separately.

## Grid-cell caching architecture (spatial granularity smartness)

**The naive port of the Open-Meteo pattern is wrong for grid-based data.** Open-Meteo's fetcher writes one parquet per station_id because each station has a unique lat/lon and the API call carries that into the result. GFS/GEFS is fundamentally different: forecasts are on a regular grid, and **many stations resolve to the same grid cell**. Caching at the station level would duplicate data and waste both bandwidth and disk.

### The duplication problem

GFS 0.25° = 0.25° × 0.25° cells = **~27.8 km × ~23.3 km** at NSW latitudes (cos(33°) × 27.8). A 0.25° cell covers ~650 km². Implications for the 4,587-station NSW roster:

- **Sydney metro** (~5,000 km²) has hundreds of stations clustered into roughly 10-15 grid cells
- **Newcastle / Wollongong / Central Coast** add ~5-10 cells with 50+ stations each
- **Regional NSW** has many cells with 1-2 stations
- **Empirical estimate: ~400-600 unique grid cells cover the entire NSW roster**, vs 4,587 stations

So the actual *unique GFS data* the model uses is ~10-15× smaller than the station count.

GEFS 1° = ~111 km × ~93 km cells = even coarser. Probably ~50-100 unique cells cover all NSW.

### Three-layer caching design

**Layer 1: Grid-cell mapping (computed once)**

`data/interim/station_grid_mapping.parquet` — schema:

```
station_id : str
lat        : float    # original
lon        : float    # original
gfs_lat_idx, gfs_lon_idx : int    # nearest grid point in GFS 0.25°
gefs_lat_idx, gefs_lon_idx : int  # nearest grid point in GEFS 1°
gefs05_lat_idx, gefs05_lon_idx : int  # nearest grid point in GEFS 0.5°
# Optionally for bilinear interp: the 4 surrounding grid points + weights
```

Computed once at pipeline init by `src/fuel_pred/spatial/gfs_grid.py` (new module). Reads `data/interim/stations.parquet`, computes nearest-neighbour (or bilinear surround) for each grid resolution, writes the mapping. Re-runs only if `stations.parquet` changes.

**Layer 2: Per-(date, horizon) grid parquet** — the actual fetch output

`data/raw/weather_gfs/<YYYY-MM-DD>/<HH>z_f<lead>.parquet` — schema:

```
gfs_lat_idx, gfs_lon_idx : int       # grid point identifier
wx_temp_max_c, wx_temp_min_c, ...    # the 5 (or 4, if weather_code dropped) values
```

One file per `(date, cycle, lead_h)` triple. Each file contains ~600 unique grid points covering the NSW bounding box (lat ∈ [-37.5, -28], lon ∈ [140.5, 154]). File size: ~20-50 KB per parquet (much smaller than per-station). Total files: 3,529 days × 7 horizons × 1 cycle = 24,703 files. Total disk: ~700 MB-1.2 GB. Way under the naïve 140 MB-per-horizon-per-station-set × 7 horizons ≈ 1 GB anyway, but with much better deduplication properties.

**Layer 3: Feature-build join** — at make_features time

`add_weather_features()` is rewritten to:

1. Load `station_grid_mapping.parquet`
2. For each (panel date, horizon) requested:
   - Load the corresponding grid parquet
   - Join panel rows (station_id, date) → grid cells via the mapping
   - Produce per-station-date-horizon rows
3. Pivot horizon dimension into wide `wx_*_tN` columns
4. Apply the 1-day shift (still — semantics unchanged)

No per-station parquet at all. The Open-Meteo pattern's `data/raw/weather/<station_id>.parquet` becomes legacy-only (for the optional paid-tier path).

### Bandwidth and disk savings

| | Naïve per-station | Grid-cell |
|---|---|---|
| Unique fetches per (date, horizon) | 4,587 | ~600 |
| Network per day (7 horizons, GFS 0.25° byte-range) | ~70 MB × deduplication-noop = ~70 MB | ~70 MB (same — file granularity is `(date, lead)`, not per-station) |
| Parsed parquet count | 4,587 × 7 horizons = 32,109 | ~24,703 (date × horizon, all stations packed inside) |
| Parsed parquet disk | ~140 MB × 7 = ~1 GB | ~700 MB-1.2 GB |
| Feature-build read pattern | 32,109 small files | 24,703 small files joined to a mapping |
| Re-doing for a new station | Refetch from API (~7s/station) | Just add a row to mapping; existing parquets work |

The **biggest win is operational**: adding new stations doesn't require any API calls — just regenerate the mapping. That's a permanent improvement over the Open-Meteo pattern. Bandwidth is mostly the same because the unit of network call is the GRIB byte-range subset (per-date-per-lead), independent of how many stations resolve to that grid.

### Interpolation: nearest-neighbour or bilinear?

GFS 0.25° cells are ~25 km. Within a single Sydney cell, two stations could be ~25 km apart — meaningful weather difference (Bondi vs Penrith). **Bilinear interpolation** between the 4 surrounding grid points fixes this: each station gets a value computed from its location within the cell, not the cell centre.

The mapping table grows to include 4 grid points and 4 weights per station. The feature-build join becomes: `value(s,d,h) = Σ_i weight_i × grid_value(d, h, lat_idx_i, lon_idx_i)`. Slightly more code but no measurable cost at runtime — the weights are pre-computed.

**Recommendation: bilinear**, pre-computed in the mapping table. Nearest-neighbour stays as an option (flag) for verification / debugging.

### What about GEFS resolution boundaries?

When a date falls in the GEFS 1° window (2017-2020), the mapping uses `gefs_lat_idx, gefs_lon_idx`. Different stations may collide more aggressively at 1°. That's fine — the lower resolution genuinely is less informative, and we accept that for the early-history window.

The router (which source for which date) selects the appropriate mapping columns automatically. No source-mixing within a single (date, station) row.

## Implementation sequencing (post-decision)

User has agreed (2026-05-27) to **bundle Phase 1 + Phase 2** (1-day + multi-horizon) for the data pipeline. The 4-session plan:

| Session | Scope |
|---|---|
| **1** | `gfs.py` skeleton + URL routing (3 path schemes) + `.idx` parser + byte-range fetch + cfgrib parse for one variable, one date, one station. Verify against an Open-Meteo Archive call for the same date/coord. |
| **2** | Full 5-variable / multi-horizon (f006..f174) aggregation. Grid-cell caching: `spatial/gfs_grid.py` computes the station→grid mapping; `gfs.py` writes per-(date, lead) grid parquets. Hermetic tests with small synthetic GRIB fixtures. |
| **3** | `make_features.py` `add_weather_features()` rewritten to join via the mapping. Pivot horizon to wide. Wire the 1-day shift on the horizon=1 column (the others are already lead-time-shifted by construction — see "Day boundary alignment" above). Update tests. |
| **4** | Router config switch (`WEATHER_SOURCE=gfs/openmeteo/auto`) + Makefile target + spec §13.7 + §13.8 revision + 2016-09 one-shot Open-Meteo Archive backfill driver + one-shot end-to-end run + write `weather_code` decision. |

Total: **4 sessions of code + ~10-14 hours wall-clock for the one-time backfill** (unattended).

After this lands, v2.1 (the 7-day modelling work in [`2026-05_7day_forecast_horizon.md`](2026-05_7day_forecast_horizon.md)) becomes a clean 5-8 session follow-up with no data-pipeline blocker.

## See also

- [`docs/research/2026-05_weather_leakage_fix.md`](2026-05_weather_leakage_fix.md) — the original v2 plan; this doc replaces the source choice from "Open-Meteo HFA" to "GFS+GEFS hybrid" but keeps the 1-day shift mechanics intact
- [`docs/research/2026-05_weather_leakage_preflight.md`](2026-05_weather_leakage_preflight.md) — burst-test results
- `src/fuel_pred/fetch/weather.py` — Open-Meteo fetcher, retained as paid-tier path
- `src/fuel_pred/build/make_features.py` — `add_weather_features()`, unchanged (still does 1-day shift)
- NOAA GFS on AWS Open Data: https://registry.opendata.aws/noaa-gfs-bdp-pds/
- NOAA GEFS on AWS Open Data: https://registry.opendata.aws/noaa-gefs/
- NOAA GEFS v12 reforecast: https://registry.opendata.aws/noaa-gefs-reforecast/
- cfgrib documentation: https://github.com/ecmwf/cfgrib
- ECMWF Open Data (retention reference): https://www.ecmwf.int/en/forecasts/datasets/open-data
