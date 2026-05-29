# Weather leakage fix v2.0 — outcome

**Date:** 2026-05-29
**Source docs:**
- Plan: [`2026-05_weather_leakage_fix.md`](2026-05_weather_leakage_fix.md)
- Pre-flight: [`2026-05_weather_leakage_preflight.md`](2026-05_weather_leakage_preflight.md)
- Architectural pivot (Open-Meteo → NOAA GFS): [`2026-06_nwp_archive_alternative.md`](2026-06_nwp_archive_alternative.md)
- Multi-day forecast research (set the v2.1 7-day path): linked from the plan doc
**Spec ref:** spec.md §13.7
**Status:** ✅ LANDED

## TL;DR

The v2.0 leakage fix landed cleanly. NOAA GFS day-ahead forecast values replaced the v1 ERA5 reanalysis actuals in the `wx_*` weather block. Absolute MAE rose modestly as predicted (~0.07-0.15 c/L) — the leakage tax. The SA2 A-vs-B comparison held up: Δ MAE on `test_normal` essentially unchanged (−0.391 → −0.353), and **Δ MAE on `test_crisis` more than doubled** (−0.183 → −0.398) once the model could no longer cheat with ERA5 actuals on the OOD 2026 fold. v1's documented caveat about the crisis-fold lift being "smaller and noisier" is gone; v1's crisis-fold RMSE regression (Model B worse than A) is also gone.

## What changed

| | v1 baseline (ERA5 leaky) | v2.0 (GFS day-ahead forecast) |
|---|---|---|
| Weather data source | Open-Meteo Archive API (ERA5 reanalysis actuals) | NOAA GFS/GEFS via anonymous AWS S3 byte-range subsetting |
| Storage | Per-station parquet (`data/raw/weather/<station_id>.parquet`) | Per-(date, horizon) grid parquet (`data/raw/weather_gfs/<YYYY-MM-DD>_h<N>.parquet`) |
| Stations / cells | 4,587 per-station files | 346 unique GFS 0.25° grid cells, joined to stations at feature-build time via bilinear interp weights |
| Join semantics | Panel row at `t` joined to wx values for date `t` (LEAKAGE — ERA5 actuals not available on day `t`) | Panel row at `t` joined to GFS file dated `t` (the day-ahead forecast issued on `t`, valid on `t+1` — what a deployed forecaster has) |
| Model column lists | `MODEL_A_BLOCKS` / `MODEL_B_BLOCKS` with `wx` block | `MODEL_A_GFS_BLOCKS` / `MODEL_B_GFS_BLOCKS` with `wx_gfs` block (selected by `config.resolve_weather_source()`) |
| Schema | 5 `wx_*` cols | 5 `wx_*_t1` cols (v2.0); 30 wider `wx_*_t2..t7` materialised but excluded from v2.0 model — v2.1 readiness only |
| Rate limits | Open-Meteo free tier 600/min / 5k/hr / 10k/day — empirically unworkable at scale (2026-05-26/27 incidents) | NOAA AWS S3 anonymous reads — no rate limits |

Open-Meteo path remains intact at `src/fuel_pred/fetch/weather.py` as the optional paid-tier upgrade. Set `WEATHER_SOURCE=openmeteo` + `OPENMETEO_API_KEY=...` to use it. `WEATHER_SOURCE=auto` picks Open-Meteo if a key is set, else GFS — so new contributors default to the strict-free path with zero config.

## v1 → v2 headline comparison

All metrics in cents/L. Negative `Δ MAE (B vs A)` = Model B beats Model A (SA2 adds value).

### test_normal (n=849,334)

| | MAE A | MAE B | Δ MAE (B vs A) | RMSE A | RMSE B |
|---|---:|---:|---:|---:|---:|
| **v1 (ERA5 leaky)** | 6.303 | 5.912 | **−0.391** | 10.973 | 10.338 |
| **v2.0 (GFS forecast)** | 6.373 | 6.020 | **−0.353** | 10.953 | 10.589 |
| Δ v2 − v1 | +0.070 | +0.108 | +0.038 | −0.020 | +0.251 |

### test_crisis (n=172,858, OOD 2026)

| | MAE A | MAE B | Δ MAE (B vs A) | RMSE A | RMSE B |
|---|---:|---:|---:|---:|---:|
| **v1 (ERA5 leaky)** | 13.466 | 13.283 | −0.183 | 18.628 | 18.739 |
| **v2.0 (GFS forecast)** | 13.616 | 13.218 | **−0.398** | 19.054 | 18.578 |
| Δ v2 − v1 | +0.150 | −0.065 | **−0.215** | +0.426 | −0.161 |

## Reading the comparison

### 1. The leakage tax is real and small

Absolute MAE rose **+0.07 c/L (Model A test_normal)** to **+0.15 c/L (Model A test_crisis)** — well within the **0.05-0.15 c/L** range predicted in the plan doc. The model can no longer cheat by using ERA5 actuals (perfect post-hoc truth) as a proxy for forecast values; it now sees the actual forecast error a deployed predictor would face.

### 2. Test_normal SA2 lift is essentially unchanged

Δ MAE on test_normal moved from −0.391 to −0.353. The 0.04 c/L reduction is small and explainable: with the weather block weakened (less informative because it's a noisier forecast rather than perfect truth), the model leans more on lag and upstream features, leaving slightly less marginal headroom for the SA2 block to fill. The fundamental finding — SA2 demographics measurably improve next-day price prediction — holds.

### 3. The crisis-fold story flipped completely

The biggest change. v1 had two documented crisis-fold caveats in `results/README.md`:

> "The crisis-fold lift is real but smaller and noisier. test_crisis shows Δ MAE −0.183 — a genuine win, but Model B's RMSE is marginally *worse* than Model A's (18.739 vs 18.628), and two brands (Metro, Independent) regress slightly. OOD generalization across a price regime the model never trained on is inherently noisy; treat the crisis fold as supportive, not headline."

**v2.0 invalidates both halves of that caveat:**

- **Δ MAE on test_crisis more than doubled**: −0.183 → −0.398. The SA2 lift is no longer "smaller and noisier" on OOD data; it's **bigger and more robust** than the in-distribution lift.
- **RMSE flipped**: Model B's crisis RMSE is now **lower** than A's (18.578 vs 19.054). The v1 RMSE regression is gone.

The most plausible mechanism: v1's ERA5 weather block was an unrealistically strong predictor in-distribution, masking the SA2 block's true marginal value. When v1 saw OOD 2026 prices, the weather features over-fit and helped less; the SA2 block didn't help as much either because it had been competing with an artificially-strong weather block during training. v2's honest weather block makes the SA2 block's contribution clearer at training time, and that translates to a more robust improvement on OOD data.

### 4. SEIFA quintile + brand patterns are unchanged

The structural shape of the SA2 lift holds:

- **SEIFA quintile (test_normal)**: Q1 −0.248 → Q5 −0.532 in v2 (vs Q1 −0.250 → Q5 −0.678 in v1). Same monotonic pattern of "lift scales with affluence" — Q5 strongest.
- **Brand**: All 7 reported brands benefit from the SA2 block in v2 (range −0.252 to −0.437). v1 had one brand (Speedway, not in v2 top-7 by volume) at −0.110.

The brand-level deltas are smaller in v2 across the board, consistent with the test_normal headline. No brand-level regression — every brand sees a meaningful SA2 lift.

### 5. Model B' (venue ablation) reproduces the null

The venue + long-weekend block from `spec §13.6 Phase 1` still adds no marginal lift over Model B:

- v1: B' lost vs B by **+0.681 c/L** on test_normal
- v2: B' lost vs B by **+0.554 c/L** on test_normal

Both runs identify the venue-distance feature set as net-negative. Independent confirmation under a different (less leaky) weather block.

## Methodological compromises (documented in spec §13.7)

1. **2016-09 → 2016-12 wx_*_t1 nulls** (~2.2% of training rows, train fold only). GFS S3 starts 2017-01-01. Per spec, accepted as null-stub for v2.0; revisit if SHAP analysis shows the 2016 nulls matter.
2. **NOAA archive sporadic 404s** — out of 3,403 attempted post-2017 fetches, 374 (~11%) returned 404 (NOAA's archive has occasional missing files, particularly in early GEFS years). Affected dates have null `wx_*_t1`. Combined with the 2016 gap, ~20% of all panel rows have null `wx_*_t1`. LightGBM handles natively; not a model correctness issue, but absolute MAE figures incorporate this null-handling.
3. **`wx_weather_code_t1` is null-stubbed** — GFS/GEFS doesn't emit WMO weather codes. v1 SHAP ranked this column low, so dropping it costs <0.01 c/L MAE per the plan.
4. **UTC vs Sydney day boundary** for the daily aggregate. GFS emits in UTC; our daily aggregation uses leads `f006..f030` from the 00Z run on day `t` to cover day `t+1`. Documented in the plan doc as acceptable noise.
5. **`wx_*_t2..t7` columns are mostly null** in `features.parquet` (only ~4,000 spurious rows from the smoke-test fixture). The model doesn't consume these for v2.0 (per `WX_COLUMNS_GFS_T1` — only `_t1` is in the model block). They exist in the schema for v2.1 readiness; when v2.1 lands, those columns will need to be backfilled via a separate `fetch-weather-gfs HORIZONS=2,3,4,5,6,7` run.
6. **Spurious wx_*_t2..t7 correlations in `results/comparison.md`**: the report computes correlations over the tiny ~4,000-row sample where those columns are populated. With such a small sample, the correlations are noise. Suppression for the next iteration of `evaluate.compare`.

## What it took (wall-clock)

| Phase | Wall-clock |
|---|---|
| GFS fetch (3,529 units, 8 workers, h=1 only) | **~17 hours** (started 2026-05-28 18:03 AEST, completed 2026-05-29 11:07 AEST) |
| Features regen | 8 min |
| Train (A, B, B') | 3 min 15 sec |
| Evaluate (`compare.py`) | ~15 sec |
| **Total** | **~17.5 hours** (dominated by the one-time fetch) |

Code work was the 4 implementation Sessions (1+2 fetcher, 3 feature-build join, 4a config/Makefile/docs) over 2026-05-27/28, plus the architectural pivot from Open-Meteo to NOAA GFS after the rate-limit lessons. See spec §13.7 for the full commit chain.

Incremental future runs (one new day added each morning) need ~30s of network + ~2 min of parse — fast enough for daily-cadence model retraining without re-running the historical backfill.

## What v2.0 ships

| File | Status |
|---|---|
| `data/raw/weather_gfs/<YYYY-MM-DD>_h1.parquet` | 3,029 files, ~2,145 NSW grid cells each (gitignored — backed up to OneDrive) |
| `data/interim/station_grid_mapping.parquet` | 4,587 stations × bilinear weights for 3 GFS/GEFS resolutions |
| `data/processed/features.parquet` | 14,993,062 rows × 132 cols, includes 35 `wx_*_tN` cols (5 populated, 30 null-stubbed) |
| `models/model_a.pkl`, `model_b.pkl`, `model_b_prime.pkl` | Retrained 2026-05-29 12:42 |
| `results/comparison.md` | Regenerated 2026-05-29 12:43, 296 lines, segmented A-vs-B-vs-B' tables |

## What v2.0 does NOT ship

- Multi-horizon weather features for v2.1 (`wx_*_t2..t7` — the data path is built but the fetch is deferred per the h=1 sampling decision; see spec §13.8 for v2.1 modelling plan)
- Updated notebooks — `notebooks/03_explainability.ipynb` references v1 features in its narrative; should be re-executed against v2 artefacts to refresh SHAP plots and case-study panels (~30 min wall-clock for the full re-run)
- 2016 Open-Meteo Archive backfill — deferred per the null-stub decision

## See also

- [`spec.md` §13.7](../../spec.md) — the LANDED entry pointing here
- [`results/README.md`](../../results/README.md) — updated headline + caveat #4 to reflect v2 baseline
- [`results/comparison.md`](../../results/comparison.md) — full segmented v2.0 metrics
- [`2026-06_nwp_archive_alternative.md`](2026-06_nwp_archive_alternative.md) — the architecture choice (NOAA GFS/GEFS vs Open-Meteo)
- spec §13.9 — backlog item for self-hosted Open-Meteo (long-term alternative path)
- spec §13.8 — v2.1 7-day forecast horizon, modelling work still queued
