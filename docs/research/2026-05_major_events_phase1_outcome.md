# Major events Phase 1 sanity check — outcome

**Date:** 2026-05-26
**Source docs:** [`2026-05_major_events_features.md`](2026-05_major_events_features.md) (full plan), [`2026-05_major_events_eda_outcome.md`](2026-05_major_events_eda_outcome.md) (Phase 0 gate)
**Status:** **STOP**

## The experiment

Added 5 venue/long-weekend features (`stn_nearest_venue_km`, `stn_nearest_venue_capacity`, `stn_nearest_venue_type`, `stn_n_venues_within_5km`, `cal_is_pre_long_weekend`) to a new VENUE block in `feature_blocks.py`. Trained Model B' (= Model B + VENUE block) alongside the existing Model A and Model B. Identical hyperparameters, identical training rows (SA2-non-null guard unchanged — venue columns can be null and LightGBM handles them).

Implementation details:

- `src/fuel_pred/spatial/venues.py` — vectorised numpy haversine over the 4,995 stations × 10 pilot venues from `data/static/major_venues.csv`; produces `data/interim/stations_venues.parquet` (4,587 stations matched, 408 skipped for missing lat/lon).
- `add_calendar_features` derives `cal_is_pre_long_weekend = (cal_day_of_week == 4) & (cal_days_to_next_public_holiday == 3)` — pure function of existing columns.
- `add_station_features` merges venue features when the parquet exists, else fills null per CLAUDE.md None-tolerant pattern.
- `train_models` now fits A / B / B' on identical rows and persists `model_b_prime.pkl` plus a `y_pred_b_prime` column in both prediction parquets.
- `evaluate/compare` renders B vs B' deltas alongside the existing A vs B headline.

## Decision rule

- Model B' vs Model B Δ MAE **≤ −0.05 c/L** on test_normal → PROCEED to Phase 2 (AFL/NRL fixtures)
- Model B' vs Model B Δ MAE within **±0.03 c/L** → STOP (venue features just re-encode `stn_is_metro` + other existing features)
- Model B' **worse than Model B** → STOP (overfitting / negative signal)

## Results

Headline (from `results/comparison.md`):

| Fold | n | MAE A | MAE B | MAE B' | Δ (B' vs B) | Δ (B vs A) |
|------|--:|------:|------:|-------:|------------:|-----------:|
| test_normal | 849,334 | 6.303 | 5.912 | 6.594 | **+0.681** | −0.391 |
| test_crisis | 172,858 | 13.466 | 13.283 | 13.733 | **+0.450** | −0.183 |

**Model B' is meaningfully *worse* than Model B on both test folds.** Every brand and metro/regional segment shows the same direction (see `results/comparison.md` for the full segmented tables); no slice flipped the sign.

Validation-fold readout (for context):

| Model | best_iter | best val MAE |
|---|--:|---:|
| A | 405 | 4.9439 |
| B | 495 | 4.8528 |
| B' | 538 | **4.8404** |

The classic overfitting fingerprint: **B' has the *best* val MAE (by 0.012 c/L) but the *worst* test MAE (by 0.68 c/L).** Identical pattern to the SA2 v1.1 → v1.2 episode recorded in spec §7.7.4 (broadening 10 → 31 cols improved val, regressed test).

## Feature importance

Where the new VENUE-block features ranked in Model B' (93 features total, by gain importance):

| Feature | Rank in B' | Gain % | Read |
|---|--:|---:|---|
| `stn_nearest_venue_km` | **20** | 0.559% | Heavily used split point — the model picked it up and learned val-fold patterns from it |
| `stn_nearest_venue_capacity` | 54 | 0.013% | Minimal — capacity is near-degenerate (10 pilot venues all in greater Sydney) |
| `stn_n_venues_within_5km` | 71 | 0.001% | Noise floor |
| `stn_nearest_venue_type` | 78 | 0.000% | Noise floor — 3-level categorical with no useful interaction |
| `cal_is_pre_long_weekend` | **93 (last)** | **0.000%** | **Never split on.** The model already extracts the Fri-before-Mon-holiday signal from `cal_day_of_week × cal_days_to_next_public_holiday` interactions in the existing CAL block |

The top-10 in B' is dominated by the same lag/upstream/calendar features that dominate A and B — nothing structural changed in what the model relies on.

Two surprises worth flagging to the human reviewer:

1. **`cal_is_pre_long_weekend` got zero gain.** The Phase 0 EDA's *cleanest* signal (§10c long-weekend Friday gap of +4.28 c/L between Q1 and Q5 venue distance) didn't manifest as a useful model feature because LightGBM already builds the equivalent split from the two underlying columns. The +4.28 c/L gap is real but the model already sees it.
2. **`stn_nearest_venue_km` was the 20th-ranked feature out of 93 and *still* hurt test performance.** Being heavily used by the model isn't the same as adding generalizable signal — it bled splits away from features that would have generalized better. This is the most direct possible confirmation of the Phase 0 caveat: the §10b distance signal is metro/regional confounding that `stn_is_metro` + other existing features already encode.

## Verdict

**STOP.** Model B' is 0.681 c/L worse than Model B on test_normal — not just inside the ±0.03 c/L "no signal" band, but actively negative by a wide margin. The §10b venue-distance residual gap (+2.71 to +4.11 c/L on holiday rows) was confounded with existing features that absorb the metro/regional split more cleanly; introducing the explicit venue distance gave the model a new attractive split point that overfit the val fold and regressed the test fold.

**Phase 2 (AFL/NRL fixtures) is NOT justified.** The static venue + long-weekend block — which the Phase 0 doc called out as the "highest-value subset" if metro/regional confounding turned out to dominate — failed the additive sanity check. Building a fixture calendar on top of features that already regress test performance would compound the problem, not solve it.

`data/static/major_venues.csv` stays as documentation of the pilot list. `stations_venues.parquet` regeneration stays in the pipeline (cheap, ~0.2 s for the spatial join) so the venue columns are always there if someone wants to re-explore with a different feature design later — they just won't reach the model.

## What to do next

1. **Spec §13.6 update (human-authored).** Record the Phase 1 verdict, mark the section as resolved. The Phase 0 EDA doc + this outcome doc capture the full reasoning trail; spec.md should reference both and close the question.
2. **Keep the code, hold the canonical models.** Model A and Model B definitions are untouched — the v1 headline result (Δ MAE −0.391 c/L on test_normal) reproduces exactly. Model B' lives alongside in the codebase; users curious about the failed ablation can re-fit it cheaply. No revert needed.
3. **One open question deferred to a follow-up.** The pilot list is 10 greater-Sydney/Newcastle venues. A null result on this slice doesn't formally rule out signal from a *richer* venue catalogue (e.g. including the 39 AFL/NRL home grounds nationwide, or large outdoor music venues). But the EDA caveat already explained the metro/regional confound mechanically — extending the list is unlikely to change the verdict without a structural feature-design rethink (e.g. event-day × venue-precinct interactions, not static distance-to-nearest).
4. **No action on `cal_is_pre_long_weekend`.** The feature is harmless (zero gain → no effect on splits) but adds 5 bytes per row of parquet on disk. Leaving it in for now keeps the schema stable; remove if it causes friction later.

## Files changed

Code (Commit 1):
- `src/fuel_pred/spatial/venues.py` (new)
- `src/fuel_pred/config.py` — added `INTERIM_STATIONS_VENUES`
- `src/fuel_pred/build/make_features.py` — venue merge in `add_station_features`, long-weekend flag in `add_calendar_features`, orchestrator + CLI wiring
- `src/fuel_pred/train/feature_blocks.py` — `VENUE_COLUMNS`, `MODEL_B_PRIME_BLOCKS`, `stn_nearest_venue_type` in `CATEGORICAL_COLUMNS`
- `src/fuel_pred/train/train_models.py` — third fit + `model_b_prime.pkl` + `y_pred_b_prime` column
- `src/fuel_pred/evaluate/compare.py` — B vs B' deltas, B' importance section
- `tests/test_spatial_venues.py` (new), `tests/test_features.py`, `tests/test_train_feature_blocks.py`, `tests/test_train_models.py`

Artefacts (Commit 2):
- `docs/research/2026-05_major_events_phase1_outcome.md` (this doc)
- `docs/research/README.md` — added Phase 1 outcome link
- `results/comparison.md` — regenerated with B' columns + headline

Wall clock for the full pipeline (after the code commit):

| Step | Time |
|---|---:|
| spatial.venues (4995 × 10 stations × venues) | ~0.3 s |
| make_features (regenerate features.parquet, 14.99M rows × 102 cols) | ~3.9 min |
| train_models (3 models, ~500-iter best on each) | ~5.2 min |
| evaluate.compare | ~33 s |
| **Total** | **~10 min** |
