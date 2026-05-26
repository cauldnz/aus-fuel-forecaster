# Weather leakage fix v2.0 — resume-tomorrow plan

**Date:** 2026-05-26 evening
**Status:** Open-Meteo daily quota exhausted; pipeline paused.
**Next action:** Resume tomorrow ≥10:30 AEST (UTC midnight + buffer).

## Why this doc exists

We attempted the v2 weather re-fetch three times today (hybrid, then forecast-only with 4 workers, then forecast-only with 8 workers). All three runs hit Open-Meteo's daily quota cap, and the third attempt verified the API now returns:

```
HTTP 429: {"reason":"Daily API request limit exceeded. Please try again tomorrow."}
```

The daily counter resets at UTC midnight, which is 10:00 AEST. Add a buffer and resume at ~10:30 AEST.

## Lessons learned (encoded into the code, ready for tomorrow)

Three defensive changes landed this evening so the resume run can't repeat today's mistakes:

### 1. 429s no longer retry via tenacity

Previously, a 429 response triggered the `@retry` decorator's exponential backoff (5 attempts × 2-30s waits). Each retry was *another HTTP request* that itself counted against the quota. A single 429 became 5 wasted requests. With ~1000 failures × 5 retries each = ~5000 wasted requests, easily explaining how 4587 stations × 1 call each (~4587 requests) burned through 10,000+ requests.

**Fix:** new `OpenMeteoRateLimitError` raised on 429. The retry predicate (`_is_retryable`) excludes it. Surfaces immediately to the orchestrator without any retries.

### 2. Circuit breaker in the parallel orchestrator

`tools/parallel_weather_fetch.py` now tracks consecutive 429 responses. After **3 consecutive 429s** it cancels all pending futures and shuts down the pool. This preserves whatever quota remains for a later resume.

Default `--rate-limit-circuit 3`; tune via CLI. Set to 0 to disable (not recommended).

### 3. Cache-resume support already existed

The `_cache_covers()` check in `fetch_one` skips any station whose cached parquet covers the requested date range. So when we resume tomorrow, all the stations from any previous successful fetch are skipped automatically — we only fetch what's missing.

## Tomorrow's resume command

Single line, ready to copy-paste at ~10:30 AEST:

```powershell
Set-Location C:\repos\cauldnz\aus-fuel-forecaster
git checkout claude/weather-leakage-fix-v2  # ensure we're on the right branch
uv run python tools/parallel_weather_fetch.py `
  --stations data/interim/stations.parquet `
  --start 2016-09-01 --end 2026-04-30 `
  --out data/raw/weather `
  --workers 4 `
  --forecast-only `
  --rate-limit-circuit 3 `
  --progress-every 100
```

**Notes on the parameters:**

- `--forecast-only` — skips the 2016 ERA5 fallback entirely. Methodologically a "null fill" for Sept-Dec 2016 (~2.2% of training rows in the train fold only); LightGBM handles the nulls.
- `--workers 4` — *deliberately lower* than today's 8. At 4 workers × ~7s/call, we average ~34 calls/min, well under the 600/min cap with room for burst variance. 8 workers occasionally tripped the per-minute cap; 4 stays comfortable.
- `--rate-limit-circuit 3` — if 3 consecutive 429s happen, abort and preserve quota. This will trigger if (a) we somehow hit the daily cap again, or (b) Open-Meteo throttles us for some other reason.

**Expected wall-clock:** 4587 stations × 7s ÷ 4 workers = **~135 minutes** (~2.25 hours).

**Expected API budget:** 4587 requests against a 10,000/day cap = 46% of daily budget. Plenty of headroom.

## After the fetch completes

The wakeup pattern from earlier is already in place. When the fetcher finishes successfully:

```powershell
# Final OneDrive snapshot of the weather data
Copy-Item C:\repos\cauldnz\aus-fuel-forecaster\data\raw\weather\*.parquet `
  C:\Users\chrisauld\OneDrive\fuel-pred-backups\weather_v2_2026-05-27_forecast_only\ -Force

# Regenerate features.parquet with the leakage-corrected wx_* values
uv run python -m fuel_pred.build.make_features

# Retrain Models A, B, B' against the new feature matrix
uv run python -m fuel_pred.train.train_models

# Regenerate results/comparison.md
uv run python -m fuel_pred.evaluate.compare

# Re-execute the 3 notebooks (their cached outputs are stripped by nbstripout
# so this only validates they still run end-to-end)
foreach ($nb in @('01_eda', '02_modeling', '03_explainability')) {
    uv run python -m jupyter nbconvert --to notebook --execute --inplace `
      --ExecutePreprocessor.timeout=600 `
      (Resolve-Path "notebooks/$nb.ipynb").Path
}
```

Total Phase C+D wall-clock after the fetch: ~10-15 minutes.

## Outcome doc to write at the end

Once everything's regenerated, write `docs/research/2026-05_weather_leakage_fix_outcome.md` following the template in the agent prompt from earlier this evening (Section D.1 of the original build agent's prompt). Headline metrics needed:

- v1 (ERA5 leaky) vs v2 (forecast-only) absolute MAE on test_normal and test_crisis, Models A and B
- v1 Δ MAE (B vs A) vs v2 Δ MAE (B vs A) — should be ~unchanged per the leakage-fix theory
- Whether the absolute rise was within the 0.05-0.15 c/L prediction
- Weather feature importance shifts in Model B's gain ranks

Then update:
- `spec.md` §13.7: status "READY TO BUILD" → "LANDED" with the actual figures
- `results/README.md` caveat #4: "v1 known compromise" → "v2.0 corrected; see outcome doc"
- `results/README.md` headline table: regenerate with v2 figures, footnote linking to v1→v2 comparison

## Safety net for the resume

If the resume hits 429s again at 10:30 AEST (suggesting the daily counter hasn't actually reset or our IP is still flagged):

1. Don't keep retrying. Each attempt burns quota.
2. Wait until early evening (after UTC midnight again) and try once more.
3. If still blocked, pivot to "Option 4" from the research doc — narrow training to 2017+ and skip the weather block entirely for now (or use the existing v1 ERA5 cache if it can be restored from anywhere).

## What we have committed and safe

- `claude/weather-leakage-fix-v2` branch with all v2 code:
  - `9a9e784` — hybrid fetcher + 1-day join shift
  - `2cf0bcb` — `--forecast-only` flag + `.env` / `python-dotenv` infrastructure
  - (TBC tonight) — 429-no-retry + circuit breaker
- `OneDrive\fuel-pred-backups\weather_v2_2026-05-26_partial\` — 27 v1-style ERA5 parquets from this morning's run (mislabelled but harmless; can be deleted or used as ERA5 reference)
- 333 tests passing on the v2 branch
- This doc

Nothing is lost.
