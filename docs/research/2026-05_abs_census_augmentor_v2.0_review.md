# abs-census-augmentor: the v1.5 → v2.0.0 upgrade

**Date:** 2026-05
**Augmentor versions reviewed:** v1.5 (current pin, commit `8fe6fa55`) and v2.0.0 (released 2026-05-27, commit `887ec011`)
**Project pin:** `abs-census-augmentor @ git+https://github.com/cauldnz/abs-census-augmentor.git@8fe6fa55247def1e5bc1be81a2cd33775680ac3e` ([`pyproject.toml`](../../pyproject.toml) line 45)
**TL;DR:** v2.0.0 is the biggest release since we integrated. The headline is **cross-edition temporal augmentation** — the thing that was missing in v1.5 and the reason we deferred temporal-DSS in [§7.7.2](../../spec.md). Plus two pieces of low-risk static surface area: **ERP age/sex columns** ([PR #82](https://github.com/cauldnz/abs-census-augmentor/pull/82)) and **three cross-dataset PRESETs** ([PR #86](https://github.com/cauldnz/abs-census-augmentor/pull/86)). Five breaking changes need handling on upgrade — all loud, none silent. **A 50-row temporal-mode spike against v2.0.0 surfaced two release-resolution gaps**: GCP cross-edition lookup returns NaN for 2016-dated rows (uses Edition-3 SA2 codes against Edition-2 keyed data), and ERP raises `RuntimeError` because ABS publishes the full annual time series in one workbook rather than per-year releases. SEIFA temporal works end-to-end. **Recommended phasing**: take the easy wins now (PR A — pin bump + dataset-id rename + ERP age/sex + the three PRESETs, all in cross-sectional mode), file upstream issues for the GCP + ERP gaps, then plan temporal mode (PR B / spec §7.7.2) as its own phase once the upstream gaps close.

## Why this review exists

`abs-census-augmentor` shipped a major version 17 weeks after we wrote the [v1.5 review](2026-05_abs_census_augmentor_v1.5_review.md). The v1.5 doc flagged temporal-DSS as **further deprioritised** ([§7.7.4](../../spec.md)) on the basis that static DSS signal was already at the noise floor — but that argument was made when "temporal" meant single-edition (2021 boundaries only). v2.0 ships **cross-edition** temporal, which is a different proposition: it's not just per-quarter DSS variation, it's per-row resolution to the contemporaneous Census release. That deserves a fresh look.

## What changed in v2.0.0

[Release notes](https://github.com/cauldnz/abs-census-augmentor/releases/tag/v2.0.0). Six functional themes, in roughly decreasing order of relevance to us.

### 1. Temporal mode is now cross-edition (Phases F.1–F.4 + G)

v1.5 shipped `Pipeline.augment(df, date_column=...)` but it only worked against ASGS Edition 3 (2021 boundaries). For our 2016-09 → 2026-04 panel, pre-2023-Q2 rows would either fail or null out. The v1.5 review labelled this as a blocker for adopting temporal mode in the project.

v2.0.0 closes the gap *in principle*. Per-row release resolution now spans both editions:
- 2016 rows resolve to Edition 2 boundaries + 2016 GCP/SEIFA values
- 2021+ rows resolve to Edition 3 boundaries + 2021 GCP/SEIFA values
- ERP: release notes promise per-row matching back to 2001 via `population_history_<year>` — but our spike found ERP temporal resolution only sees the latest publication; see [Surprise 1](#surprise-1-erp-temporal-resolution-is-single-release-only)
- DSS releases quarterly; default resolution is `closest_at_or_before` (causal), overridable to `closest`
- G-NAF geocoder also fans out per-row to the closest geocoder release

Spike findings below show the implementation has at least two release-resolution gaps that block clean adoption for our use case today.

New output columns surface what was chosen:
- `<dataset>_release` — which snapshot supplied this row's values (e.g. `seifa_release="2021"`)
- `sa2_code_edition` — which ASGS edition the row's SA2 code is keyed against
- `<dataset>_sa2_code_source` — the dataset-specific source-edition SA2 code (for cross-edition joins)
- `gnaf_release` — per-row YYYYMM G-NAF release (temporal + G-NAF only)

### 2. Historical-data expansion (PRs #75, #81)

The 2016 release for both SEIFA and GCP is now registered alongside 2021:
- **GCP 2016** — same DataPack parser, same `G##.*` variable namespace
- **SEIFA 2016** — uses `python-calamine` for the legacy `.xls` format (2021 was `.xlsx`)

The dataset IDs renamed accordingly: `gcp_2021 → gcp`, `seifa_2021 → seifa`. Variable refs (`G02.Median_age_persons`, `SEIFA.irsd_score`) are unchanged — only the *dataset id* used at the pipeline-API level moved.

Implication for us: combined with temporal mode, this is what makes per-row Census selection meaningful. Without 2016 snapshots, "temporal mode" would just be "always use 2021" for the Census-family variables and only DSS would have anything to vary on.

### 3. ERP age/sex columns ([PR #82](https://github.com/cauldnz/abs-census-augmentor/pull/82))

v1.5 fetched `ERP.population_total` plus the `population_history_<year>` series. v2.0 adds:
- `population_male`, `population_female`
- `population_0_14`, `population_15_64`, `population_65_plus`
- `median_age`

These were on v1.5's spec markdown but missing from the fetcher (that's the spec-vs-fetcher drift documented in [§7.7.3](../../spec.md) and upstream [#65](https://github.com/cauldnz/abs-census-augmentor/issues/65)). v2.0 closes the gap by actually emitting them.

Implication: our SA2 block currently has `sa2_total_population` (rank-low feature in the curation). The age-cohort splits could complement the DSS welfare columns we already curated — e.g. `pct_age_pension_recipients` over `population_65_plus` instead of total population is a cleaner demand-side ratio. These are usable in **cross-sectional** mode immediately; no temporal refactor required.

### 4. Three cross-dataset PRESETs ([PR #86](https://github.com/cauldnz/abs-census-augmentor/pull/86))

The first PRESETs that pull numerator and denominator from *different* datasets:
- `PRESET.pct_age_pension_recipients` — Age Pension recipients (DSS) ÷ 65+ population (ERP)
- `PRESET.pct_jobseeker_recipients` — JobSeeker recipients (DSS) ÷ working-age population (ERP)
- `PRESET.welfare_density_index` — composite of 9 DSS payment types ÷ total population (ERP)

We currently take **raw DSS recipient counts**, which collinearise with `n_pop_total` and the various `stn_competitors_within_*km` density signals. Ratios should be more decorrelated. Cheap to add as part of the static-surface bump.

### 5. Five breaking changes

All five are loud at config-load or runtime — none silently mis-behave.

| Change | Files to update in our repo |
|---|---|
| `gcp_2021` → `gcp` | `src/fuel_pred/config.py::AUGMENTOR_VARIABLES` (3 entries) — variable refs (`G##.*`) unchanged |
| `seifa_2021` → `seifa` | `src/fuel_pred/config.py::AUGMENTOR_VARIABLES` (4 entries) — variable refs (`SEIFA.*`) unchanged |
| `ato_personal_income` → `abs_personal_income`, namespace `ATO.*` → `ABS_PIA.*` | `src/fuel_pred/config.py::AUGMENTOR_VARIABLES` (4 entries) — already handled at the v1.5 review (we use `ABS_PIA.*` today; double-check no `ATO.*` lingers) |
| Cache layout: flat → per-ASGS-edition subdirs | Wipe `data/raw/<augmentor-cache>/` on upgrade (no auto-migration) |
| Devcontainer Docker socket removal | Not relevant — we don't use their devcontainer |

### 6. Performance + architectural ([#43](https://github.com/cauldnz/abs-census-augmentor/issues/43))

Warm-cache run 5.4s → 2.2s (parsed-result pickle/feather sidecars). Not material for us — we run the augmentor once per spec phase and parquet-cache the result. The four ~330-line dataset modules collapsed to ~150 lines each via a `_AbsXlsxDataset` shared base; no API-surface impact.

## Spike findings: temporal-mode behaviour on a 50-row sample

**Goal:** verify that v2.0 actually swaps in different SA2 values for pre-2021 vs. post-2021 rows at the same station — i.e. that temporal mode buys us something *empirically*, not just architecturally.

**Method:** [`tools/research/v2_spike.py`](../../tools/research/v2_spike.py) — sample 25 stations that have rows on both 2017-06-15 and 2023-06-15, join lat/lon from `stations.parquet`, run `Pipeline.augment(df, date_column='date')` with a mix of GCP / SEIFA / ERP / PRESET variables. Compare per-station 2017 vs 2023 values.

**Environment:** v2.0.0 installed ephemerally via `uv run --no-project --with "abs-census-augmentor @ git+...@v2.0.0"`. No changes to the project venv. Output captured to `tools/research/v2_spike.out.txt` (gitignored).

### Surprise 1: ERP temporal resolution is single-release only

First spike iteration requested `ERP.population_total`, `ERP.population_65_plus`, `ERP.median_age` alongside GCP + SEIFA + the new `PRESET.pct_age_pension_recipients`. Failed at temporal resolution for the 2017 rows:

```
RuntimeError: ERP release '2017' not found. Available: ['2024']
```

This is structural, not a bug. ABS publishes ERP as a **single annual workbook** containing the full time series back to 2001 in `population_history_<year>` columns. There is no separate "2017 ERP release" to load — there's the 2024 publication, which contains a 2017 column. v2.0's per-release temporal resolution doesn't model this shape; it treats ERP like SEIFA (one snapshot per release year) and fails on any row whose date doesn't match a publication year.

**Implication for PR A (static-surface bump):** the new ERP age/sex columns are usable, but only as the latest-release snapshot (i.e. 2024). The `population_history_<year>` time series is also available (we don't use it) and could provide per-row population for a specific year if we wired it manually — but that's not what v2.0's temporal mode does today.

**Implication for PR B (temporal mode):** ERP must be excluded from the temporal-augment pass, or the call must be split (cross-sectional for ERP, temporal for SEIFA / GCP / DSS). Worth filing upstream as a clarifying issue — the README implies "temporal mode works for all datasets", but ERP's annual-publication-with-time-series shape needs special handling.

### Surprise 2: PRESETs with ERP denominators inherit the limitation

`PRESET.pct_age_pension_recipients` divides DSS Age Pension recipients by ERP 65+ population. The DSS numerator has per-quarter releases. The ERP denominator hits the same RuntimeError above. So all three new cross-dataset PRESETs from PR #86 are de facto cross-sectional in v2.0, regardless of whether `date_column` is set.

**Implication for PR A:** still worth adopting, just understand they freeze to the latest ERP release rather than tracking the row's date.

### What worked

Second iteration (GCP + SEIFA only). `releases_used` came back as:

```python
{"gcp": ["2016", "2021"], "seifa": ["2016", "2021"]}
```

Per-row release columns confirm per-row resolution is happening:

```
station_id        date        sa2_code   gcp_release  seifa_release
0202f5920e10c16d  2017-06-15  107041144  2016         2016
0202f5920e10c16d  2023-06-15  107041144  2021         2021
```

**SEIFA temporal works end-to-end**: 25 of 25 sampled stations had different `sa2_seifa_irsd` values between 2017 and 2023 (median 991.00 → 987.20 — population disadvantage scores drifted with the demographic shift). This is what we wanted to confirm.

### Surprise 3: GCP cross-edition lookup returns NaN for 2016 rows

For every 2017-dated row the GCP variables came back null, even though the 2016 release was loaded. Diagnosing with [`tools/research/v2_spike_diagnose.py`](../../tools/research/v2_spike_diagnose.py) for a single station (Lyneham, ACT, SA2 `801051057`):

| Date | gcp_release | gcp_sa2_code_source | sa2_g01_total_pop | sa2_g02_median_age | sa2_seifa_irsd |
|---|---|---|---|---|---|
| 2017-06-15 | 2016 | 801051057 | **NaN** | **NaN** | 1056.0 |
| 2023-06-15 | 2021 | 801051057 | 5703.0 | 35.0 | 1053.5 |

The augmentor *believes* the 2016 release was used (column says so), but the value is null. The `gcp_sa2_code_source` column reports the **Edition 3** SA2 code (`801051057`) for the 2016 release lookup — but 2016 GCP is keyed by Edition 2 SA2 codes (different format). The cross-edition SA2 code translation isn't happening for GCP, so the lookup misses.

SEIFA doesn't have this problem in our sample (values returned for both years). Possible reasons: SEIFA 2016 may be name-keyed rather than code-keyed, or SEIFA's spatial reload path happens to converge on a valid code by coincidence. Worth probing further.

### Net spike conclusion

| Component | Cross-sectional (today) | Temporal (v2.0) |
|---|---|---|
| GCP 2021 | ✅ works | ✅ works for 2021+ rows |
| GCP 2016 (new in v2.0) | ✅ works if explicitly pinned | ❌ **returns NaN via temporal mode** — see Surprise 3 |
| SEIFA 2021 | ✅ works | ✅ works for 2021+ rows |
| SEIFA 2016 (new in v2.0) | ✅ works if explicitly pinned | ✅ works via temporal mode |
| ERP (all years) | ✅ works (latest release) | ❌ **raises RuntimeError** — see Surprise 1 |
| DSS (untested in spike — quarterly) | ✅ works | ⚠️ likely works for Ed.3 era (2023-Q2+), untested for Ed.2 era |
| Cross-dataset PRESETs (new in v2.0) | ✅ works | ❌ inherit ERP limitation |

The architectural ambition of v2.0's temporal mode is correct, but the implementation has at least two release-resolution gaps that block a clean adoption today. SEIFA is the only dataset where v2.0 temporal mode delivers what the release notes promise for our use case.

## Recommended phasing

**Don't auto-upgrade.** A major version with five breaking changes that also redesigns the temporal-augmentation surface deserves planning, not a one-shot pin bump. **The spike further argues against a single combined upgrade**: temporal mode in v2.0.0 only delivers per-row values for SEIFA cleanly; GCP nulls out for 2016 rows and ERP raises RuntimeError. Temporal mode is not ready for our panel as shipped.

Suggested phasing (one phase per PR, per [CLAUDE.md](../../CLAUDE.md) §workflow):

### PR A: v2.0 upgrade + low-risk static-surface bump (recommended next)

Scope:
- Bump pin in `pyproject.toml` + `uv.lock` to `@v2.0.0` (or the equivalent commit SHA `887ec011`)
- Update `src/fuel_pred/config.py::AUGMENTOR_VARIABLES`: `gcp_2021.*` → `gcp.*`, `seifa_2021.*` → `seifa.*`
- Audit for any lingering `ato_personal_income` / `ATO.*` refs and finish the rename
- Wipe augmentor cache dir (one-line in `Makefile clean`-equivalent)
- Add ERP age/sex columns to `AUGMENTOR_VARIABLES`: `population_65_plus`, `median_age` (skip the gender split for v1 — single rank in the curation if added)
- Add the three cross-dataset PRESETs: `pct_age_pension_recipients`, `pct_jobseeker_recipients`, `welfare_density_index`
- Re-run `enrich_census` + `train_models` end-to-end; document headline-experiment delta in [`results/README.md`](../../results/README.md) (per the v1.5 review's recommendation #1 — every minor-version bump is a hyperparameter)
- Update spec §7.7 to mention the new ERP columns + PRESETs and reflect the dataset-id rename

Risk: low. Variable refs unchanged for the most-used datasets; new columns drop into the existing 31-col superset; curation step (15-col model block) decides what survives. Cross-sectional mode is unchanged in v2.0 — same call shape, same return semantics.

Estimate: 1–2 sessions.

### Pre-PR-B work: upstream issues filed

Two upstream issues filed alongside PR A:

1. **[abs-census-augmentor#91](https://github.com/cauldnz/abs-census-augmentor/issues/91)** — GCP cross-edition lookup returns NaN for the 2016 release ([Surprise 3](#surprise-3-gcp-cross-edition-lookup-returns-nan-for-2016-rows)). The augmentor reports `gcp_release="2016"` and `gcp_sa2_code_source="<Ed.3 code>"` but the value lookup misses. The diagnostic script [`tools/research/v2_spike_diagnose.py`](../../tools/research/v2_spike_diagnose.py) is a 30-line reproducer.
2. **[abs-census-augmentor#92](https://github.com/cauldnz/abs-census-augmentor/issues/92)** — ERP temporal-release resolution only sees the latest publication ([Surprise 1](#surprise-1-erp-temporal-resolution-is-single-release-only)). ABS publishes ERP as one annual workbook with the full time series back to 2001; the augmentor's per-release temporal resolution doesn't model this. The "fix" might be docs (clarify that ERP is cross-sectional only, or use `population_history_<year>` for back-fill) rather than code.

These aren't blockers for PR A. They are blockers for PR B until resolved upstream.

### PR B: temporal-mode adoption (replaces deferred §7.7.2) — blocked until upstream gaps close

Scope per spec §7.7.2's deferred architecture (held until the two upstream issues land):
- New pipeline step between `panel_grid` and `make_features` (or fold into `enrich_census`) that takes the panel rather than `stations.parquet` as input
- Call `Pipeline.augment(panel_df, date_column='date', latitude_column='lat', longitude_column='lon')` once per (station, date) row
- Persist enriched panel to `data/processed/panel_enriched.parquet` (or fold into `features.parquet` directly)
- Decide DSS resolution rule: `closest_at_or_before` (default, causal — what we want) vs `closest` (smoother but peeks 6+ weeks forward at quarter mid)
- Re-curate the SA2 block — likely several columns will swap from "static 2021 value broadcast across panel" to "snapshot-at-row-date value", which may reshuffle gain rankings
- Headline experiment: does the cross-edition + per-row DSS resolution beat the static-block baseline on test_normal? (test_crisis is more interesting still — the 2026 crisis fold may legitimately benefit from DSS values from 2025-Q4 rather than the static 2025-Q3 we use today)

Risk: medium-high. Even with upstream fixes, panel-shaped augmentor runs against ~15M rows in temporal mode; expect a slower run + more cache disk than the per-station model. Cache hits across (station, date) combinations should dominate runtime, but the dedupe-to-unique-(station, release)-tuples optimisation should be implemented in our pipeline regardless of what the augmentor does internally.

Open questions to resolve before starting:
- Status of GCP cross-edition + ERP temporal resolution upstream
- Does v2.0 cache deduplicate per-(SA2, release) lookups, or recompute spatial joins for every row? If the latter, 15M rows × per-row spatial lookup is unworkable and we need a pre-pass to dedupe to ~5K unique (station, release) tuples
- What's the disk + time budget for first-run cache warmup? v2.0 keeps ASGS Ed.2 + Ed.3 + 2016 + 2021 GCP DataPacks + DSS quarterly history — likely 1-2 GB
- Cross-edition `<dataset>_sa2_code_source` columns: do we expose these in our feature schema, or hide them inside enrich_census? (Probably hide — they're plumbing, not features)

Estimate: 2–3 sessions plus the headline-experiment retrain, *after* upstream gaps close.

### What NOT to do

Don't combine PR A and PR B. The v1.5 review's recommendation #1 is "re-run your headline experiment on every minor-version bump." A combined PR conflates upstream-parsing deltas (PR A's scope) with temporal-architecture deltas (PR B's scope), and we lose the ability to attribute any regression to either cause. **Plus**: PR B's value proposition rests on temporal mode working correctly across all our SA2-family datasets, and the spike shows that's not the case in v2.0.0. Shipping PR A independently captures the documented wins immediately; PR B waits for upstream readiness.

## PR B outcome (2026-05-30 update)

Upstream resolved both issues filed during the spike:

- **#92 (ERP single-publication)** — fully fixed via PR #95. `ErpDataSource` now serves any historical year ≤ latest via column projection. Spike verification confirmed `releases_used` reports per-year resolution (2017, 2018, ..., 2024); per-row `sa2_erp_population_total` 100% non-null.
- **#91 (GCP cross-edition NaN)** — Stage 1 only (PR #94). Silent NaN now raises a loud `ValueError`. The proper per-release `DataPacksDataSource` routing (Stage 2) remains on backlog. We route GCP cross-sectional in PR B to avoid hitting the error.

Bumped to upstream main at commit `65fd3fa6` to pick up both fixes plus PR #89 (SEIFA 2011), PR #90 (4 more cross-dataset PRESETs), and PR #93 (carer-payment PRESET) — the last three are available in the augmentor surface but not adopted yet (would need a separate curation experiment).

### PR B implementation: split-pass architecture

- **Cross-sectional pass** (existing `build.enrich_census`): GCP direct + GCP-internal PRESETs + ERP age/sex + ABS_PIA + cross-dataset PRESETs + DSS welfare. DSS held back from temporal pending upstream #99.
- **Temporal pass** (new `build.enrich_panel_temporal`): SEIFA scores + `ERP.population_total`. Output joins on (station_id, date).

The architecture works end-to-end. Coverage on the 6.86M unique (station_id, date) panel: SEIFA 99.2-99.6%, ERP 100%, releases used `{seifa: ['2016', '2021'], erp_by_sa2: ['2017', ..., '2024']}`.

### PR B headline result: temporal demographics regressed both folds

Same 15-col SA2 block as PR A, same Model A/B contrast — just SEIFA + ERP-total swapped from static (per-station) to temporal (per-row):

| Fold | PR A Δ MAE (cross-sectional) | PR B Δ MAE (temporal) | Difference |
|---|---:|---:|---:|
| test_normal | −0.353 | −0.239 | +0.114 (worse) |
| test_crisis | −0.398 | −0.321 | +0.077 (worse) |

The hypothesis behind §7.7.2 — "temporal demographics will improve test-fold accuracy" — is **not empirically supported** on this problem as configured. Three plausible mechanisms for the regression:

1. **2016 SEIFA values may be noisier than 2021** (older Census, different methodology). Pre-2021 panel rows that now get 2016 SEIFA scores might lose accuracy vs the cleaner 2021 values they had under static-mode.
2. **The model was already extracting whatever temporal-demographic signal exists** via `date`/year features in the cal block. Per-row variation re-encodes a signal that the model had a different (working) handle on.
3. **The panel skews post-2020** (lag/rolling features can't fit early dates), so the older releases get applied to a smaller training subset where they're less informative.

Diagnosing which mechanism dominates would require a follow-up experiment (e.g. temporal-SEIFA-only vs temporal-ERP-only vs both vs neither). Out of scope for PR B.

### What landed in PR B (and what didn't)

| What | Status |
|---|---|
| Augmentor pin bump to main `65fd3fa6` | ✅ |
| `config.AUGMENTOR_VARIABLES_{CROSS_SECTIONAL,TEMPORAL}` split | ✅ |
| `build.enrich_panel_temporal` module | ✅ |
| `make_features.add_sa2_features` temporal-merge logic | ✅ |
| Makefile `enrich-panel-temporal` target | ✅ |
| Spec §7.7.2 marked landed + §7.7.5 follow-up note | ✅ |
| `results/README.md` iteration table + headline updated to PR B numbers | ✅ |
| Tests: 8 new for `enrich_panel_temporal`, 22 passing total for the SA2 enrich modules | ✅ |
| DSS in the temporal pass | ❌ — blocked by upstream cauldnz/abs-census-augmentor#99 (DSS XLSX parser fails on 2022-Q4). One-line config change once that lands. |
| Net improvement in headline Δ MAE | ❌ — regression of ~0.1 c/L. Architecture stays in place anyway; future ablation experiments can flip columns between passes trivially. |

### Recommendation for future projects

Don't assume "more temporal resolution = better predictions". For tree-based models on already-well-featured tabular problems, the model often extracts temporal-demographic signal indirectly via date/year features and adding per-row SA2 variation introduces noise on the release boundaries. **Always treat temporal-mode adoption as a hypothesis to test, not a free upgrade.** Land the architecture so future ablations are cheap, but report the headline honestly.

## Known limitations of v2.0

- **Address-retirement awareness deferred post-G** ([release notes](https://github.com/cauldnz/abs-census-augmentor/releases/tag/v2.0.0)). Retired G-NAF addresses don't auto-rejoin to current-edition counterparts. Low-impact for our use case (fuel stations rarely change PID).
- **Per-dataset G-NAF resolution rules deferred.** Global `temporal.resolution` setting applies to all temporal datasets; can't yet mix `closest_at_or_before` for DSS with `closest` for ERP.
- **Spec-vs-fetcher drift may recur.** v1.5 had over-promised columns ([§7.7.3](../../spec.md), [#65](https://github.com/cauldnz/abs-census-augmentor/issues/65)). v2.0 closes the ERP gap (PR #82) but we should still probe-fetch before promising columns in our config.
- **Cache layout breaking change.** Plan for ~150–300 MB re-download on upgrade (ASGS Ed.2 boundaries + 2016 GCP + 2016 SEIFA).

## Recommendations for other downstream consumers

The v1.5 review's recommendations all still apply. Adding three more, specific to v2.0:

6. **Adopt the static-surface additions before the temporal refactor.** ERP age/sex + cross-dataset PRESETs are independent of temporal mode and add demographic richness immediately. Treat them as the next minor-version-bump experiment.
7. **For temporal mode, dedupe to unique (lat, lon, release) tuples before augmenting.** Per-row spatial joins on a 15M-row panel will dominate runtime if not deduplicated.
8. **Document the choice of DSS resolution rule.** `closest_at_or_before` is causally clean; `closest` is smoother but peeks forward. For prediction problems (where peeking = leakage), `closest_at_or_before` is correct. Make this an explicit decision in your spec.

## See also

- abs-census-augmentor repo: https://github.com/cauldnz/abs-census-augmentor
- v2.0.0 release notes: https://github.com/cauldnz/abs-census-augmentor/releases/tag/v2.0.0
- [`docs/research/2026-05_abs_census_augmentor_v1.5_review.md`](2026-05_abs_census_augmentor_v1.5_review.md) — previous upgrade review
- [`tools/research/v2_spike.py`](../../tools/research/v2_spike.py) — the 50-row temporal-mode spike (run via `uv run --no-project --with "abs-census-augmentor @ git+https://github.com/cauldnz/abs-census-augmentor.git@v2.0.0" --with pandas --with pyarrow python tools/research/v2_spike.py`)
- [`tools/research/v2_spike_diagnose.py`](../../tools/research/v2_spike_diagnose.py) — single-station reproducer for the GCP cross-edition NaN issue (suitable for upstream filing)
- [`spec.md`](../../spec.md) §7.7.2 (temporal-DSS deferral — about to be re-opened), §7.7.4 (block curation, signal-floor argument)
- [`src/fuel_pred/build/enrich_census.py`](../../src/fuel_pred/build/enrich_census.py) — integration code touched by PR A
- [`src/fuel_pred/config.py`](../../src/fuel_pred/config.py) — `AUGMENTOR_VARIABLES` (the rename surface)
- Upstream PRs: [#75 (SEIFA 2016)](https://github.com/cauldnz/abs-census-augmentor/pull/75), [#81 (GCP 2016)](https://github.com/cauldnz/abs-census-augmentor/pull/81), [#82 (ERP age/sex)](https://github.com/cauldnz/abs-census-augmentor/pull/82), [#86 (cross-dataset PRESETs)](https://github.com/cauldnz/abs-census-augmentor/pull/86), [#87 (v2.0.0 cut)](https://github.com/cauldnz/abs-census-augmentor/pull/87)
