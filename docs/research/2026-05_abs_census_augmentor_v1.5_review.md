# abs-census-augmentor: the v1.4.2 → v1.5 upgrade

**Date:** 2026-05
**Augmentor versions reviewed:** v1.4.2 (initial integration, commit `28cd65e3`) and v1.5 (current pin, head `8fe6fa55`, post-PR [#64](https://github.com/cauldnz/abs-census-augmentor/pull/64))
**Project pin:** `abs-census-augmentor @ git+https://github.com/cauldnz/abs-census-augmentor.git` ([`pyproject.toml`](../../pyproject.toml) line 37 — unversioned, follows main)
**TL;DR:** Re-running the same 10-column SA2 schema against a newer augmentor changed our headline test-fold result from "Model B *loses* by 0.104 c/L MAE" to "Model B *wins* by 0.059 c/L" — a 0.163 c/L swing produced entirely by upstream parsing fixes, no schema changes on our side. If you're using this library on any prediction problem, upgrading across v1.3 → v1.4.2 → v1.5 is not a no-op; you should re-run your headline experiment.

## Why this review exists

`abs-census-augmentor` is this project's most reusable dependency — it has no fuel-price-specific code, and the same call (`Pipeline.augment(df, latitude_column=..., longitude_column=...)`) drops Census, SEIFA, ERP, ABS-PIA, and DSS welfare features into any DataFrame with Australian lat-lon. That makes it a candidate for school catchment analysis, healthcare access modelling, retail siting, insurance pricing — anywhere a SA2-keyed demographic signal might matter.

We integrated at v1.4.2, then upgraded to v1.5 weeks later. The upgrade had two distinct impacts: (1) the same column names produced materially better column values (parsing fixes), and (2) a wide new column surface opened up (DSS welfare, expanded SEIFA, ERP, ABS-PIA), which we had to learn to *not* use uncritically. The first is the more important and is described first.

## What changed in v1.5

### Parsing improvements (the big one)

v1.3 introduced the `PRESET.<id>` namespace — curated derived ratios (`pct_renters`, `pct_drive_to_work`, `motor_vehicles_per_dwelling`, etc.) with the right denominators pre-baked. v1.4.0 ([PR #18](https://github.com/cauldnz/abs-census-augmentor/pull/18)) promoted them to first-class pipeline variables. **All of them silently returned wrong values until v1.4.2.**

[Issue #23](https://github.com/cauldnz/abs-census-augmentor/issues/23) ("PRESET column references don't match real GCP DataPack") catalogued the breakage. Every shipped PRESET referenced column names that didn't exist in the actual ABS GCP 2021 DataPack — and the *tests passed* because the synthetic fixtures encoded the same broken names. Examples:

| PRESET | v1.3 / v1.4.0 spec referenced | Real GCP column |
|---|---|---|
| `pct_renters` | `G37.R_Tot` / `G37.OPDs_Total` | `G37.R_Tot_Total` / `G37.Total_Total` |
| `pct_drive_to_work` | `G46.OneMethod_CarAsDriver_P` (camelCase) | `G46.One_method_*_P` (snake_case) |
| `pct_aged_65_plus` | `G04.<col>` | `G04A.<col>` (males) + `G04B.<col>` (females) |
| `motor_vehicles_per_dwelling` | `G31.Total_*` | `G31.Num_MVs_per_dweling_*` (weighted sum) |
| `pct_one_parent_family` | `G25.OneP_F_C_Tot` | `G25.CF_*` / `G25.OPF_*` |

v1.4.2 (commit `28cd65e3`, [PR #26](https://github.com/cauldnz/abs-census-augmentor/pull/26)) rewrote every PRESET against the real DataPack and added `tools/verify_real_parsers.py` — an acid-test that resolves every PRESET source-field reference against the live GCP catalogue — plus reference dumps at `tests/fixtures/gcp-schemas/G##.txt`.

We wired the augmentor in *after* v1.4.2 shipped, so we never hit the broken PRESETs directly. We hit something subtler. Moving from v1.4.2 to v1.5 with **the same 10 columns** ([`enrich_census.py`](../../src/fuel_pred/build/enrich_census.py) `DIRECT_VARIABLES`), the parsing logic kept improving — including a format-agnostic `_read_grids()` / `_parse_grids()` refactor in temporal Phase F.3 — and the *values* coming back tightened up enough to flip the experiment's sign.

Lesson for downstream consumers: **construction-time validation only proves variable refs parse, not that they resolve to values you'd recognise**. Even a green upstream test suite doesn't mean the numbers are right. Re-run your headline experiment after any minor-version bump.

### New column expansions

v1.5 widened the augmentor's namespace from "GCP + SEIFA-IRSD" to a broad demographic surface:

- **SEIFA**: 1 → 4 scores. IRSD (disadvantage), IRSAD (advantage + disadvantage), IER (economic resources), IEO (education + occupation). Phase F.3 also added the 2016 release (`seifa_2021` → `seifa` dataset rename) via `python-calamine` for legacy `.xls`.
- **ERP** (Estimated Resident Population — annual): new namespace. Spec promised 5 derived columns; the v1.5 fetcher emits 1 (`population_total`) + 25 historical-year columns. Drift documented in [spec §7.7.3](../../spec.md), filed upstream as [#65](https://github.com/cauldnz/abs-census-augmentor/issues/65).
- **ABS_PIA** (ABS Personal Income Analysis — renamed from `ATO` in temporal Phase C): 5 summary stats per SA2 (`median_total_income`, `mean_total_income`, `income_earners_count`, `median_age_of_earners`, `sum_total_income`).
- **DSS** (welfare-payment recipient counts, quarterly): 21 columns per SA2 — Age Pension, JobSeeker, DSP, Parenting Payment (Single/Partnered), Carer Payment/Allowance, Youth Allowance, Commonwealth Rent Assistance, FTB-A/B, and more. The closest thing in Australia to a demand-side instrument tied to SA2.

Structural change: `Pipeline.augment()` now dispatches SEIFA / ERP / DSS / ABS_PIA through the same unified path as GCP, so the bespoke `SeifaDataSource` join we had in v1.4.2 became redundant. Caches also moved to edition-keyed subdirectories (`<data_dir>/boundaries/2021/`, …) — a breaking change requiring a cache wipe on upgrade.

### Temporal DSS (deferred in our project)

The headline v1.5 feature is **temporal augmentation**: `Pipeline.augment(df, date_column=...)` resolves each row to the closest dataset snapshot in time, giving per-(entity, date) values rather than a single static snapshot. For DSS this is especially interesting — DSS publishes every calendar quarter going back to 2022-Q4, so a fortnightly Centrelink-day pricing story could use the *current* welfare population at each prediction date.

v1.5 ships temporal mode **single-edition only** (ASGS Edition 3, 2021 boundaries). Cross-edition orchestration (Phases F.1/F.2) is in the upstream backlog ([PR #64](https://github.com/cauldnz/abs-census-augmentor/pull/64) `BACKLOG.md`).

We deferred for three reasons (documented in [spec §7.7.2](../../spec.md) / §7.7.4):

1. **Architectural change.** Temporal mode runs against a panel-shaped DataFrame (one row per (station, date), ~50M rows here), not per-station `stations.parquet` (~2,500 rows). Adopting it requires a new pipeline step and Makefile rewire.
2. **Cross-edition limitation.** Our train fold runs to 2022-12-31, but pre-2023-Q2 DSS releases are on ASGS Edition 2 while our pipeline is Edition 3. Single-edition temporal would null out for the bulk of training data.
3. **Signal floor.** Static DSS columns contributed at-best 0.04% gain in our final model. **If the static signal floor is the ceiling, temporal resolution has nowhere to go** — the actionable inference for anyone weighing the panel-augmentation refactor.

## The empirical impact in our project

From [`results/README.md`](../../results/README.md), the iteration table:

| Iteration | SA2 cols | Test_normal Δ MAE | What we learned |
|---|--:|--:|---|
| v1.4.2 augmentor, 10-col block | 10 | **+0.104** (Model B *lost*) | First run; SA2 hurt the headline fold |
| v1.5 augmentor, 10-col block | 10 | **−0.059** | Improved parsing alone flipped the sign — same names, better values |
| v1.5 augmentor, 31-col block | 31 | **−0.025** | All v1.5 namespaces regressed — better val, worse test: overfitting |
| **v1.5 augmentor, 15-col block (final)** | **15** | **−0.391** | Original 10 + top-5 by gain importance recovered the full benefit |

Read the first two rows together: **same 10 column names, +0.104 → −0.059 c/L, just from upgrading the library**. A 0.163 c/L swing (~2.5% of absolute MAE) on the headline test-fold from upstream parsing improvements alone.

## How we used the augmentor

[`src/fuel_pred/build/enrich_census.py`](../../src/fuel_pred/build/enrich_census.py) is the integration point (~400 lines). Patterns worth borrowing:

- **Fetch broader than you model.** [`config.AUGMENTOR_VARIABLES`](../../src/fuel_pred/config.py) requests ~31 columns into `stations.parquet`; the model consumes 15. The extra 16 stay available for future ablation — augmentation is expensive (network + ABS workbook parsing), feature selection is cheap.
- **Test seam: `pipeline_factory`.** `enrich()` takes an optional `pipeline_factory` so unit tests can swap in a synthetic augmentor without network or cache. CLAUDE.md mandates hermetic tests.
- **GCP/PRESET collision workaround.** v1.4.2 has a bug where requesting a direct GCP variable AND a PRESET using the same code as a source crashes `_build_gcp_lookup` with `ValueError: cannot reindex on an axis with duplicate labels`. We split colliding requests into two augment passes (`_split_for_gcp_collision`). See `UPSTREAM_GCP_COLLISION` in the file.

## Recommendations for other projects

1. **Re-run your headline experiment on every minor version bump.** Our 10-col schema swung 0.16 c/L (~2.5% MAE) just from v1.4.2 → v1.5 parsing fixes. Treat the augmentor's version like a hyperparameter that affects every row.
2. **Pin to a specific commit, not `@main`.** Our [`pyproject.toml`](../../pyproject.toml) currently pins the git URL without ref — a known weakness; `uv sync` can silently shift. Use `@<commit-sha>`, bump deliberately, re-run the headline.
3. **Probe-fetch before promising columns.** Spec markdown documents columns v1.5 fetchers don't emit (ERP, ABS_PIA, DSS — [#65](https://github.com/cauldnz/abs-census-augmentor/issues/65)). Call `dataset.load()` and check `.columns` before writing a variable list. `Pipeline.create(variables=...)` succeeding only proves your refs *parse*.
4. **Curate before you train.** DSS alone is 21 columns; our 31-col block overfit (val MAE improved, test MAE regressed). Use feature-importance ranking from a broadened experiment to pick the top-N. See [`2026-05_sa2_feature_curation.md`](2026-05_sa2_feature_curation.md).
5. **For temporal data: budget the panel refactor.** Temporal mode reshapes the augmentor's place in your pipeline (per-row, not per-entity). Don't adopt until your static experiments show the per-quarter signal is genuinely large.

## Known limitations of v1.5

- **Cross-edition temporal not implemented** (Phases F.1/F.2 — [PR #64](https://github.com/cauldnz/abs-census-augmentor/pull/64) `BACKLOG.md`). If your span crosses ASGS Edition 2 → 3 (pre-/post-2023-Q2), temporal mode nulls out for the older portion. `Pipeline.augment(df, date_column=...)` works against Edition 3 only; cross-edition raises `NotImplementedError`. SEIFA 2016 itself is now supported (Phase F.3), but boundary cross-edition orchestration isn't.
- **Spec-vs-fetcher drift recurs.** v1.5 dataset markdown over-promises for ERP (5 vapourware columns), ABS_PIA (5 vapourware), and DSS (1 wrong name, 12 undocumented) — see [spec §7.7.3](../../spec.md) and [#65](https://github.com/cauldnz/abs-census-augmentor/issues/65). Same root cause as [#23](https://github.com/cauldnz/abs-census-augmentor/issues/23). Until #65's suggested `test_spec_matches_fetcher_columns` rung lands, treat spec markdown as aspirational.
- **Cache layout breaking change** (Phase D): boundary / GCP / Mesh Block caches moved to edition-keyed subdirectories. Wipe `data/raw/<augmentor>/` on upgrade.
- **`ATO` → `ABS_PIA` namespace rename** (Phase C). Update variable refs.
- **Signal floor may be ceiling.** Per [spec §7.7.4](../../spec.md), even the best new v1.5 column (`sa2_dss_parenting_payment_partnered_recipients`) contributed only 0.04% gain; 16 of 21 added features were below noise floor. For tree-based models on already-feature-rich problems, the augmentor's marginal lift over a small curated baseline is modest.

## See also

- abs-census-augmentor repo: https://github.com/cauldnz/abs-census-augmentor
- [`docs/research/2026-05_sa2_feature_curation.md`](2026-05_sa2_feature_curation.md) — the 31 → 15 curation, gain-vs-SHAP divergence
- [`docs/research/2026-05_centrelink_seifa_interaction.md`](2026-05_centrelink_seifa_interaction.md) — day-of-fortnight × SEIFA hypothesis
- [`spec.md`](../../spec.md) §7.7 — integration spec: §7.7.1 PRESET history, §7.7.2 temporal-DSS deferral, §7.7.3 spec drift, §7.7.4 block curation
- [`src/fuel_pred/build/enrich_census.py`](../../src/fuel_pred/build/enrich_census.py) — integration code with `pipeline_factory` test seam and `_split_for_gcp_collision` workaround
- Upstream: [#18](https://github.com/cauldnz/abs-census-augmentor/pull/18), [#19](https://github.com/cauldnz/abs-census-augmentor/issues/19), [#23](https://github.com/cauldnz/abs-census-augmentor/issues/23), [#26](https://github.com/cauldnz/abs-census-augmentor/pull/26), [#64](https://github.com/cauldnz/abs-census-augmentor/pull/64), [#65](https://github.com/cauldnz/abs-census-augmentor/issues/65)
