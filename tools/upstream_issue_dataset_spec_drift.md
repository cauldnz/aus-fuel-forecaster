# Pending upstream issue — file at `cauldnz/abs-census-augmentor`

This file holds the body of an issue we want filed against the augmentor.
Once filed, link it from `src/fuel_pred/config.py` (search for
`PR #46`) and `spec.md` §7.7.3, then delete this file.

---

**Title:** `Dataset spec markdowns promise columns the v1.5 fetchers don't actually emit`

## Summary

The dataset spec markdown files in `datasets/` document an aspirational schema for `erp_by_sa2`, `abs_personal_income`, and (to a lesser degree) `dss_payments`. The actual columns the v1.5 fetchers return are a strict subset — sometimes a tiny subset — of what the markdown advertises. Downstream consumers reading the spec markdown as ground truth will write variable lists that crash with `Dataset 'X' doesn't expose columns [...]` at the first `Pipeline.augment(...)` call.

This is a documentation-vs-code mismatch, not a runtime bug. But it bit us hard enough that I wanted to flag it: I shipped a PR (cauldnz/aus-fuel-forecaster#45) against the spec markdowns and had to immediately ship a fix-up PR (#46) against the *real* fetcher output once an `enrich` run blew up.

## Specific gaps observed against `8fe6fa55` (post-PR #64, "Temporal Phase H")

### `datasets/erp_by_sa2.md` — Schema table claims:

| Variable | Status |
|---|---|
| `ERP.population_total` | ✅ emitted |
| `ERP.population_male` | ❌ not emitted |
| `ERP.population_female` | ❌ not emitted |
| `ERP.population_0_14` | ❌ not emitted |
| `ERP.population_15_64` | ❌ not emitted |
| `ERP.population_65_plus` | ❌ not emitted |
| `ERP.median_age` | ❌ not emitted |
| `ERP.reference_year` | ✅ emitted (as metadata) |
| `ERP.population_density_per_km2` | ❌ not emitted |

What the fetcher *actually* returns: `state_abbreviation`, `reference_year`, `population_total`, plus `population_history_2001` through `population_history_2025` (25 historical-year columns not mentioned in the spec at all).

### `datasets/abs_personal_income.md` — Schema table claims:

| Variable | Status |
|---|---|
| `ABS_PIA.median_total_income` | ✅ emitted |
| `ABS_PIA.mean_total_income` | ✅ emitted |
| `ABS_PIA.median_employee_income` | ❌ not emitted |
| `ABS_PIA.median_investment_income` | ❌ not emitted |
| `ABS_PIA.median_super_income` | ❌ not emitted |
| `ABS_PIA.median_own_business_income` | ❌ not emitted |
| `ABS_PIA.gini_coefficient` | ❌ not emitted |
| `ABS_PIA.income_earners_count` | ✅ emitted |
| `ABS_PIA.reference_financial_year` | ✅ emitted (as metadata) |

What the fetcher *actually* returns: 5 summary stats (`income_earners_count`, `median_age_of_earners`, `sum_total_income`, `median_total_income`, `mean_total_income`) plus `reference_financial_year`. No income breakdowns by source (employee / investment / super / business). No gini coefficient.

This appears to be by design — the loader explicitly only parses Table 1.4 of the source XLSX (the total-income summary sheet), not Tables 2-9 which would carry the breakdowns. But the spec markdown reads as if all of it is wired up.

### `datasets/dss_payments.md` — minor naming drift

The schema table lists `DSS.youth_allowance_student_recipients`. The actual column emitted is `youth_allowance_student_and_apprentice_recipients` (because that's what DSS publishes the column as, and the snake-caser preserves it).

The DSS spec table also under-promises: it lists 9 columns but the fetcher emits 21 (everything DSS publishes per quarter). That's *better* than the spec, but still a gap between docs and reality.

## Suggested resolution (pick one)

**Option A — Trim the spec markdowns to what the fetchers emit.** Conservative, ships immediately, removes the foot-gun. Add a note like "Schema may grow over releases; check `Registry.get('erp_by_sa2').load().columns` for the live list."

**Option B — Implement the spec'd columns.** Wire up ERP age bands + density + median age (the source XLSX has them on a different sheet); wire up ABS_PIA Tables 2-9 for the income breakdowns + gini. Honour the spec.

Either is fine; **the current state — spec drifting from code — is the problem**.

## Why this matters for downstream consumers

Anyone using `Pipeline.create(variables={...})` based on reading the spec markdown will hit `ValueError: Dataset 'X' doesn't expose columns [...]` only at the first `augment(df)` call, after the fetchers have already downloaded and parsed their workbooks (so several seconds in). The error message is good ("Available: [first 10 cols]...") but the user has no signal up-front that the spec is aspirational.

A `Registry.validate_variables(variables)` helper that did a one-shot dispatch + column check without doing the full augmentation would catch this at construction time. Could be a nice complement to either Option A or B.

## Tracking on our side

- `cauldnz/aus-fuel-forecaster#46` (the fix-up PR) lists the corrected variable names we settled on.
- `src/fuel_pred/config.py` AUGMENTOR_VARIABLES has comments referencing this issue.
- `spec.md` §7.7.3 documents the resulting narrowed surface for v1.

Happy to PR the spec-markdown trim (Option A) if you'd like.
