# Filed upstream issue

> **Status:** filed as <https://github.com/cauldnz/abs-census-augmentor/issues/65> on 2026-05-13.
> This file kept locally as a copy of the body — useful for context if the
> issue is ever closed without resolution.

---

# Spec markdowns claim columns the v1.5 fetchers don't emit (regression of #23 pattern)

## TL;DR

Three of the five registered datasets ship a `Schema (variables exposed by the augmentor)` table in their spec markdown that lists columns the v1.5 fetcher does not actually emit. Downstream consumers reading the spec markdown as the source of truth — which they should be able to, since it's the documented schema — write variable lists that crash with `Dataset 'X' doesn't expose columns [...]` only at the first `Pipeline.augment(df)` call, after several seconds of fetching and parsing.

This is the **same root-cause pattern as #23** ("PRESET column refs don't match real GCP DataPack columns"), where v1.3 PRESETs referenced columns that didn't exist and tests passed because synthetic fixtures encoded the same broken names. The category is *spec drift from code*; the test suite has no rung that catches it.

## Concrete drift, against current `main` (`8fe6fa55`, post-PR #64)

Each row below is **exactly what the fetcher returns from `.load().columns`**, probed by:

```python
from pathlib import Path
from census_augment.datasets._erp import ErpDataSource
from census_augment.datasets._dss import DssDataSource
from census_augment.datasets._abs_pia import AbsPiaDataSource

for name, cls, root in [
    ("ERP", ErpDataSource, "erp_by_sa2"),
    ("DSS", DssDataSource, "dss_payments"),
    ("ABS_PIA", AbsPiaDataSource, "abs_personal_income"),
]:
    ds = cls(root=Path("/tmp/probe") / root)
    ds.fetch()
    print(name, sorted(ds.load().columns))
```

### `datasets/erp_by_sa2.md`

| `Schema (...)` table promises | `ds.load().columns` actually has |
|---|---|
| `population_total` | ✅ |
| `population_male` | ❌ not emitted |
| `population_female` | ❌ not emitted |
| `population_0_14` | ❌ not emitted |
| `population_15_64` | ❌ not emitted |
| `population_65_plus` | ❌ not emitted |
| `median_age` | ❌ not emitted |
| `population_density_per_km2` | ❌ not emitted |
| `reference_year` | ✅ |

Plus 25 columns the spec doesn't mention at all: `population_history_2001` through `population_history_2025`, plus `state_abbreviation`.

So the spec promises 9 columns, 7 of which are vapourware, and the fetcher silently emits 26 columns the spec doesn't mention.

### `datasets/abs_personal_income.md`

| `Schema (...)` table promises | `ds.load().columns` actually has |
|---|---|
| `median_total_income` | ✅ |
| `mean_total_income` | ✅ |
| `median_employee_income` | ❌ not emitted |
| `median_investment_income` | ❌ not emitted |
| `median_super_income` | ❌ not emitted |
| `median_own_business_income` | ❌ not emitted |
| `gini_coefficient` | ❌ not emitted |
| `income_earners_count` | ✅ |
| `reference_financial_year` | ✅ |

Plus `median_age_of_earners` and `sum_total_income` — emitted, not in the spec table.

The fetcher's source code (`_abs_pia.py:50-63`) makes the design explicit: it only parses Table 1.4 (the total-income summary sheet). The spec table reads as if Tables 2-9 (income breakdowns, gini) are also wired up. They aren't.

### `datasets/dss_payments.md`

This one is closer to reality but has a naming bug:

- Spec claims `youth_allowance_student_recipients`. The fetcher emits `youth_allowance_student_and_apprentice_recipients` (because that's the column DSS publishes, snake-cased verbatim).

The DSS spec table also under-promises (lists 9 columns, fetcher emits 21). That's *better* than the other two, but still a docs-vs-code gap.

## Root cause

The augmentor's test suite has no rung that connects the spec markdown to the fetcher output. Specifically:

- `tests/test_dataset_erp.py:168` (`test_load_returns_sa2_indexed_dataframe`) builds a synthetic XLSX from `_erp_xlsx_bytes(...)` whose only data columns are `S/T code`, `S/T name`, ..., `SA2 code`, `SA2 name`, and a series of year columns. The test asserts `population_total`, `population_history_YYYY`, `reference_year` exist — i.e. it asserts what the fetcher *does* produce, not what the spec *says* it should produce.
- `tests/test_datasets_registry.py` has `test_parse_spec_with_schema_table` that verifies the schema-table parser. It does not verify those parsed `VariableSpec` entries actually correspond to columns the fetcher returns.

Result: spec markdowns can document any schema; fetcher tests can pass against a different schema; nothing fails. This is precisely the failure mode #23 was about, transposed from PRESET source-fields to dataset variable lists.

The recurrence is the part that warrants this issue. One-off documentation drift happens; a *pattern* of it on a project that publishes itself as a clean library API is something to fix at the test-architecture level, not by patching individual specs.

## Suggested fix

Add a `tests/test_spec_matches_fetcher_columns.py` that, for each registered dataset, fetches one record (offline against the existing synthetic fixtures) and asserts every `field` in `spec.variables` is present in `fetcher.load().columns`. Pseudocode:

```python
@pytest.mark.parametrize("dataset_id", ALL_REGISTERED_DATASETS)
def test_spec_columns_match_fetcher_output(dataset_id, tmp_path, monkeypatch):
    spec = registry.get(dataset_id)
    fetcher = registry.make_fetcher(dataset_id, root=tmp_path)
    # ... point fetcher at a synthetic fixture that mirrors the *real* source schema ...
    df = fetcher.load()
    spec_cols = {v.field for v in spec.variables}
    fetcher_cols = set(df.columns)
    missing_from_fetcher = spec_cols - fetcher_cols
    extras_in_fetcher = fetcher_cols - spec_cols
    assert not missing_from_fetcher, (
        f"{dataset_id}: spec claims columns the fetcher doesn't emit: "
        f"{sorted(missing_from_fetcher)}"
    )
    # Optional, but worth having: also flag undocumented bonus columns.
    if extras_in_fetcher:
        warnings.warn(
            f"{dataset_id}: fetcher emits undocumented columns: "
            f"{sorted(extras_in_fetcher)}",
            DocsDriftWarning,
        )
```

Hooking into the existing weekly `real-data-check.yml` workflow would catch real-data schema drift too: ABS could rename a column in a future release and we'd see it before downstream does.

The cost is low (one parametrized test) and the benefit is exactly the class of bug we keep tripping over.

## Downstream impact (FYI)

In `cauldnz/aus-fuel-forecaster`:
- PR #45 was drafted against the spec markdown; assumed all spec'd columns existed.
- First `make enrich --force` run: `ValueError: Dataset 'erp_by_sa2' doesn't expose columns [...]`. Hours of stations.parquet and ~150 MB of cached ABS workbooks already on disk by then.
- PR #46 walked the variable list back to what the fetchers actually emit. Schema we ended up with: ERP shrank 5 → 1, ABS_PIA grew 1 → 4 (gini was promised-but-missing), DSS grew 9 → 13 (corrected name + 4 bonus payments).

Diff: <https://github.com/cauldnz/aus-fuel-forecaster/pull/46>

## Suggested resolutions (pick one — but pick *one*; the current state is the problem)

- **Option A — trim specs to reality.** Remove the claimed-but-not-emitted rows from each `Schema (...)` table; add a "(More columns may be wired up in future releases — call `Registry.get(id).load().columns` for the live list.)" footnote. Cheapest fix.
- **Option B — implement the spec'd columns.** Wire up ERP age bands / density / median age (the source XLSX has them on a different sheet); wire up ABS_PIA Tables 2-9 for income breakdowns and gini. Honour the spec.

Either is fine. The third option — leave it as is — keeps the foot-gun primed for the next downstream consumer.

Happy to PR Option A (the trim) if you'd like a starting point. Or to write the `test_spec_matches_fetcher_columns` test that locks the door behind whichever resolution lands.
