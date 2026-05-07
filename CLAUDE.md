# CLAUDE.md

Conventions for AI-agent contributors (Claude Code, Cursor, etc.) working in this repository. Human contributors should follow them too.

## Source of truth

`spec.md` is the design specification and the source of truth. If code disagrees with `spec.md`, the code is wrong. Design changes are made by editing `spec.md` first, then code.

If you find yourself wanting to deviate from `spec.md`, **stop and propose the change in the PR description rather than guessing**. Ambiguity resolved silently is the most expensive kind of bug in this repo.

## Workflow

1. Pick a phase from `spec.md` §12. One phase per PR. Don't bundle.
2. Read the relevant section of `spec.md` end-to-end before writing code.
3. Implement. Add tests. Run `make lint typecheck test` until green.
4. Open a PR. Reference the spec section. Note any deviations in the description.
5. If a deviation requires a spec change, edit `spec.md` in the same PR.

## Code conventions

- **Python 3.11+**. Use modern syntax (`from __future__ import annotations`, `|` unions, `match`).
- **Type hints required** on all public functions. Validated by mypy strict on `src/`.
- **Lint and format with ruff.** No exceptions.
- **No `print`.** Use `logging` from stdlib. Each module gets its own logger via `logger = logging.getLogger(__name__)`.
- **No hard-coded paths.** All paths come from `fuel_pred.config`. If you need a new path, add it to `config.py`.
- **Each module gets a CLI.** Top-level modules under `src/fuel_pred/{fetch,clean,spatial,build,train,evaluate}/` have a `__main__` block:
  ```bash
  python -m fuel_pred.fetch.brent --start 2016-09-01 --end 2026-04-30 --out data/raw/brent.parquet
  ```
  Arguments are explicit. The Makefile composes these into pipeline targets.
- **Parquet for tabular data.** Use `pyarrow` engine, `compression="zstd"`. Don't write CSVs anywhere except `data/static/`.
- **Pandas, not Polars** in v1. We're matching `abs-census-augmentor`'s API surface.

## Testing

- **Hermetic by default.** Tests in `tests/` must not hit the network. HTTP is mocked with `responses`. File IO uses `tmp_path`.
- **Real-network integration** tests live in a separate `tools/` directory (mirroring `abs-census-augmentor`). They are opt-in, run manually, and never gate CI.
- **Test-after-implement is fine** for fetchers and cleaners (the implementation logic is mostly schema-handling).
- **Test-first for feature engineering.** Each function in `make_features.py` has a unit test that pins down its lag/window/null-handling behaviour. Bugs in feature engineering are silent and devastating.
- **Do not commit large fixtures.** Use small synthetic DataFrames in tests, ideally constructed inline.

## Caching philosophy

Every fetcher writes to `data/raw/<source>/` with a deterministic filename (date-stamped or content-hashed). Re-runs should be cheap — if the cache file exists and is newer than X days, skip the fetch unless `--force` is passed.

The cache is the user's responsibility to invalidate. Don't auto-prune.

## Network etiquette

- Respect rate limits. RBA, Open-Meteo, and data.nsw.gov.au are generous but not unlimited.
- Set a meaningful `User-Agent` header on all requests: `fuel-pred/0.1 (<your-contact>)`.
- Use `tenacity` for retries, exponential backoff, max 5 attempts.
- For fetchers that paginate (CKAN), check the dataset's `total` field and assert you got everything.

## Logging conventions

- INFO: high-level pipeline progress, source URLs, row counts, cache hits/misses
- DEBUG: per-row or per-batch detail, only useful when debugging
- WARNING: data quality issues that don't block progress (a station that failed to geocode, a missing day in a series)
- ERROR: hard failures that aborted the run

Every fetcher logs the source URL and final row count at INFO. Every cleaner logs the input row count, output row count, and dedupe ratio at INFO.

## When you don't know

Three options, in order of preference:

1. **Check `spec.md`.** It might already answer the question.
2. **Open a draft PR with the question in the description**, leaving the code blocked at the unclear point. The maintainer responds.
3. **Make the conservative choice and call it out** in the PR description as a `TODO(spec)` comment in the code.

Do *not*:
- Silently invent a behaviour and assume it's right.
- Ship code that depends on data sources that aren't in `spec.md` §5.
- Bundle hyperparameter changes with feature additions.
- Touch files outside the phase you're working on.

## File ownership map

If a file isn't listed here, it's owned by `spec.md` §10 (repo layout). Edit at your own risk.

| File / directory | Owner | Notes |
|---|---|---|
| `spec.md` | maintainer + agent (any) | Source of truth. Changes require justification in PR. |
| `CLAUDE.md` | maintainer | Convention changes only via maintainer review. |
| `README.md` | maintainer + agent | Update when public-facing behaviour changes. |
| `data/static/brand_aliases.csv` | maintainer + agent | Append rows when new brands appear in FuelCheck. Don't reorder. |
| `data/static/nsw_school_terms.csv` | maintainer | Update annually when NSW Education publishes new dates. |
| `data/static/crisis_events.csv` | informational only | Not used as features in v1. Curate as historical record. |
| `pyproject.toml` | maintainer + agent | Add deps freely; remove deps via PR with justification. |

## Phase status

Tracked in `spec.md` §12. When you complete a phase, mark it as `✅ done` in the spec and reference the merged PR.
