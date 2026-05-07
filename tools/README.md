# tools/

Scripts that aren't part of the library. Mirrors the `abs-census-augmentor` pattern:

- **Run manually**, not as part of `make`/`pytest`.
- Allowed to make real network calls (unlike `tests/`).
- Allowed to write to the working tree.

## Scripts

### `make_notebooks.py`

One-shot generator for the three notebooks in `notebooks/`. Run once at project setup; do not re-run unless you genuinely want to discard notebook content (it errors out unless `--force` is passed).

```bash
python tools/make_notebooks.py            # safe: skip existing
python tools/make_notebooks.py --force    # overwrite (destructive)
```

## What goes here

Add scripts here that:
- Verify parsers against live ABS / RBA / data.nsw endpoints
- Generate one-off seed data
- Backfill caches from archived snapshots

Don't add:
- Anything that's part of the pipeline (it goes under `src/fuel_pred/`)
- Anything tested in CI (it goes under `tests/`)
