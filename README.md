# NSW Fuel Price Prediction

A regression model that forecasts daily retail fuel prices at NSW service stations, used to demonstrate that augmenting per-station feature sets with SA2-level Australian Census demographics (via [`abs-census-augmentor`](https://github.com/cauldnz/abs-census-augmentor)) measurably improves predictive performance.

The headline experiment compares two LightGBM models with identical pipelines except for one feature block:

- **Model A** — lag, upstream commodity, calendar, demand context, station-static, weather features
- **Model B** — Model A + 10 SA2-level Census demographic features

The "result" of the project is the lift from Model A to Model B on a held-out future test set, segmented by metro/regional, brand, fuel type, and SEIFA quintile, with SHAP explanations of the top SA2 features and their interactions with calendar features.

See [`spec.md`](./spec.md) for the full design specification — it is the source of truth for this repository.

## Status

🚧 v1 implementation in progress — see [`spec.md` §12](./spec.md) for phase tracking.

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- ~5 GB free disk for cached raw data
- Network access for first-run fetches (raw data is then cached locally)

## Quickstart

### In VS Code (recommended)

A [devcontainer](.devcontainer/devcontainer.json) ships with the repo. With the **Dev Containers** extension installed, opening the project will offer "Reopen in Container" — accept it. The container provides Python 3.11, `uv`, all project deps, and the matching VS Code extensions (Pylance, Ruff, Mypy, Jupyter). First attach takes ~1 min for `uv sync`; subsequent attaches are instant. Works identically in **GitHub Codespaces**.

### On the host

```bash
# Setup
uv sync

# Run the full pipeline (first run takes a while — fetches all raw data)
make all

# Or run individual phases
make fetch         # download all tier-1 sources
make clean-data    # produce interim parquets
make enrich        # call abs-census-augmentor
make features      # build the final feature matrix
make train         # fit Models A and B
make evaluate      # produce results/comparison.md
make notebooks     # execute all three notebooks

# Development
make test          # pytest, hermetic only
make lint          # ruff check
make format        # ruff format
make typecheck     # mypy strict on src/
```

## Repository layout

```
.
├── spec.md                  # Source of truth — design specification
├── CLAUDE.md                # Conventions for AI-agent contributors
├── README.md                # This file
├── pyproject.toml           # uv-managed deps
├── Makefile                 # Task runner
├── data/
│   ├── raw/                 # gitignored — cached fetches
│   ├── interim/             # gitignored — cleaned intermediates
│   ├── processed/           # gitignored — final feature matrix
│   └── static/              # checked in — manually-curated tables
├── src/fuel_pred/           # Library code
│   ├── fetch/               # One module per data source
│   ├── clean/               # Cleaning + dedupe per source
│   ├── spatial/             # G-NAF resolution, nearest-neighbour joins
│   ├── build/               # Census enrichment, panel build, feature engineering
│   ├── train/               # Model A and Model B fitting
│   └── evaluate/            # Metrics and comparison report
├── tests/                   # Hermetic pytest suite
├── notebooks/               # 01_eda, 02_modeling, 03_explainability
├── models/                  # gitignored — trained model artefacts
└── results/                 # comparison.md, SHAP plots
```

## Data sources

All data sources are documented in [`spec.md` §5](./spec.md). Tier 1 sources (required for the headline result):

- NSW FuelCheck Price History (target variable)
- Brent crude futures
- AUD/USD exchange rate (RBA F11.1)
- NSW Roads Traffic Volume Counts
- Australian public holidays (`python-holidays`)
- NSW school terms (manual table)
- Open-Meteo historical weather

Tier 2 sources (additive features, fail-soft if unavailable):

- AIP Terminal Gate Prices
- RBA cash rate (F1.1)
- ASX 200
- ANZ-Roy Morgan Consumer Confidence

The augmentation block uses 10 SA2-level variables from the 2021 ABS Census GCP DataPack plus SEIFA IRSD scores. See [`spec.md` §5.4](./spec.md) for the exact list.

## Methodological notes

A few design choices worth surfacing — see `spec.md` for full rationale:

- **Hyperparameters fixed in v1.** Model A and Model B use identical LightGBM configs. Tuning is deferred so the comparison is apples-to-apples.
- **Identical training rows** across A and B. Only rows where every Model B column is non-null are used in either model — this prevents Model B from looking better just because it has a smaller, easier row set.
- **Petrol cycle is not hand-encoded.** It should emerge from the lag block + day-of-week. There's a sanity check for this in `notebooks/01_eda.ipynb`.
- **MAE objective, not RMSE.** Fuel prices have spike days during the 2026 crisis that an L2 loss would over-fit; loss-and-metric are deliberately aligned.
- **The 2026 fuel crisis is held out as a separate test fold** — reported alongside the headline metrics as an out-of-distribution check, but kept out of the headline numbers so they're comparable to a pre-crisis baseline.

## License

MIT — see [`LICENSE`](./LICENSE).
