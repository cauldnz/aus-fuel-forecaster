# Makefile for fuel-pred
# Each target maps to a phase in spec.md §12.
# Targets call `python -m fuel_pred.<module>` per CLAUDE.md conventions.

PYTHON := uv run python
PKG := fuel_pred

DATA_RAW := data/raw
DATA_INTERIM := data/interim
DATA_PROCESSED := data/processed
RESULTS := results
MODELS := models

# Span for v1 — adjust as needed
START_DATE := 2016-09-01
END_DATE := 2026-04-30

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  setup           uv sync (install deps)"
	@echo "  fetch           run all tier-1 fetchers"
	@echo "  fetch-tier2     run tier-2 fetchers (cash rate, asx, consumer confidence)"
	@echo "  clean-data      run cleaners → data/interim/"
	@echo "  enrich          call abs-census-augmentor → SA2 features"
	@echo "  features        build data/processed/features.parquet"
	@echo "  train           fit Models A and B → models/"
	@echo "  evaluate        produce results/comparison.md"
	@echo "  notebooks       execute all 3 notebooks"
	@echo "  all             full pipeline (fetch → ... → evaluate)"
	@echo "  test            pytest, hermetic only"
	@echo "  lint            ruff check"
	@echo "  format          ruff format"
	@echo "  typecheck       mypy strict on src/"
	@echo "  clean           remove data/interim, data/processed, models, results"

# ----------------------------- Setup -----------------------------

.PHONY: setup
setup:
	uv sync --extra dev --extra notebooks

# ----------------------------- Fetch -----------------------------

.PHONY: fetch fetch-tier1 fetch-tier2

fetch: fetch-tier1

fetch-tier1:
	$(PYTHON) -m $(PKG).fetch.fuelcheck --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/fuelcheck
	$(PYTHON) -m $(PKG).fetch.brent --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/brent.parquet
	$(PYTHON) -m $(PKG).fetch.audusd --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/audusd.parquet
	$(PYTHON) -m $(PKG).fetch.traffic --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/traffic

# Weather fetch needs the post-clean stations roster for lat/lons — runs as
# a separate target and requires `make clean-data` (or `make enrich`) first.
.PHONY: fetch-weather
fetch-weather:
	$(PYTHON) -m $(PKG).fetch.weather --stations $(DATA_INTERIM)/stations.parquet --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/weather

fetch-tier2:
	$(PYTHON) -m $(PKG).fetch.cash_rate --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/cash_rate.parquet
	$(PYTHON) -m $(PKG).fetch.asx200 --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/asx200.parquet
	$(PYTHON) -m $(PKG).fetch.inflation_expectations --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/inflation_expectations.parquet
	$(PYTHON) -m $(PKG).fetch.aip_tgp --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/aip_tgp.parquet

# ----------------------------- Build -----------------------------

.PHONY: clean-data enrich features

clean-data:
	$(PYTHON) -m $(PKG).clean.fuelcheck --in $(DATA_RAW)/fuelcheck --out $(DATA_INTERIM)/fuel_daily.parquet --stations-out $(DATA_INTERIM)/stations.parquet
	$(PYTHON) -m $(PKG).clean.traffic --in $(DATA_RAW)/traffic --out $(DATA_INTERIM)/traffic_daily.parquet --stations-out $(DATA_INTERIM)/traffic_stations.parquet

enrich: clean-data
	$(PYTHON) -m $(PKG).spatial.resolve_addrs --in $(DATA_INTERIM)/stations.parquet --out $(DATA_INTERIM)/stations.parquet
	$(PYTHON) -m $(PKG).build.enrich_census --in $(DATA_INTERIM)/stations.parquet --out $(DATA_INTERIM)/stations.parquet --data-dir $(DATA_RAW)

features: enrich
	$(PYTHON) -m $(PKG).spatial.nearest --stations $(DATA_INTERIM)/stations.parquet --counters $(DATA_INTERIM)/traffic_stations.parquet --out $(DATA_INTERIM)/station_to_counter.parquet
	$(PYTHON) -m $(PKG).build.panel_grid --stations $(DATA_INTERIM)/stations.parquet --fuel $(DATA_INTERIM)/fuel_daily.parquet --out $(DATA_INTERIM)/panel.parquet
	$(PYTHON) -m $(PKG).build.make_features --panel $(DATA_INTERIM)/panel.parquet --out $(DATA_PROCESSED)/features.parquet

# ----------------------------- Train + evaluate -----------------------------
#
# `train` and `evaluate` are file-target-aware on purpose. The data
# pipeline targets above (clean-data, enrich, features) are .PHONY
# because they cover several output files each + some are mutate-in-place;
# threading them through full file dependency tracking is brittle. Result:
# `make features` always re-runs everything upstream, which costs ~85 min
# of geocoding even when nothing changed.
#
# For training + eval, that's the wrong default — most iterations are
# "I changed the model code, re-fit on the existing features.parquet."
# So we pin train to depend on the actual features.parquet *file*. If
# the file exists, training fires immediately. If it doesn't, the
# fallback rule routes through `make features` to build it.
#
# To force a full pipeline rebuild: `make features train` explicitly.

.PHONY: train evaluate train-fresh evaluate-fresh

## Override knobs for `make train`. Usage:
##   make train                  -> spec §8.2 defaults (n_estimators=2000)
##   make train N_ESTIMATORS=800 -> cap boosting rounds for rough-iteration runs
##   make train LOG_PERIOD=1     -> XGBoost-style every-iter eval output
##
## To force a full pipeline rebuild (including ~85 min geocoding):
##   make features train
## or use the dedicated alias:
##   make train-fresh
TRAIN_OPTS := \
	$(if $(N_ESTIMATORS),--n-estimators $(N_ESTIMATORS),) \
	$(if $(LOG_PERIOD),--log-period $(LOG_PERIOD),)

# File-target fallback: when features.parquet is missing, defer to
# `make features` to build it. The recursive $(MAKE) is the standard
# pattern; doesn't loop because the second invocation finds the file
# either present (no-op) or fires the .PHONY chain to make it.
$(DATA_PROCESSED)/features.parquet:
	@echo ">>> $(@F) missing — running 'make features' to build it"
	$(MAKE) features

train: $(DATA_PROCESSED)/features.parquet
	$(PYTHON) -m $(PKG).train.train_models --features $(DATA_PROCESSED)/features.parquet --out $(MODELS) $(TRAIN_OPTS)

# Same pattern for the per-fold prediction parquets — they're what
# evaluate.compare actually reads. If they exist, `make evaluate` skips
# straight to the report renderer; if not, we route through training.
$(MODELS)/predictions_test_normal.parquet $(MODELS)/predictions_test_crisis.parquet: $(DATA_PROCESSED)/features.parquet
	@echo ">>> prediction parquets missing — running 'make train'"
	$(MAKE) train

evaluate: $(MODELS)/predictions_test_normal.parquet $(MODELS)/predictions_test_crisis.parquet
	$(PYTHON) -m $(PKG).evaluate.compare --features $(DATA_PROCESSED)/features.parquet --models $(MODELS) --out $(RESULTS)/comparison.md

# Force-rebuild aliases — explicit opt-in to re-run upstream stages.
train-fresh: features
	$(MAKE) train

evaluate-fresh: train-fresh
	$(MAKE) evaluate

# ----------------------------- Notebooks -----------------------------

.PHONY: notebooks
notebooks:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_explainability.ipynb

# ----------------------------- All -----------------------------

.PHONY: all
all: fetch fetch-tier2 features train evaluate notebooks

# ----------------------------- Quality -----------------------------

.PHONY: test lint format typecheck check
test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/

check: lint typecheck test

# ----------------------------- Clean -----------------------------

.PHONY: clean clean-all
clean:
	rm -rf $(DATA_INTERIM)/* $(DATA_PROCESSED)/* $(MODELS)/* $(RESULTS)/*
	@touch $(DATA_INTERIM)/.gitkeep $(DATA_PROCESSED)/.gitkeep $(MODELS)/.gitkeep $(RESULTS)/.gitkeep $(RESULTS)/shap/.gitkeep

clean-all: clean
	rm -rf $(DATA_RAW)/*
	@touch $(DATA_RAW)/.gitkeep
