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
# TODO(phase2): weather fetch needs station lat/lons from clean.fuelcheck — re-enable after Phase 2.
#	$(PYTHON) -m $(PKG).fetch.weather --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/weather

fetch-tier2:
	$(PYTHON) -m $(PKG).fetch.cash_rate --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/cash_rate.parquet
	$(PYTHON) -m $(PKG).fetch.asx200 --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/asx200.parquet
	$(PYTHON) -m $(PKG).fetch.consumer_confidence --start $(START_DATE) --end $(END_DATE) --out $(DATA_RAW)/consumer_confidence.parquet

# ----------------------------- Build -----------------------------

.PHONY: clean-data enrich features

clean-data:
	$(PYTHON) -m $(PKG).clean.fuelcheck --in $(DATA_RAW)/fuelcheck --out $(DATA_INTERIM)/fuel_daily.parquet --stations-out $(DATA_INTERIM)/stations.parquet
	$(PYTHON) -m $(PKG).clean.traffic --in $(DATA_RAW)/traffic --out $(DATA_INTERIM)/traffic_daily.parquet --stations-out $(DATA_INTERIM)/traffic_stations.parquet

enrich: clean-data
	$(PYTHON) -m $(PKG).spatial.resolve_addrs --in $(DATA_INTERIM)/stations.parquet --out $(DATA_INTERIM)/stations.parquet
	$(PYTHON) -m $(PKG).build.enrich_census --in $(DATA_INTERIM)/stations.parquet --out $(DATA_INTERIM)/stations.parquet --seifa-cache $(DATA_RAW)/seifa

features: enrich
	$(PYTHON) -m $(PKG).spatial.nearest --stations $(DATA_INTERIM)/stations.parquet --counters $(DATA_INTERIM)/traffic_stations.parquet --out $(DATA_INTERIM)/station_to_counter.parquet
	$(PYTHON) -m $(PKG).build.panel_grid --stations $(DATA_INTERIM)/stations.parquet --fuel $(DATA_INTERIM)/fuel_daily.parquet --out $(DATA_INTERIM)/panel.parquet
	$(PYTHON) -m $(PKG).build.make_features --panel $(DATA_INTERIM)/panel.parquet --out $(DATA_PROCESSED)/features.parquet

# ----------------------------- Train + evaluate -----------------------------

.PHONY: train evaluate

train: features
	$(PYTHON) -m $(PKG).train.train_models --features $(DATA_PROCESSED)/features.parquet --out $(MODELS)

evaluate: train
	$(PYTHON) -m $(PKG).evaluate.compare --features $(DATA_PROCESSED)/features.parquet --models $(MODELS) --out $(RESULTS)/comparison.md

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
