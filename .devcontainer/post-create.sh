#!/usr/bin/env bash
# First-attach setup for the aus-fuel-forecaster devcontainer.
# Runs once per container creation (Codespaces) or per rebuild (local Docker).
set -euo pipefail

cd "$(dirname "$0")/.."

echo ">>> uv sync (installing project deps)"
uv sync --extra dev --extra notebooks

echo ">>> verifying the project imports + pytest discovers tests"
uv run python -c "import fuel_pred; print('fuel_pred OK', fuel_pred.__version__)"
uv run pytest --collect-only -q | tail -5

echo
echo "Devcontainer ready. Useful commands:"
echo "  make test           # hermetic suite"
echo "  make lint typecheck # ruff + mypy"
echo "  make fetch          # Tier-1 fetchers (writes data/raw/)"
echo "  make features       # full pipeline → data/processed/features.parquet"
echo "  uv run jupyter lab  # notebooks/"
