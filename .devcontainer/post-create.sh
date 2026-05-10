#!/usr/bin/env bash
# First-attach setup for the aus-fuel-forecaster devcontainer.
# Runs once per container creation (Codespaces) or per rebuild (local Docker).
set -euo pipefail

cd "$(dirname "$0")/.."

# Heal the uv cache volume's ownership in case it was created by an older
# image revision (before the Dockerfile pre-created the dir with vscode
# ownership). Docker won't re-init a non-empty named volume, so the
# Dockerfile fix alone leaves pre-existing volumes broken — we chown
# defensively here. Cheap (one stat + chown of a normally-tiny tree on
# fresh volumes; trivial on populated ones because ownership is already
# correct).
if [ -d "$HOME/.cache/uv" ] && [ "$(stat -c %U "$HOME/.cache/uv")" != "vscode" ]; then
    echo ">>> repairing /home/vscode/.cache/uv ownership (named-volume init quirk)"
    sudo chown -R vscode:vscode "$HOME/.cache/uv"
fi

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
