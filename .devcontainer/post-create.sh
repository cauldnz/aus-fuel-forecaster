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

# Same heal for the .venv volume. A freshly-created Podman/Docker named
# volume mounts as root:root mode 755 — fine for read, but uv can't create
# the venv inside it. Without this chown, `uv sync` fails with
#   "failed to open file `.venv/CACHEDIR.TAG`: Permission denied (os error 13)"
# on the first attach after a clean rebuild. Cheap on populated volumes
# (ownership already correct → no-op recurse).
VENV_DIR="$(pwd)/.venv"
if [ -d "$VENV_DIR" ] && [ "$(stat -c %U "$VENV_DIR")" != "vscode" ]; then
    echo ">>> repairing $VENV_DIR ownership (named-volume init quirk)"
    sudo chown -R vscode:vscode "$VENV_DIR"
fi

echo ">>> uv sync (installing project deps)"
uv sync --extra dev --extra notebooks

# Add `pip` to the venv. uv intentionally doesn't install pip (it
# replaces the package manager entirely), but VS Code's Jupyter
# extension probes for `pip` alongside `ipykernel` when first opening
# a notebook and pops a "Running cells with ... requires the ipykernel
# and pip package" prompt if pip is missing. Adding pip here makes
# Run All Just Work on first attach.
echo ">>> installing pip into the venv (for VS Code Jupyter compatibility)"
uv pip install pip --quiet

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
