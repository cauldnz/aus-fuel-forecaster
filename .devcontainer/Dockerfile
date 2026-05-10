# Devcontainer for aus-fuel-forecaster.
#
# Base: Microsoft's official Python 3.11 devcontainer image (Debian Bookworm).
# Adds:
#   - uv (the project's package manager) via the official Astral image
#   - libgomp1 (LightGBM runtime)
#   - make (used by the project's Makefile)
#
# This image is consumed by both VS Code Dev Containers (local Docker) and
# GitHub Codespaces. Project deps install lazily via post-create.sh on
# first attach, so the image stays small and rebuilds are fast.

FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# OS-level packages the project needs at runtime:
#   - libgomp1: OpenMP runtime, required by lightgbm
#   - make:     used by the project's Makefile
# git, build-essential, ca-certificates, openssh-client are already in the
# base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Drop in uv from the official Astral image. Pinning to the latest stable
# release tag would be safer for reproducibility; using `latest` for now
# matches the project's `uv sync` cadence.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Sensible defaults for uv inside a devcontainer:
#   - Resolve the venv inside the workspace so VS Code's Python extension
#     can pick it up without configuration.
#   - Cache uv downloads in a persistent path (the workspace mount survives
#     container rebuilds in Codespaces but resets in local Docker — that's
#     fine; uv's cache is fast to repopulate).
ENV UV_PROJECT_ENVIRONMENT=/workspaces/aus-fuel-forecaster/.venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
