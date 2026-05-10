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

# Workaround: the upstream Microsoft Python devcontainer image ships a
# yarn apt source at /etc/apt/sources.list.d/yarn.list whose signing
# key (62D54FD4003F6525) was rotated by the yarn project. The pre-baked
# key in the image no longer validates, so every `apt-get update` run
# during devcontainer build fails with:
#
#     The following signatures couldn't be verified because the public
#     key is not available: NO_PUBKEY 62D54FD4003F6525
#
# Removing the yarn source ahead of any apt-get traffic sidesteps the
# issue. Nothing in this project uses yarn; if a future change needs it,
# install via a devcontainer feature or `npm install -g yarn` rather
# than reinstating this apt source.
#
# Same fix as cauldnz/abs-census-augmentor#27 — both projects sit on
# the same Microsoft base image. Delete this RUN once Microsoft refreshes
# the image with the new yarn key
# (track via https://github.com/devcontainers/images/issues).
RUN rm -f /etc/apt/sources.list.d/yarn.list

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
