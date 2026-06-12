#!/bin/bash
# One-time setup: creates the project virtual environment with uv.
# Run from the repo root:
#   bash setup_env.sh

set -e

# Install uv if not already present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi

# Create .venv and install all dependencies (reads pyproject.toml)
uv sync

echo "Done. Activate the environment with:"
echo "  source .venv/bin/activate"
