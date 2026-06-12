#!/bin/bash
# One-time setup: creates the 'aml-math' conda environment.
# Run from the login node before submitting any jobs:
#   bash setup_env.sh

set -e

module load conda/25.3.0

conda create -n aml-math python=3.11 -y

source activate aml-math

# PyTorch — let conda pick the right version and CUDA build for this system
conda install pytorch -c pytorch -c nvidia -y

# Remaining project dependencies
pip install -r requirements.txt

echo "Done. Activate with:"
echo "  module load conda/25.3.0 && source activate aml-math"
