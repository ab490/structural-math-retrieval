#!/bin/bash
#SBATCH --job-name=mamba_eval
#SBATCH --output=../logs/mamba_eval_%j.out
#SBATCH --error=../logs/mamba_eval_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH -p gpu
#SBATCH -A 
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4

# Usage:
#   MODEL=state-spaces/mamba2-130m sbatch mamba_eval.sh
#   MODEL=state-spaces/mamba2-1.3b sbatch mamba_eval.sh
#
# Per submission, evaluates both poolings (mean, last_token) for the given MODEL,
# in two phases:
#   1. Zero-shot baseline (pretrained model, no fine-tuning)
#        -> results/<model-slug>-zeroshot-<pooling>/eval_results.json
#   2. Fine-tuned best_model
#        -> results/<model-slug>-<pooling>/eval_results.json
#
# Progress: ../logs/mamba_eval_<jobid>.out

module load conda/25.3.0
source activate aml-math
module load cudatoolkit/12.6
# Conda env ships a newer libstdc++ than the system / Cray PrgEnv default;
# put it first so torch's compiled extensions (optree, mamba_ssm kernels) find
# the GLIBCXX symbols they were built against.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

MODEL=${MODEL:-state-spaces/mamba2-130m}
POOLINGS=("mean" "last_token")
MODEL_SLUG=$(basename "$MODEL")

echo "Model      : $MODEL"
echo "Model slug : $MODEL_SLUG"
echo "Poolings   : ${POOLINGS[*]}"
echo "Job ID     : $SLURM_JOB_ID"
echo "Started    : $(date)"

cd ../src

# --- Phase 1: zero-shot baseline ---
for POOLING in "${POOLINGS[@]}"; do
    echo ""
    echo "=== Zero-shot baseline | pooling: $POOLING ==="
    python -u mamba_eval.py \
        --checkpoint "$MODEL" \
        --pooling "$POOLING" \
        --output-dir "../results/${MODEL_SLUG}-zeroshot-${POOLING}"
    echo "Done zero-shot $POOLING at $(date)"
done

# --- Phase 2: fine-tuned best_model ---
for POOLING in "${POOLINGS[@]}"; do
    CHECKPOINT="../results/${MODEL_SLUG}-${POOLING}/best_model"
    echo ""
    echo "=== Fine-tuned | pooling: $POOLING ==="
    echo "Checkpoint: $CHECKPOINT"
    python -u mamba_eval.py \
        --checkpoint "$CHECKPOINT" \
        --pooling "$POOLING"
    echo "Done fine-tuned $POOLING at $(date)"
done

echo ""
echo "All evals complete: $(date)"