#!/bin/bash
#SBATCH --job-name=mamba_train
#SBATCH --output=../logs/mamba_train_%j.out
#SBATCH --error=../logs/mamba_train_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH -p gpu
#SBATCH -A
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8

# Usage:
#   MODEL=state-spaces/mamba2-130m POOLING=mean sbatch mamba_train.sh
#   MODEL=state-spaces/mamba2-130m POOLING=last_token sbatch mamba_train.sh
#   MODEL=state-spaces/mamba2-1.3b POOLING=mean BATCH_SIZE=8 GRAD_ACCUM=8 sbatch mamba_train.sh
#   MODEL=state-spaces/mamba2-1.3b POOLING=last_token BATCH_SIZE=8 GRAD_ACCUM=8 sbatch mamba_train.sh
#
# Progress:
#   All print() output goes to ../logs/mamba_train_<jobid>.out  (tail -f to monitor)
#   Structured step-level loss log: ../logs/<run-name>_train.jsonl  (flushed after every step)

module load conda/25.3.0
source activate aml-math
module load cudatoolkit/12.6
# Conda env ships a newer libstdc++ than the system / Cray PrgEnv default;
# put it first so torch's compiled extensions (optree, mamba_ssm kernels) find
# the GLIBCXX symbols they were built against.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

MODEL=${MODEL:-state-spaces/mamba2-130m}
POOLING=${POOLING:-mean}
# batch_size * grad_accum = effective batch (InfoNCE benefits from a stable
# effective batch across model sizes). For 1.3b we drop bs to 8 and bump
# grad_accum to 8 so activations fit in 80GB while keeping eff batch = 64.
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-4}

echo "Model      : $MODEL"
echo "Pooling    : $POOLING"
echo "Batch size : $BATCH_SIZE"
echo "Grad accum : $GRAD_ACCUM"
echo "Job ID     : $SLURM_JOB_ID"
echo "Started    : $(date)"

cd ../src
# -u = unbuffered stdout so output appears in .out immediately even if job is killed
python -u mamba_finetune.py \
    --model "$MODEL" \
    --pooling "$POOLING" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM"

echo "Finished: $(date)"