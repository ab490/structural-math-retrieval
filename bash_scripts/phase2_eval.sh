#!/bin/bash
#SBATCH --job-name=phase2_eval
#SBATCH --output=../logs/phase2_eval_%j.out
#SBATCH --error=../logs/phase2_eval_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH -p gpu
#SBATCH -A
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4

echo "Job ID  : $SLURM_JOB_ID"
echo "Started : $(date)"

cd ../src

python -u phase2_eval.py

echo "Done: $(date)"
