#!/bin/bash
#SBATCH --job-name=roberta_eval
#SBATCH --output=../logs/roberta_eval_%j.out
#SBATCH --error=../logs/roberta_eval_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH -p gpu
#SBATCH -A 
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4

MODELS=("roberta-base" "roberta-large")
POOLINGS=("cls" "mean" "last_token")

echo "Models   : ${MODELS[*]}"
echo "Poolings : ${POOLINGS[*]}"
echo "Job ID   : $SLURM_JOB_ID"
echo "Started  : $(date)"

cd ../src

for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG="${MODEL##*/}"

    echo ""
    echo "Model: $MODEL"

    # Zero-shot pretrained baseline
    for POOLING in "${POOLINGS[@]}"; do
        echo ""
        echo "Zero-shot baseline pooling: $POOLING"
        python -u roberta_eval.py \
            --checkpoint "$MODEL" \
            --pooling "$POOLING" \
            --output-dir "../results/${MODEL_SLUG}-zeroshot-${POOLING}"
        echo "Done zero-shot pooling: $POOLING at $(date)"
    done

    # Fine-tuned best checkpoint
    for POOLING in "${POOLINGS[@]}"; do
        CHECKPOINT="../results/${MODEL_SLUG}-${POOLING}/best_model"
        echo ""
        echo "Evaluating pooling: $POOLING"
        echo "Checkpoint: $CHECKPOINT"
        python -u roberta_eval.py \
            --checkpoint "$CHECKPOINT" \
            --pooling "$POOLING"
        echo "Done with pooling: $POOLING at $(date)"
    done
done

echo ""
echo "All models and pooling strategies complete: $(date)"