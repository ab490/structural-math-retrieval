#!/bin/bash
#SBATCH --job-name=roberta_train
#SBATCH --output=../logs/roberta_train_%j.out
#SBATCH --error=../logs/roberta_train_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH -p gpu
#SBATCH -A
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8

MODELS=("roberta-base" "roberta-large")
POOLINGS=("cls" "mean" "last_token")

echo "Models   : ${MODELS[*]}"
echo "Poolings : ${POOLINGS[*]}"
echo "Job ID   : $SLURM_JOB_ID"
echo "Started  : $(date)"
cd ../src

for MODEL in "${MODELS[@]}"; do
    if [[ "$MODEL" == *"large"* ]]; then
        BATCH_SIZE=8
        GRAD_ACCUM=8
    else
        BATCH_SIZE=16
        GRAD_ACCUM=4
    fi

    echo ""
    echo "Model    : $MODEL (batch=$BATCH_SIZE, accum=$GRAD_ACCUM)"

    for POOLING in "${POOLINGS[@]}"; do
        echo ""
        echo "Training pooling: $POOLING"
        python -u roberta_finetune.py \
            --model "$MODEL" \
            --pooling "$POOLING" \
            --batch-size $BATCH_SIZE \
            --grad-accum $GRAD_ACCUM
        echo "Done with pooling: $POOLING at $(date)"
    done
done

echo ""
echo "All models and pooling strategies complete!"