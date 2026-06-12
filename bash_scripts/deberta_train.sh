#!/bin/bash
#SBATCH --job-name=deberta_train
#SBATCH --output=../logs/deberta_train_%j.out
#SBATCH --error=../logs/deberta_train_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH -p gpu
#SBATCH -A 
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8

MODELS=("microsoft/deberta-v3-large" "microsoft/deberta-v2-xxlarge")

POOLINGS=("cls" "mean" "last_token")

echo "Models   : ${MODELS[*]}"
echo "Poolings : ${POOLINGS[*]}"
echo "Job ID   : $SLURM_JOB_ID"

cd ../src

BATCH_SIZE=8
GRAD_ACCUM=8

for MODEL in "${MODELS[@]}"; do
    [[ "$MODEL" == *"xxlarge"* ]] && GRAD_CKPT="--gradient-checkpointing" || GRAD_CKPT=""

    echo ""
    echo "Model    : $MODEL"
    echo "Batch    : $BATCH_SIZE  Grad accum: $GRAD_ACCUM  Grad ckpt: ${GRAD_CKPT:-off}"

    for POOLING in "${POOLINGS[@]}"; do
        echo ""
        echo "Training pooling: $POOLING"
        python -u deberta_finetune.py \
            --model "$MODEL" \
            --pooling "$POOLING" \
            --batch-size "$BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            $GRAD_CKPT
        echo "Done with pooling: $POOLING"
    done
done

echo ""
echo "All models and pooling strategies complete!"