# Structural Math Retrieval:  Can State-Space Models Encode Mathematical Structure Better Than Transformers

Do Structured State Space Models encode mathematical operation structure better than Transformer encoders when used as dense retrievers? We fine-tune Mamba-2 (130M, 370M, 1.3B), RoBERTa-base, RoBERTa-large, DeBERTa-v3-large, and DeBERTa-v2-xxlarge on 251,558 contrastive triplets from ARQMath-3, where hard negatives share the same topic but require a different operation. Models are evaluated on retrieval quality (pairwise accuracy, NDCG@10, MRR) and whether their embedding spaces organise geometrically by operation type (ARI, silhouette).

**Short answer:** at matched parameter counts, Transformers win. The architectural inductive-bias hypothesis is not supported.

Data and results: [OneDrive](https://indiana-my.sharepoint.com/:f:/g/personal/anobajaj_iu_edu/IgD8rvyoOtzJQ7sxXv_ey8qVAcfxyNMMTpBysuiK1A94Txg?e=BYm6Dg)

---

## Repo Structure

```
src/
  parse_arqmath.py        # XML -> arqmath_*.parquet
  build_triplets.py       # triplets_train/val/test.jsonl
  build_diagnostic_set.py # Phase 2 diagnostic sets
  tokenization_check.py   # tokenizer comparison (results/tokenization_check.json)
  inspect_data.py         # tag frequency analysis
  triplet_dataset.py      # shared dataset class
  train_utils.py          # shared pool, infonce_loss, StepLogger, optimizer_step
  constants.py            # paths
  transformer_finetune.py  # transformer training — RoBERTa, DeBERTa, any AutoModel
  transformer_eval.py      # Phase 1 eval for transformers
  ssm_finetune.py          # Mamba-2 (SSM) training
  ssm_eval.py              # Phase 1 eval for Mamba-2
  mamba_io.py             # Mamba checkpoint loading
  phase2_eval.py          # Phase 2 clustering (ARI, silhouette)
  compare_results.py      # cross-model comparison table

data/processed/           # triplets + diagnostic sets (gitignored, on OneDrive)
results/                  # checkpoints + eval JSON (gitignored, on OneDrive)
logs/                     # training .jsonl logs (gitignored)
```

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
bash setup_env.sh
```

This creates `.venv/` and installs all dependencies. CUDA wheels are pulled automatically on Linux. Data lives at `data/processed/` (download from OneDrive if not present). Checkpoints land in `results/<model-slug>-<pooling>/best_model/`.

---

## Reproducing

All commands run from the repo root via `uv run`. Training requires a CUDA GPU; 80 GB (A100-class) is needed for the 1.3B Mamba and DeBERTa-v2-xxlarge runs. Eval can run on CPU but is slow for large models.

**Raw data:** Step 1 expects `data/raw/arqmath/Posts.V1.3.xml` from the ARQMath-3 dataset (Math Stack Exchange posts). If you downloaded the already-processed triplets from OneDrive, place them in `data/processed/` and skip Step 1.

```bash
# 1. Data — parse ARQMath XML and build contrastive triplets
uv run python src/parse_arqmath.py
uv run python src/build_triplets.py
uv run python src/build_diagnostic_set.py

# 2. Training — checkpoints saved to results/<model-slug>-<pooling>/
# RoBERTa
uv run python src/transformer_finetune.py --model roberta-base  --pooling cls
uv run python src/transformer_finetune.py --model roberta-base  --pooling mean
uv run python src/transformer_finetune.py --model roberta-base  --pooling last_token
uv run python src/transformer_finetune.py --model roberta-large --pooling cls
uv run python src/transformer_finetune.py --model roberta-large --pooling mean
uv run python src/transformer_finetune.py --model roberta-large --pooling last_token

# DeBERTa (--gradient-checkpointing for xxlarge to fit in memory)
uv run python src/transformer_finetune.py --model microsoft/deberta-v3-large   --pooling cls        --batch-size 8 --grad-accum 8
uv run python src/transformer_finetune.py --model microsoft/deberta-v3-large   --pooling mean       --batch-size 8 --grad-accum 8
uv run python src/transformer_finetune.py --model microsoft/deberta-v3-large   --pooling last_token --batch-size 8 --grad-accum 8
uv run python src/transformer_finetune.py --model microsoft/deberta-v2-xxlarge --pooling cls        --batch-size 8 --grad-accum 8 --gradient-checkpointing

# Mamba-2 (reduce --batch-size / increase --grad-accum for larger models)
uv run python src/ssm_finetune.py --model state-spaces/mamba2-130m --pooling mean
uv run python src/ssm_finetune.py --model state-spaces/mamba2-130m --pooling last_token
uv run python src/ssm_finetune.py --model state-spaces/mamba2-370m --pooling mean
uv run python src/ssm_finetune.py --model state-spaces/mamba2-370m --pooling last_token
uv run python src/ssm_finetune.py --model state-spaces/mamba2-1.3b --pooling mean       --batch-size 8 --grad-accum 8
uv run python src/ssm_finetune.py --model state-spaces/mamba2-1.3b --pooling last_token --batch-size 8 --grad-accum 8

# 3. Phase 1 eval — retrieval metrics on the 31k-triplet test set
# RoBERTa-base (zero-shot baselines + fine-tuned)
uv run python src/transformer_eval.py --checkpoint roberta-base                              --pooling cls        --output-dir results/roberta-base-zeroshot-cls
uv run python src/transformer_eval.py --checkpoint roberta-base                              --pooling mean       --output-dir results/roberta-base-zeroshot-mean
uv run python src/transformer_eval.py --checkpoint roberta-base                              --pooling last_token --output-dir results/roberta-base-zeroshot-last_token
uv run python src/transformer_eval.py --checkpoint results/roberta-base-cls/best_model        --pooling cls
uv run python src/transformer_eval.py --checkpoint results/roberta-base-mean/best_model       --pooling mean
uv run python src/transformer_eval.py --checkpoint results/roberta-base-last_token/best_model --pooling last_token

# RoBERTa-large
uv run python src/transformer_eval.py --checkpoint roberta-large                              --pooling cls        --output-dir results/roberta-large-zeroshot-cls
uv run python src/transformer_eval.py --checkpoint roberta-large                              --pooling mean       --output-dir results/roberta-large-zeroshot-mean
uv run python src/transformer_eval.py --checkpoint roberta-large                              --pooling last_token --output-dir results/roberta-large-zeroshot-last_token
uv run python src/transformer_eval.py --checkpoint results/roberta-large-cls/best_model        --pooling cls
uv run python src/transformer_eval.py --checkpoint results/roberta-large-mean/best_model       --pooling mean
uv run python src/transformer_eval.py --checkpoint results/roberta-large-last_token/best_model --pooling last_token

# DeBERTa-v3-large
uv run python src/transformer_eval.py --checkpoint microsoft/deberta-v3-large                   --pooling cls        --output-dir results/deberta-v3-large-zeroshot-cls
uv run python src/transformer_eval.py --checkpoint microsoft/deberta-v3-large                   --pooling mean       --output-dir results/deberta-v3-large-zeroshot-mean
uv run python src/transformer_eval.py --checkpoint microsoft/deberta-v3-large                   --pooling last_token --output-dir results/deberta-v3-large-zeroshot-last_token
uv run python src/transformer_eval.py --checkpoint results/deberta-v3-large-cls/best_model        --pooling cls
uv run python src/transformer_eval.py --checkpoint results/deberta-v3-large-mean/best_model       --pooling mean
uv run python src/transformer_eval.py --checkpoint results/deberta-v3-large-last_token/best_model --pooling last_token

# DeBERTa-v2-xxlarge
uv run python src/transformer_eval.py --checkpoint results/deberta-v2-xxlarge-cls/best_model --pooling cls

# Mamba-2
uv run python src/ssm_eval.py --checkpoint results/mamba2-130m-mean/best_model        --pooling mean
uv run python src/ssm_eval.py --checkpoint results/mamba2-130m-last_token/best_model  --pooling last_token
uv run python src/ssm_eval.py --checkpoint results/mamba2-370m-mean/best_model        --pooling mean
uv run python src/ssm_eval.py --checkpoint results/mamba2-370m-last_token/best_model  --pooling last_token
uv run python src/ssm_eval.py --checkpoint results/mamba2-1.3b-mean/best_model        --pooling mean
uv run python src/ssm_eval.py --checkpoint results/mamba2-1.3b-last_token/best_model  --pooling last_token

# 4. Phase 2 — clustering eval across all saved checkpoints
uv run python src/phase2_eval.py
```

---

## Results

### Phase 1 - Retrieval (31,445-triplet test set)

**Small scale (~125–130M params)**

| Model | Pooling | P-Acc | NDCG@10 | MRR |
|---|---|---|---|---|
| RoBERTa-base | cls | 0.987 | 0.433 | 0.403 |
| RoBERTa-base | mean | **0.990** | **0.468** | **0.435** |
| RoBERTa-base | last_token | 0.987 | 0.435 | 0.404 |
| Mamba-2 130M | mean | 0.931 | 0.078 | 0.071 |
| Mamba-2 130M | last_token | 0.630 | 0.011 | 0.010 |

**Medium scale (~370–418M params)**
*DeBERTa-v3-large cut at ~33% of planned training; numbers are provisional.*

| Model | Pooling | P-Acc | NDCG@10 | MRR |
|---|---|---|---|---|
| DeBERTa-v3-large | cls | 0.981 | 0.181 | 0.161 |
| DeBERTa-v3-large | mean | 0.982 | 0.207 | 0.184 |
| DeBERTa-v3-large | last_token | 0.978 | 0.167 | 0.149 |
| Mamba-2 370M | mean | **0.982** | **0.321** | **0.295** |
| Mamba-2 370M | last_token | 0.978 | 0.262 | 0.238 |

RoBERTa-large (~304M) was fully trained but Phase 1 eval timed out on CPU; it appears in Phase 2 only.

**Large scale (~1.3–1.5B params)**
*DeBERTa-v2-xxlarge: only `cls` pooling completed within the 7-day wall time.*

| Model | Pooling | P-Acc | NDCG@10 | MRR |
|---|---|---|---|---|
| DeBERTa-v2-xxlarge | cls | **0.997** | **0.592** | **0.555** |
| Mamba-2 1.3B | mean | 0.994 | 0.525 | 0.489 |
| Mamba-2 1.3B | last_token | 0.994 | 0.504 | 0.466 |

### Phase 2 - Operation Clustering (120-example diagnostic sets)

`Broad ARI op` = ARI vs operation labels on real MSE questions. `Broad ARI top` = same vs topic labels. `Tight ARI op` = operation ARI on controlled synthetic problems. Zero-shot broad ARI(op) lies in [−0.012, 0.068] across all 18 zero-shot configurations - fine-tuning is what produces operation-aware geometry.

| Model | Pool. | Broad ARI op | Broad ARI top | Tight ARI op |
|---|---|---|---|---|
| RoBERTa-base | cls | 0.219 | 0.014 | 1.000 |
| RoBERTa-base | mean | 0.223 | 0.026 | 1.000 |
| RoBERTa-base | last_t | **0.244** | 0.063 | 1.000 |
| RoBERTa-large | cls | 0.198 | 0.055 | 1.000 |
| RoBERTa-large | mean | 0.132 | 0.010 | 1.000 |
| RoBERTa-large | last_t | 0.187 | 0.032 | 1.000 |
| DeBERTa-v3-large | cls | 0.103 | 0.002 | 0.736 |
| DeBERTa-v3-large | mean | 0.111 | 0.073 | 0.722 |
| DeBERTa-v3-large | last_t | 0.158 | 0.093 | 0.604 |
| Mamba-2 130M | mean | 0.124 | 0.196 | 0.654 |
| Mamba-2 130M | last_t | −0.002 | 0.000 | 0.573 |
| Mamba-2 370M | mean | 0.152 | 0.064 | 0.896 |
| Mamba-2 370M | last_t | 0.174 | 0.093 | **1.000** |
| Mamba-2 1.3B | mean | 0.188 | 0.212 | **1.000** |
| Mamba-2 1.3B | last_t | 0.144 | 0.055 | **1.000** |
| DeBERTa-v2-xxlarge | cls | 0.176 | **0.279** | 0.664 |

### Latency and Per-Parameter Efficiency

| Model | Tok/s | NDCG@10 | NDCG/100M params |
|---|---|---|---|
| RoBERTa-base (125M) | 318,356 | 0.468 | 0.374 |
| Mamba-2 130M | 204,402 | 0.078 | 0.060 |
| Mamba-2 370M | 77,008 | 0.321 | 0.087 |
| Mamba-2 1.3B | 26,257 | 0.525 | 0.040 |
| DeBERTa-v2-xxlarge cls | 16,706 | 0.592 | 0.039 |

Mamba-2 1.3B is ~12× slower than RoBERTa-base for a 12% NDCG@10 gain. Per-parameter efficiency peaks at 370M.

---




## Authors

Aishwarya Dhaigude, Anooshka Bajaj, Ravi Regulagedda - Indiana University Bloomington.
