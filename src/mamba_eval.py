"""
Evaluation script for fine-tuned Mamba-2 dense retriever.

Retrieval metrics (Phase 1):
  - Pairwise accuracy: sim(anchor, positive) > sim(anchor, negative)
  - Margin: mean and median of (sim_pos - sim_neg)
  - Mean positive cosine sim / mean negative cosine sim
  - NDCG@10: normalized discounted cumulative gain at rank 10 over full candidate pool
  - MRR: mean reciprocal rank of the positive over full candidate pool

Also tracks inference latency vs sequence length (per professor feedback).

Saves:
  <output-dir>/eval_results.json   — all metrics + run metadata

Usage:
    python src/mamba_eval.py \\
        --checkpoint results/mamba2-130m-mean/best_model \\
        --pooling mean \\
        --test-file data/processed/triplets_test.jsonl \\
        --output-dir results/mamba2-130m-mean
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer

from constants import PROJECT_ROOT
from mamba_io import load_mamba
from triplet_dataset import TripletDataset, collate_fn


# ---------------------------------------------------------------------------
# Pooling (same implementation as mamba_finetune.py)
# ---------------------------------------------------------------------------

def pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    elif strategy == "last_token":
        lengths = attention_mask.sum(dim=1) - 1          # [B]
        B, _, D = hidden_states.shape
        idx = lengths.view(B, 1, 1).expand(B, 1, D)
        return hidden_states.gather(dim=1, index=idx).squeeze(1)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy!r}")


def embed_batch(
    model: MambaLMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """Returns L2-normalised embeddings [B, D]."""
    with torch.no_grad():
        hidden = model.backbone(input_ids)
    emb = pool(hidden, attention_mask, pooling)
    return F.normalize(emb, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    model: MambaLMHeadModel,
    loader: DataLoader,
    device: torch.device,
    pooling: str,
    latency_log: list[dict],
) -> dict:
    """
    For each triplet: embed anchor, positive, negative.
    Compute pairwise accuracy and cosine similarity statistics.
    Latency is measured on the anchor pass (clean signal, no other work mixed in).
    """
    sim_pos_all: list[float] = []
    sim_neg_all: list[float] = []
    correct: list[int] = []

    model.eval()
    for batch in loader:
        a_ids  = batch["anchor_input_ids"].to(device)
        a_mask = batch["anchor_attention_mask"].to(device)
        p_ids  = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_ids  = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)

        # Time the anchor embedding only for a clean latency signal
        t0 = time.time()
        a_emb = embed_batch(model, a_ids, a_mask, pooling)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_ms = (time.time() - t0) * 1000
        mean_seq_len = float(a_mask.sum(dim=1).float().mean().item())
        tokens_total = int(a_mask.sum().item())
        latency_log.append({
            "batch_size": a_ids.size(0),
            "mean_seq_len": round(mean_seq_len, 1),
            "batch_ms": round(batch_ms, 2),
            "tokens_per_sec": round(tokens_total / (batch_ms / 1000 + 1e-9), 1),
        })

        p_emb = embed_batch(model, p_ids, p_mask, pooling)
        n_emb = embed_batch(model, n_ids, n_mask, pooling)

        sim_pos = (a_emb * p_emb).sum(dim=-1)   # [B] cosine sim (vectors are normalised)
        sim_neg = (a_emb * n_emb).sum(dim=-1)   # [B]

        sim_pos_all.extend(sim_pos.cpu().tolist())
        sim_neg_all.extend(sim_neg.cpu().tolist())
        correct.extend((sim_pos > sim_neg).int().cpu().tolist())

    arr_pos = np.array(sim_pos_all)
    arr_neg = np.array(sim_neg_all)
    arr_correct = np.array(correct)

    return {
        "pairwise_accuracy": float(arr_correct.mean()),
        "mean_positive_sim": float(arr_pos.mean()),
        "mean_negative_sim": float(arr_neg.mean()),
        "mean_margin": float((arr_pos - arr_neg).mean()),
        "median_margin": float(np.median(arr_pos - arr_neg)),
        "n_triplets": len(arr_correct),
    }


def summarise_latency(latency_log: list[dict]) -> dict:
    if not latency_log:
        return {}
    ms_per_token = [
        e["batch_ms"] / (e["mean_seq_len"] * e["batch_size"])
        for e in latency_log
    ]
    tps = [e["tokens_per_sec"] for e in latency_log]
    seq_lens = [e["mean_seq_len"] for e in latency_log]
    return {
        "mean_ms_per_token": round(float(np.mean(ms_per_token)), 4),
        "mean_tokens_per_sec": round(float(np.mean(tps)), 1),
        "median_tokens_per_sec": round(float(np.median(tps)), 1),
        "seq_len_range": [round(min(seq_lens), 1), round(max(seq_lens), 1)],
        "n_batches": len(latency_log),
        "per_batch": latency_log,   # kept for plotting latency vs seq_len
    }


# ---------------------------------------------------------------------------
# Ranking metrics (NDCG@10, MRR)
# ---------------------------------------------------------------------------

def compute_ranking_metrics(
    model: MambaLMHeadModel,
    dataset: TripletDataset,
    device: torch.device,
    pooling: str,
    batch_size: int,
) -> dict:
    """
    Compute NDCG@10 and MRR over a candidate pool built from the test set.

    Candidate pool = all unique positive texts in the test set (~one per question).
    For each anchor, its positive is the one relevant document; everything else
    in the pool is a distractor.  This approximates a real retrieval setting
    where the corpus is the set of candidate answers.

    NDCG@10 (binary relevance, one relevant doc):
        = 1 / log2(rank + 1)  if rank <= 10, else 0
        (IDCG@10 = 1 because the ideal is rank 1)

    MRR = mean of 1/rank across all queries.
    """
    model.eval()
    tokenizer = dataset.tokenizer
    max_len = dataset.max_len
    records = dataset.records

    # --- Build corpus of unique positive texts ---
    corpus_texts: list[str] = []
    corpus_idx_map: dict[str, int] = {}
    anchor_to_pos_idx: list[int] = []

    for rec in records:
        pos_text = rec["positive"]
        if pos_text not in corpus_idx_map:
            corpus_idx_map[pos_text] = len(corpus_texts)
            corpus_texts.append(pos_text)
        anchor_to_pos_idx.append(corpus_idx_map[pos_text])

    corpus_size = len(corpus_texts)
    print(f"  Ranking corpus size: {corpus_size} unique positives")

    # --- Embed entire corpus ---
    corpus_embs_list: list[torch.Tensor] = []
    for i in range(0, corpus_size, batch_size):
        batch_texts = corpus_texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        emb = embed_batch(model, input_ids, attention_mask, pooling)
        corpus_embs_list.append(emb.cpu())

    corpus_embs = torch.cat(corpus_embs_list, dim=0)  # [N, D] on CPU

    # --- Embed anchors in batches, compute rank of positive ---
    ndcg_scores: list[float] = []
    rr_scores: list[float] = []

    for i in range(0, len(records), batch_size):
        batch_recs = records[i : i + batch_size]
        anchor_texts = [r["anchor"] for r in batch_recs]
        enc = tokenizer(
            anchor_texts,
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        anchor_embs = embed_batch(model, input_ids, attention_mask, pooling)  # [B, D]

        # Cosine sim against full corpus: [B, N]
        sims = anchor_embs @ corpus_embs.to(device).T

        for j, rec_idx in enumerate(range(i, min(i + batch_size, len(records)))):
            pos_idx = anchor_to_pos_idx[rec_idx]
            sim_row = sims[j]                           # [N]
            pos_score = sim_row[pos_idx]
            # Rank = number of candidates with strictly higher score + 1
            rank = int((sim_row > pos_score).sum().item()) + 1

            ndcg_scores.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)
            rr_scores.append(1.0 / rank)

    return {
        "ndcg_at_10": round(float(np.mean(ndcg_scores)), 4),
        "mrr": round(float(np.mean(rr_scores)), 4),
        "corpus_size": corpus_size,
        "n_queries": len(records),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | pooling: {args.pooling}")

    # --- Load tokenizer & model ---
    # Tokenizer is the canonical Mamba-2 one (gpt-neox-20b); we don't save it
    # alongside the checkpoint since it never changes during training.
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --checkpoint can be either:
    #   (a) a local fine-tuned checkpoint dir (best_model/ or final_model/) — load via mamba_io
    #   (b) an HF model id (e.g. state-spaces/mamba2-130m) for zero-shot baselines
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir() and (checkpoint_path / "config.json").exists():
        print(f"Loading fine-tuned checkpoint from {checkpoint_path} ...")
        model = load_mamba(checkpoint_path, device=device, dtype=dtype)
    else:
        print(f"Loading pretrained checkpoint from HF: {args.checkpoint} ...")
        model = MambaLMHeadModel.from_pretrained(args.checkpoint, device=device, dtype=dtype)
    model.eval()

    # --- Dataset & loader ---
    test_ds = TripletDataset(args.test_file, tokenizer, args.max_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Test set: {len(test_ds)} triplets")

    # --- Pairwise retrieval metrics ---
    print("Computing pairwise retrieval metrics ...")
    latency_log: list[dict] = []
    metrics = compute_retrieval_metrics(model, test_loader, device, args.pooling, latency_log)
    print(f"  Pairwise accuracy : {metrics['pairwise_accuracy']:.4f}")
    print(f"  Mean margin       : {metrics['mean_margin']:.4f}")
    print(f"  Mean pos sim      : {metrics['mean_positive_sim']:.4f}")
    print(f"  Mean neg sim      : {metrics['mean_negative_sim']:.4f}")

    latency = summarise_latency(latency_log)
    print(f"  Tokens/sec (median): {latency.get('median_tokens_per_sec', 'N/A')}")

    # --- Ranking metrics (NDCG@10, MRR) ---
    print("Computing ranking metrics (NDCG@10, MRR) ...")
    ranking = compute_ranking_metrics(model, test_ds, device, args.pooling, args.batch_size)
    print(f"  NDCG@10           : {ranking['ndcg_at_10']:.4f}")
    print(f"  MRR               : {ranking['mrr']:.4f}")
    print(f"  Corpus size       : {ranking['corpus_size']}")

    # --- Save ---
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "pooling": args.pooling,
        "max_len": args.max_len,
        "test_file": str(args.test_file),
        "retrieval_metrics": metrics,
        "ranking_metrics": ranking,
        "latency": latency,
    }

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned Mamba-2 retriever")

    p.add_argument("--checkpoint",  required=True,
                   help="Path to saved model checkpoint (best_model/ or final_model/)")
    p.add_argument("--pooling",     choices=["mean", "last_token"], required=True,
                   help="Must match the pooling used during training")
    p.add_argument("--test-file",   default="data/processed/triplets_test.jsonl")
    p.add_argument("--output-dir",  default=None,
                   help="Where to write eval_results.json. Defaults to checkpoint's parent directory.")
    p.add_argument("--batch-size",  type=int, default=32,
                   help="Larger batch is fine at eval (no gradients stored)")
    p.add_argument("--max-len",     type=int, default=512)

    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent)

    for attr in ("test_file",):
        path = Path(getattr(args, attr))
        if not path.is_absolute():
            setattr(args, attr, str(PROJECT_ROOT / path))

    return args


if __name__ == "__main__":
    evaluate(parse_args())