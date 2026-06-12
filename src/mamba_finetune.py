"""
Fine-tuning script for Mamba-2 dense retriever on math triplets.

Uses InfoNCE contrastive loss with in-batch negatives + explicit hard negatives.
Each pooling strategy (mean, last_token) is a separate training run so that
training and evaluation use the same pooling — enabling clean ablation reporting.

Usage:
    python src/mamba_finetune.py \\
        --model state-spaces/mamba2-130m \\
        --pooling mean \\
        --train-file data/processed/triplets_train.jsonl \\
        --val-file   data/processed/triplets_val.jsonl \\
        --output-dir results/mamba2-130m-mean
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from constants import PROJECT_ROOT
from mamba_io import save_mamba
from triplet_dataset import TripletDataset, collate_fn


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor, strategy: str) -> torch.Tensor:
    """
    Reduce sequence of hidden states to a single embedding vector.

    hidden_states : [B, T, D]
    attention_mask: [B, T]  (1=real token, 0=padding)
    returns       : [B, D]
    """
    if strategy == "mean":
        mask = attention_mask.unsqueeze(-1).float()          # [B, T, 1]
        summed = (hidden_states * mask).sum(dim=1)           # [B, D]
        counts = mask.sum(dim=1).clamp(min=1e-9)             # [B, 1]
        return summed / counts

    elif strategy == "last_token":
        # Index of the last real (non-padding) token per sequence.
        # Right-padding is assumed (tokenizer.padding_side = "right").
        lengths = attention_mask.sum(dim=1) - 1              # [B]
        B, T, D = hidden_states.shape
        idx = lengths.view(B, 1, 1).expand(B, 1, D)
        return hidden_states.gather(dim=1, index=idx).squeeze(1)   # [B, D]

    else:
        raise ValueError(f"Unknown pooling strategy: {strategy!r}. Choose 'mean' or 'last_token'.")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def infonce_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    InfoNCE loss with in-batch negatives + explicit hard negative.

    All inputs are L2-normalised [B, D] embeddings.

    Candidates = [positives; negatives] → [2B, D].
    For anchor i: positive is candidate i (diagonal in the [B, 2B] sim matrix).
    Off-diagonal positives + all negatives serve as soft/hard negatives.
    """
    # [2B, D]
    candidates = torch.cat([positives, negatives], dim=0)
    # [B, 2B]
    logits = torch.matmul(anchors, candidates.T) / temperature
    # Labels: i-th anchor matches i-th positive (first B candidates)
    labels = torch.arange(anchors.size(0), device=anchors.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Embedding extraction (single forward pass for one text batch)
# ---------------------------------------------------------------------------

def embed(
    model: MambaLMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """Forward pass → pooled, L2-normalised embedding. Returns [B, D]."""
    # backbone returns the final hidden states tensor [B, T, D] directly
    # (skipping the LM head we don't need for retrieval).
    hidden = model.backbone(input_ids)
    emb = pool(hidden, attention_mask, pooling)
    return F.normalize(emb, p=2, dim=-1)   # [B, D]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class StepLogger:
    """Appends one JSON line per step to a JSONL file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def log(self, **kwargs) -> None:
        self.f.write(json.dumps(kwargs) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_val(
    model: MambaLMHeadModel,
    val_loader: DataLoader,
    device: torch.device,
    pooling: str,
    temperature: float,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            a_ids  = batch["anchor_input_ids"].to(device)
            a_mask = batch["anchor_attention_mask"].to(device)
            p_ids  = batch["positive_input_ids"].to(device)
            p_mask = batch["positive_attention_mask"].to(device)
            n_ids  = batch["negative_input_ids"].to(device)
            n_mask = batch["negative_attention_mask"].to(device)

            a_emb = embed(model, a_ids, a_mask, pooling)
            p_emb = embed(model, p_ids, p_mask, pooling)
            n_emb = embed(model, n_ids, n_mask, pooling)

            loss = infonce_loss(a_emb, p_emb, n_emb, temperature)
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = output_dir.name
    log_path = PROJECT_ROOT / "logs" / f"{run_name}_train.jsonl"
    logger = StepLogger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    print(f"Device: {device} | dtype: {dtype} | pooling: {args.pooling}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    # ------------------------------------------------------------------
    # Tokenizer & model
    # ------------------------------------------------------------------
    # state-spaces/mamba2-* HF repos ship no tokenizer files. Canonical tokenizer
    # for all Mamba-2 checkpoints is EleutherAI/gpt-neox-20b (matches the Mamba paper's
    # Pile training setup and the team's tokenization_check.py).
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # mamba-ssm's MambaLMHeadModel reads state-spaces' native config schema
    # (d_model, n_layer, ssm_cfg, ...) — unlike transformers.Mamba2Model which
    # expects HF schema and silently fell back to mamba2-2.8b defaults.
    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
    model.train()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = TripletDataset(args.train_file, tokenizer, args.max_len)
    val_ds   = TripletDataset(args.val_file,   tokenizer, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Total opt steps: {total_steps} | Warmup steps: {warmup_steps}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    global_step = 0
    accum_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            t0 = time.time()

            a_ids  = batch["anchor_input_ids"].to(device)
            a_mask = batch["anchor_attention_mask"].to(device)
            p_ids  = batch["positive_input_ids"].to(device)
            p_mask = batch["positive_attention_mask"].to(device)
            n_ids  = batch["negative_input_ids"].to(device)
            n_mask = batch["negative_attention_mask"].to(device)

            a_emb = embed(model, a_ids, a_mask, args.pooling)
            p_emb = embed(model, p_ids, p_mask, args.pooling)
            n_emb = embed(model, n_ids, n_mask, args.pooling)

            loss = infonce_loss(a_emb, p_emb, n_emb, args.temperature)
            # Scale by grad_accum so the effective loss magnitude is consistent
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss / args.grad_accum
                accum_loss = 0.0
                elapsed = time.time() - t0

                logger.log(
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=round(avg_loss, 6),
                    lr=scheduler.get_last_lr()[0],
                    elapsed_s=round(elapsed, 3),
                )

                if global_step % 100 == 0:
                    print(
                        f"Epoch {epoch} | Step {global_step} | "
                        f"loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                # ------ Validation ------
                if global_step % args.val_every == 0:
                    val_loss = run_val(model, val_loader, device, args.pooling, args.temperature)
                    print(f"  >> Val loss: {val_loss:.4f}")
                    logger.log(
                        epoch=epoch,
                        global_step=global_step,
                        val_loss=round(val_loss, 6),
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = output_dir / "best_model"
                        save_mamba(model, best_dir)
                        print(f"  >> New best ({val_loss:.4f}) — saved to {best_dir}")

        # End-of-epoch validation
        val_loss = run_val(model, val_loader, device, args.pooling, args.temperature)
        print(f"Epoch {epoch} done | Val loss: {val_loss:.4f}")
        logger.log(epoch=epoch, global_step=global_step, val_loss_epoch=round(val_loss, 6))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = output_dir / "best_model"
            save_mamba(model, best_dir)
            print(f"  >> New best ({val_loss:.4f}) — saved to {best_dir}")

    # ------------------------------------------------------------------
    # Save final checkpoint
    # ------------------------------------------------------------------
    final_dir = output_dir / "final_model"
    save_mamba(model, final_dir)
    print(f"Final model saved to {final_dir}")

    # Save run config alongside checkpoints
    config = vars(args)
    config["best_val_loss"] = best_val_loss
    config["total_steps"] = global_step
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.close()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Mamba-2 with InfoNCE contrastive loss")

    p.add_argument("--model",       default="state-spaces/mamba2-130m",
                   help="HuggingFace model ID or local path")
    p.add_argument("--pooling",     choices=["mean", "last_token"], required=True,
                   help="Pooling strategy — determines the model variant being trained")
    p.add_argument("--train-file",  default="data/processed/triplets_train.jsonl")
    p.add_argument("--val-file",    default="data/processed/triplets_val.jsonl")
    p.add_argument("--output-dir",  default=None,
                   help="Directory to save checkpoints. Defaults to results/<model-slug>-<pooling>/")
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--grad-accum",  type=int, default=4,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum")
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--max-len",     type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--warmup-ratio",type=float, default=0.1,
                   help="Fraction of total steps used for linear warmup")
    p.add_argument("--val-every",   type=int, default=500,
                   help="Validate every N optimizer steps")

    args = p.parse_args()

    if args.output_dir is None:
        model_slug = args.model.split("/")[-1]
        args.output_dir = str(PROJECT_ROOT / "results" / f"{model_slug}-{args.pooling}")

    # Resolve data paths relative to project root if not absolute
    for attr in ("train_file", "val_file"):
        path = Path(getattr(args, attr))
        if not path.is_absolute():
            setattr(args, attr, str(PROJECT_ROOT / path))

    return args


if __name__ == "__main__":
    train(parse_args())