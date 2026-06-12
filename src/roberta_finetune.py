"""
Fine-tuning script for RoBERTa dense retriever on math triplets.

Uses InfoNCE contrastive loss with in-batch negatives + explicit hard negatives.
Each pooling strategy (cls, mean, last_token) is a separate training run so that
training and evaluation use the same pooling — enabling clean ablation reporting.

Usage:
    python src/roberta_finetune.py \\
        --model roberta-base \\
        --pooling mean \\
        --train-file data/processed/triplets_train.jsonl \\
        --val-file   data/processed/triplets_val.jsonl \\
        --output-dir results/roberta-base-cls
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
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from constants import PROJECT_ROOT
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
    if strategy == "cls":
        # [CLS] token is always at position 0 for RoBERTa.
        return hidden_states[:, 0, :]                        # [B, D]

    elif strategy == "mean":
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
        raise ValueError(f"Unknown pooling strategy: {strategy!r}. Choose 'cls', 'mean', or 'last_token'.")


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
    candidates = torch.cat([positives, negatives], dim=0)   # [2B, D]
    logits = torch.matmul(anchors, candidates.T) / temperature  # [B, 2B]
    labels = torch.arange(anchors.size(0), device=anchors.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Embedding extraction (single forward pass for one text batch)
# ---------------------------------------------------------------------------

def embed(
    model: AutoModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """Forward pass → pooled, L2-normalised embedding. Returns [B, D]."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state                          # [B, T, D]
    emb = pool(hidden, attention_mask, pooling)
    return F.normalize(emb, p=2, dim=-1)                   # [B, D]


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
    model: AutoModel,
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


def _optimizer_step(model, optimizer, scheduler):
    """Clip gradients, step optimizer and scheduler, then zero gradients."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "train_log.jsonl"
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(args.model, dtype=dtype)
    model = model.to(device)
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
    best_epoch = 0
    best_step = 0
    global_step = 0
    accum_loss = 0.0
    no_improve_epochs = 0
    stopped_epoch = args.epochs  # updated if early stopping fires

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        epoch_train_loss = 0.0
        epoch_opt_steps = 0
        accum_t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(pbar, start=1):
            # Reset accumulation window timer at the start of each window
            if (step - 1) % args.grad_accum == 0:
                accum_t0 = time.time()

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
                elapsed = time.time() - accum_t0
                _optimizer_step(model, optimizer, scheduler)
                global_step += 1

                avg_loss = accum_loss / args.grad_accum
                accum_loss = 0.0
                epoch_train_loss += avg_loss
                epoch_opt_steps += 1

                logger.log(
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=round(avg_loss, 6),
                    lr=scheduler.get_last_lr()[0],
                    elapsed_s=round(elapsed, 3),
                )

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=global_step,
                )

                # ------ Mid-epoch validation ------
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
                        best_epoch = epoch
                        best_step = global_step
                        best_dir = output_dir / "best_model"
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        print(f"  >> New best ({val_loss:.4f}) — saved to {best_dir}")

        # Flush any remaining accumulated gradients at end of epoch
        if step % args.grad_accum != 0:
            elapsed = time.time() - accum_t0
            _optimizer_step(model, optimizer, scheduler)
            global_step += 1

        # ------ End-of-epoch validation ------
        val_loss = run_val(model, val_loader, device, args.pooling, args.temperature)
        avg_epoch_train_loss = epoch_train_loss / max(epoch_opt_steps, 1)
        print(
            f"Epoch {epoch} done | "
            f"Train loss: {avg_epoch_train_loss:.4f} | Val loss: {val_loss:.4f}"
        )
        logger.log(
            epoch=epoch,
            global_step=global_step,
            epoch_train_loss=round(avg_epoch_train_loss, 6),
            val_loss_epoch=round(val_loss, 6),
        )

        # ------ Save epoch checkpoint ------
        epoch_dir = output_dir / f"checkpoint_epoch_{epoch:02d}"
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"  >> Epoch checkpoint saved to {epoch_dir}")

        # ------ Best model + early stopping ------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_step = global_step
            no_improve_epochs = 0
            best_dir = output_dir / "best_model"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  >> New best ({val_loss:.4f}) — saved to {best_dir}")
        else:
            no_improve_epochs += 1
            print(f"  >> No improvement for {no_improve_epochs}/{args.patience} epochs")
            if no_improve_epochs >= args.patience:
                stopped_epoch = epoch
                print(f"Early stopping triggered after epoch {epoch} (patience={args.patience})")
                break

    # ------------------------------------------------------------------
    # Save final checkpoint
    # ------------------------------------------------------------------
    final_dir = output_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")

    # Save run config alongside checkpoints
    config = vars(args)
    config["best_val_loss"] = best_val_loss
    config["best_epoch"] = best_epoch
    config["best_step"] = best_step
    config["total_steps"] = global_step
    config["stopped_epoch"] = stopped_epoch
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.close()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune RoBERTa with InfoNCE contrastive loss")

    p.add_argument("--model",       default="roberta-base",
                   help="HuggingFace model ID or local path")
    p.add_argument("--pooling",     choices=["cls", "mean", "last_token"], required=True,
                   help="Pooling strategy — determines the model variant being trained")
    p.add_argument("--train-file",  default="data/processed/triplets_train.jsonl")
    p.add_argument("--val-file",    default="data/processed/triplets_val.jsonl")
    p.add_argument("--output-dir",  default=None,
                   help="Directory to save checkpoints. Defaults to results/<model-slug>-<pooling>/")
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--grad-accum",  type=int, default=4,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum")
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--epochs",      type=int, default=15)
    p.add_argument("--patience",    type=int, default=3,
                   help="Early stopping patience in epochs. Training stops if val loss does not improve for this many consecutive epochs.")
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