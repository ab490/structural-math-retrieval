"""
Fine-tuning script for Mamba-2 dense retriever on math triplets.

Uses InfoNCE contrastive loss with in-batch negatives + explicit hard negatives.
Each pooling strategy (mean, last_token) is a separate training run so that
training and evaluation use the same pooling — enabling clean ablation reporting.

Usage:
    python src/ssm_finetune.py \\
        --model state-spaces/mamba2-130m \\
        --pooling mean \\
        --train-file data/processed/triplets_train.jsonl \\
        --val-file   data/processed/triplets_val.jsonl \\
        --output-dir results/mamba2-130m-mean
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from constants import PROJECT_ROOT
from ssm_io import save_mamba, load_mamba
from train_utils import pool, infonce_loss, StepLogger, optimizer_step
from triplet_dataset import TripletDataset, collate_fn


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

def _load_best_val_loss(log_path: Path) -> float:
    """Scan an existing train_log.jsonl and return the lowest val_loss_epoch seen."""
    best = float("inf")
    if not log_path.exists():
        return best
    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "val_loss_epoch" in entry:
                    best = min(best, entry["val_loss_epoch"])
            except json.JSONDecodeError:
                pass
    return best


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
    start_epoch = 1
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if args.start_epoch is not None:
            start_epoch = args.start_epoch
        else:
            m = re.search(r"checkpoint_epoch_(\d+)", resume_path.name)
            if m:
                start_epoch = int(m.group(1)) + 1
            else:
                raise ValueError(
                    f"Cannot infer start_epoch from '{resume_path.name}'. "
                    "Pass --start-epoch explicitly."
                )
        print(f"Resuming from {resume_path} → starting at epoch {start_epoch}")
        model = load_mamba(resume_path, device=device, dtype=dtype)
    else:
        # state-spaces/mamba2-* HF repos ship no tokenizer files. Canonical tokenizer
        # for all Mamba-2 checkpoints is EleutherAI/gpt-neox-20b (matches the Mamba paper's
        # Pile training setup and the team's tokenization_check.py).
        model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    # Resume: fast-forward scheduler to the correct step position
    # ------------------------------------------------------------------
    global_step = 0
    best_val_loss = float("inf")

    if args.resume_from and start_epoch > 1:
        completed_opt_steps = steps_per_epoch * (start_epoch - 1) // args.grad_accum
        for _ in range(completed_opt_steps):
            scheduler.step()
        global_step = completed_opt_steps
        best_val_loss = _load_best_val_loss(log_path)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Resumed: fast-forwarded scheduler to step {global_step}/{total_steps} "
            f"(lr={current_lr:.3e}) | best_val_loss so far: {best_val_loss:.4f}"
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_epoch = 0
    best_step = 0
    accum_loss = 0.0
    no_improve_epochs = 0
    stopped_epoch = args.epochs

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        epoch_train_loss = 0.0
        epoch_opt_steps = 0
        accum_t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(pbar, start=1):
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
                optimizer_step(model, optimizer, scheduler)
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
                        save_mamba(model, best_dir)
                        print(f"  >> New best ({val_loss:.4f}) — saved to {best_dir}")

        # Flush any remaining accumulated gradients at end of epoch
        if step % args.grad_accum != 0:
            optimizer_step(model, optimizer, scheduler)
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

        # ------ Save epoch checkpoint (enables --resume-from) ------
        epoch_dir = output_dir / f"checkpoint_epoch_{epoch:02d}"
        save_mamba(model, epoch_dir)
        print(f"  >> Epoch checkpoint saved to {epoch_dir}")

        # ------ Best model + early stopping ------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_step = global_step
            no_improve_epochs = 0
            best_dir = output_dir / "best_model"
            save_mamba(model, best_dir)
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
    save_mamba(model, final_dir)
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
    p.add_argument("--patience",    type=int, default=3,
                   help="Early stopping patience in epochs. Training stops if val loss does not improve for this many consecutive epochs.")
    p.add_argument("--max-len",     type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--warmup-ratio",type=float, default=0.1,
                   help="Fraction of total steps used for linear warmup")
    p.add_argument("--val-every",   type=int, default=500,
                   help="Validate every N optimizer steps")
    p.add_argument("--resume-from", default=None,
                   help="Checkpoint directory to resume from (e.g. results/mamba2-130m-mean/checkpoint_epoch_02). "
                        "start-epoch is inferred from the directory name unless --start-epoch is also passed.")
    p.add_argument("--start-epoch", type=int, default=None,
                   help="Epoch to resume from (1-indexed). Only needed with --resume-from if the "
                        "checkpoint name does not follow the checkpoint_epoch_NN convention.")

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
