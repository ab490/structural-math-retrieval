"""Shared utilities used across training and evaluation scripts."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor, strategy: str) -> torch.Tensor:
    """
    Reduce sequence of hidden states to a single embedding vector.

    hidden_states : [B, T, D]
    attention_mask: [B, T]  (1=real token, 0=padding)
    returns       : [B, D]
    """
    if strategy == "cls":
        return hidden_states[:, 0, :]

    elif strategy == "mean":
        mask = attention_mask.unsqueeze(-1).float()          # [B, T, 1]
        summed = (hidden_states * mask).sum(dim=1)           # [B, D]
        counts = mask.sum(dim=1).clamp(min=1e-9)             # [B, 1]
        return summed / counts

    elif strategy == "last_token":
        # Index of the last real (non-padding) token per sequence.
        # Right-padding is assumed (tokenizer.padding_side = "right").
        lengths = attention_mask.sum(dim=1) - 1              # [B]
        B, _, D = hidden_states.shape
        idx = lengths.view(B, 1, 1).expand(B, 1, D)
        return hidden_states.gather(dim=1, index=idx).squeeze(1)   # [B, D]

    else:
        raise ValueError(f"Unknown pooling strategy: {strategy!r}. Choose 'cls', 'mean', or 'last_token'.")


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


class StepLogger:
    """Appends one JSON line per step to a JSONL file."""

    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def log(self, **kwargs) -> None:
        self.f.write(json.dumps(kwargs) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def optimizer_step(model, optimizer, scheduler) -> None:
    """Clip gradients, step optimizer and scheduler, then zero gradients."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
