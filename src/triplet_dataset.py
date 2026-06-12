"""
TripletDataset for each model fine-tuning.

Reads JSONL triplets produced by build_triplets.py and tokenizes
anchor / positive / negative texts. Returns tensors plus raw metadata
fields needed by the Phase 2 diagnostic evaluation.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TripletDataset(Dataset):
    """
    Each item is one (anchor, positive, negative) triplet.

    Parameters
    ----------
    jsonl_path : str | Path
        Path to a triplets_{train,val,test}.jsonl file.
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding the text fields.
    max_len : int
        Maximum token length for truncation.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.records: list[dict] = []

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, text: str) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),       # [T]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [T]
        }

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        anchor_enc = self._encode(rec["anchor"])
        positive_enc = self._encode(rec["positive"])
        negative_enc = self._encode(rec["negative"])

        return {
            # Tokenized tensors
            "anchor_input_ids":      anchor_enc["input_ids"],
            "anchor_attention_mask": anchor_enc["attention_mask"],
            "positive_input_ids":      positive_enc["input_ids"],
            "positive_attention_mask": positive_enc["attention_mask"],
            "negative_input_ids":      negative_enc["input_ids"],
            "negative_attention_mask": negative_enc["attention_mask"],
            # Metadata — kept as strings/ints for collation (not tensors)
            "anchor_qid":  rec.get("anchor_qid", -1),
            "topic":       rec.get("topic", ""),
            "anchor_op":   rec.get("anchor_op", ""),
            "negative_op": rec.get("negative_op", ""),
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Default DataLoader collation stacks tensors and keeps metadata as lists.
    This function is provided for explicitness; torch's default_collate also works
    for the tensor fields but not for mixed-type metadata.
    """
    tensor_keys = [
        "anchor_input_ids", "anchor_attention_mask",
        "positive_input_ids", "positive_attention_mask",
        "negative_input_ids", "negative_attention_mask",
    ]
    meta_keys = ["anchor_qid", "topic", "anchor_op", "negative_op"]

    out: dict = {}
    for k in tensor_keys:
        out[k] = torch.stack([item[k] for item in batch])
    for k in meta_keys:
        out[k] = [item[k] for item in batch]
    return out