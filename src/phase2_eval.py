"""
Phase 2: Operational Similarity Clustering Analysis.

Runs on both diagnostic sets automatically:
  - diagnostic_set.jsonl                (broad: real MSE questions, same topic area)
  - diagnostic_set_tight_synthetic.jsonl (tight synthetic: controlled problems, same expression)

For each model checkpoint × diagnostic set, embeds all texts and evaluates
whether the embedding space clusters by mathematical operation.

For the broad set (which has topic labels), also computes topic-level clustering to
distinguish operation-driven separation from surface topic-driven separation — the
core diagnostic question from the prospectus.

Metrics:
  - ARI (operation):   k-means clusters vs ground-truth operation labels (0=random, 1=perfect)
  - Silhouette (op):   embedding space separation by operation (ground-truth labels)
  - ARI (topic):       k-means clusters vs ground-truth topic labels [broad set only]
  - Silhouette (topic): embedding space separation by topic [broad set only]

Checkpoints are auto-discovered from results/ using the training naming
convention: results/<model_slug>-<pooling>/best_model.
Skips any checkpoint that does not exist (allows partial runs).

Outputs:
  results/phase2_results_broad.json
  results/phase2_results_tight_synthetic.json

Usage:
    python src/phase2_eval.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer

from constants import PROJECT_ROOT

BATCH_SIZE = 32
MAX_LEN    = 512

DIAGNOSTIC_SETS = [
    {
        "name":    "broad",
        "file":    PROJECT_ROOT / "data" / "processed" / "diagnostic_set.jsonl",
        "out":     PROJECT_ROOT / "results" / "phase2_results_broad.json",
    },
    {
        "name":    "tight_synthetic",
        "file":    PROJECT_ROOT / "data" / "processed" / "diagnostic_set_tight_synthetic.jsonl",
        "out":     PROJECT_ROOT / "results" / "phase2_results_tight_synthetic.json",
    },
]

MODELS = [
    # Zero-shot baselines (no fine-tuning)
    {"name": "RoBERTa zeroshot (cls)",                  "slug": "roberta-base-zeroshot-cls",               "pooling": "cls",        "type": "auto",    "checkpoint": "roberta-base"},
    {"name": "RoBERTa zeroshot (mean)",                 "slug": "roberta-base-zeroshot-mean",              "pooling": "mean",       "type": "auto",    "checkpoint": "roberta-base"},
    {"name": "RoBERTa zeroshot (last_token)",           "slug": "roberta-base-zeroshot-last_token",        "pooling": "last_token", "type": "auto",    "checkpoint": "roberta-base"},
    {"name": "RoBERTa-large zeroshot (cls)",            "slug": "roberta-large-zeroshot-cls",              "pooling": "cls",        "type": "auto",    "checkpoint": "roberta-large"},
    {"name": "RoBERTa-large zeroshot (mean)",           "slug": "roberta-large-zeroshot-mean",             "pooling": "mean",       "type": "auto",    "checkpoint": "roberta-large"},
    {"name": "RoBERTa-large zeroshot (last_token)",     "slug": "roberta-large-zeroshot-last_token",       "pooling": "last_token", "type": "auto",    "checkpoint": "roberta-large"},
    {"name": "Mamba-2 130M zeroshot (mean)",            "slug": "mamba2-130m-zeroshot-mean",               "pooling": "mean",       "type": "mamba",   "checkpoint": "state-spaces/mamba2-130m"},
    {"name": "Mamba-2 130M zeroshot (last_token)",      "slug": "mamba2-130m-zeroshot-last_token",         "pooling": "last_token", "type": "mamba",   "checkpoint": "state-spaces/mamba2-130m"},
    {"name": "DeBERTa-v3-large zeroshot (cls)",         "slug": "deberta-v3-large-zeroshot-cls",           "pooling": "cls",        "type": "deberta", "checkpoint": "microsoft/deberta-v3-large"},
    {"name": "DeBERTa-v3-large zeroshot (mean)",        "slug": "deberta-v3-large-zeroshot-mean",          "pooling": "mean",       "type": "deberta", "checkpoint": "microsoft/deberta-v3-large"},
    {"name": "DeBERTa-v3-large zeroshot (last_token)",  "slug": "deberta-v3-large-zeroshot-last_token",    "pooling": "last_token", "type": "deberta", "checkpoint": "microsoft/deberta-v3-large"},
    {"name": "Mamba-2 370M zeroshot (mean)",            "slug": "mamba2-370m-zeroshot-mean",               "pooling": "mean",       "type": "mamba",   "checkpoint": "state-spaces/mamba2-370m"},
    {"name": "Mamba-2 370M zeroshot (last_token)",      "slug": "mamba2-370m-zeroshot-last_token",         "pooling": "last_token", "type": "mamba",   "checkpoint": "state-spaces/mamba2-370m"},
    {"name": "DeBERTa-v2-xxlarge zeroshot (cls)",       "slug": "deberta-v2-xxlarge-zeroshot-cls",         "pooling": "cls",        "type": "deberta", "checkpoint": "microsoft/deberta-v2-xxlarge"},
    {"name": "DeBERTa-v2-xxlarge zeroshot (mean)",      "slug": "deberta-v2-xxlarge-zeroshot-mean",        "pooling": "mean",       "type": "deberta", "checkpoint": "microsoft/deberta-v2-xxlarge"},
    {"name": "DeBERTa-v2-xxlarge zeroshot (last_token)","slug": "deberta-v2-xxlarge-zeroshot-last_token",  "pooling": "last_token", "type": "deberta", "checkpoint": "microsoft/deberta-v2-xxlarge"},
    {"name": "Mamba-2 1.3B zeroshot (mean)",            "slug": "mamba2-1.3b-zeroshot-mean",               "pooling": "mean",       "type": "mamba",   "checkpoint": "state-spaces/mamba2-1.3b"},
    {"name": "Mamba-2 1.3B zeroshot (last_token)",      "slug": "mamba2-1.3b-zeroshot-last_token",         "pooling": "last_token", "type": "mamba",   "checkpoint": "state-spaces/mamba2-1.3b"},
    # Fine-tuned: small-scale
    {"name": "RoBERTa (cls)",                           "slug": "roberta-base-cls",                        "pooling": "cls",        "type": "auto"},
    {"name": "RoBERTa (mean)",                          "slug": "roberta-base-mean",                       "pooling": "mean",       "type": "auto"},
    {"name": "RoBERTa (last_token)",                    "slug": "roberta-base-last_token",                 "pooling": "last_token", "type": "auto"},
    {"name": "RoBERTa-large (cls)",                     "slug": "roberta-large-cls",                       "pooling": "cls",        "type": "auto"},
    {"name": "RoBERTa-large (mean)",                    "slug": "roberta-large-mean",                      "pooling": "mean",       "type": "auto"},
    {"name": "RoBERTa-large (last_token)",              "slug": "roberta-large-last_token",                "pooling": "last_token", "type": "auto"},
    {"name": "Mamba-2 130M (mean)",                     "slug": "mamba2-130m-mean",                        "pooling": "mean",       "type": "mamba"},
    {"name": "Mamba-2 130M (last_token)",               "slug": "mamba2-130m-last_token",                  "pooling": "last_token", "type": "mamba"},
    # Fine-tuned: mid-scale
    {"name": "DeBERTa-v3-large (cls)",                  "slug": "deberta-v3-large-cls",                    "pooling": "cls",        "type": "deberta"},
    {"name": "DeBERTa-v3-large (mean)",                 "slug": "deberta-v3-large-mean",                   "pooling": "mean",       "type": "deberta"},
    {"name": "DeBERTa-v3-large (last_token)",           "slug": "deberta-v3-large-last_token",             "pooling": "last_token", "type": "deberta"},
    {"name": "Mamba-2 370M (mean)",                     "slug": "mamba2-370m-mean",                        "pooling": "mean",       "type": "mamba"},
    {"name": "Mamba-2 370M (last_token)",               "slug": "mamba2-370m-last_token",                  "pooling": "last_token", "type": "mamba"},
    # Fine-tuned: large-scale
    {"name": "DeBERTa-v2-xxlarge (cls)",                "slug": "deberta-v2-xxlarge-cls",                  "pooling": "cls",        "type": "deberta"},
    {"name": "DeBERTa-v2-xxlarge (mean)",               "slug": "deberta-v2-xxlarge-mean",                 "pooling": "mean",       "type": "deberta"},
    {"name": "DeBERTa-v2-xxlarge (last_token)",         "slug": "deberta-v2-xxlarge-last_token",           "pooling": "last_token", "type": "deberta"},
    {"name": "Mamba-2 1.3B (mean)",                     "slug": "mamba2-1.3b-mean",                        "pooling": "mean",       "type": "mamba"},
    {"name": "Mamba-2 1.3B (last_token)",               "slug": "mamba2-1.3b-last_token",                  "pooling": "last_token", "type": "mamba"},
]


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy == "cls":
        return hidden_states[:, 0, :]
    elif strategy == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    elif strategy == "last_token":
        lengths = attention_mask.sum(dim=1) - 1
        B, _, D = hidden_states.shape
        idx = lengths.view(B, 1, 1).expand(B, 1, D)
        return hidden_states.gather(dim=1, index=idx).squeeze(1)
    else:
        raise ValueError(f"Unknown pooling: {strategy!r}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(checkpoint: str, model_type: str, dtype, device):
    if model_type == "mamba":
        # Mamba checkpoints are saved by src/mamba_io.save_mamba (mamba-ssm format),
        # not transformers' Mamba2Model. Tokenizer is hardcoded to EleutherAI/gpt-neox-20b
        # (canonical for Mamba-2; state-spaces repos ship no tokenizer).
        from mamba_io import load_mamba
        model = load_mamba(checkpoint, device=device, dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type == "deberta":
        from transformers import DebertaV2Tokenizer
        model = AutoModel.from_pretrained(checkpoint, torch_dtype=dtype).to(device)
        tokenizer = DebertaV2Tokenizer.from_pretrained(checkpoint)
    else:
        model = AutoModel.from_pretrained(checkpoint, torch_dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.padding_side = "right"
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(model, tokenizer, texts: list[str], pooling: str, device: torch.device, model_type: str = "auto") -> np.ndarray:
    all_embs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(batch, max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt")
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            # mamba-ssm exposes hidden states via model.backbone (skipping the LM head);
            # transformer encoders return last_hidden_state on the model output.
            if model_type == "mamba":
                hidden = model.backbone(input_ids)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state
            emb    = F.normalize(pool(hidden, attention_mask, pooling), p=2, dim=-1)
            all_embs.append(emb.cpu().float())
    return torch.cat(all_embs, dim=0).numpy()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_and_score(embeddings: np.ndarray, true_labels: np.ndarray, n_clusters: int) -> dict:
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    pred_labels = km.fit_predict(embeddings)
    return {
        "ari":        round(float(adjusted_rand_score(true_labels, pred_labels)), 4),
        "silhouette": round(float(silhouette_score(embeddings, true_labels)), 4),
    }


def score_against_labels(embeddings: np.ndarray, labels: np.ndarray, n_clusters: int) -> dict:
    """ARI + silhouette for an arbitrary label set (e.g. topic labels)."""
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    pred_labels = km.fit_predict(embeddings)
    return {
        "ari":        round(float(adjusted_rand_score(labels, pred_labels)), 4),
        "silhouette": round(float(silhouette_score(embeddings, labels)), 4),
    }


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def eval_dataset(ds: dict, device: torch.device, dtype) -> None:
    diag_file = ds["file"]
    out_path  = ds["out"]

    if not diag_file.exists():
        print(f"\n[SKIP] {ds['name']} diagnostic set not found: {diag_file}")
        return

    records: list[dict] = []
    with open(diag_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    texts      = [r["text"] for r in records]
    operations = [r["operation"] for r in records]

    le_op     = LabelEncoder()
    op_labels = le_op.fit_transform(operations)
    n_ops     = len(le_op.classes_)

    # Topic labels only available in the broad set
    has_topics  = all("topic" in r for r in records)
    topic_labels = None
    n_topics     = 0
    if has_topics:
        le_topic     = LabelEncoder()
        topic_labels = le_topic.fit_transform([r["topic"] for r in records])
        n_topics     = len(le_topic.classes_)

    print(f"\n{'='*60}")
    print(f"Diagnostic set: {ds['name']}  ({len(texts)} texts, {n_ops} operations)")
    print(f"Operations: {list(le_op.classes_)}")
    if has_topics:
        print(f"Topics ({n_topics}): {list(le_topic.classes_)}")
    print(f"{'='*60}")

    results_dir = PROJECT_ROOT / "results"
    model_results: list[dict] = []

    for cfg in MODELS:
        if "checkpoint" in cfg:
            # zeroshot: load directly from HuggingFace model ID
            checkpoint = cfg["checkpoint"]
        else:
            # fine-tuned: look for local best_model directory
            local = results_dir / cfg["slug"] / "best_model"
            if not local.exists():
                print(f"[SKIP] {cfg['name']} — checkpoint not found")
                continue
            checkpoint = str(local)

        print(f"\n[EVAL] {cfg['name']}")
        model, tokenizer = load_model_and_tokenizer(checkpoint, cfg["type"], dtype, device)
        embs = embed_texts(model, tokenizer, texts, cfg["pooling"], device, cfg["type"])

        op_scores = cluster_and_score(embs, op_labels, n_ops)
        result = {
            "model":               cfg["name"],
            "slug":                cfg["slug"],
            "pooling":             cfg["pooling"],
            "ari_operation":       op_scores["ari"],
            "silhouette_operation": op_scores["silhouette"],
        }
        print(f"  Operation  — ARI={op_scores['ari']:.4f}  Silhouette={op_scores['silhouette']:.4f}")

        if has_topics:
            topic_scores = score_against_labels(embs, topic_labels, n_topics)
            result["ari_topic"]        = topic_scores["ari"]
            result["silhouette_topic"] = topic_scores["silhouette"]
            print(f"  Topic      — ARI={topic_scores['ari']:.4f}  Silhouette={topic_scores['silhouette']:.4f}")

        model_results.append(result)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "diagnostic_set": ds["name"],
        "n_samples":      len(texts),
        "n_operations":   n_ops,
        "operations":     list(le_op.classes_),
        "results":        model_results,
    }
    if has_topics:
        output["n_topics"] = n_topics
        output["topics"]   = list(le_topic.classes_)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary table
    if has_topics:
        print(f"\n{'Model':<28} {'ARI(op)':>9} {'Sil(op)':>9} {'ARI(top)':>10} {'Sil(top)':>10}")
        print("-" * 68)
        for r in sorted(model_results, key=lambda x: -x["ari_operation"]):
            print(f"{r['model']:<28} {r['ari_operation']:>9.4f} {r['silhouette_operation']:>9.4f} "
                  f"{r['ari_topic']:>10.4f} {r['silhouette_topic']:>10.4f}")
    else:
        print(f"\n{'Model':<28} {'ARI(op)':>9} {'Sil(op)':>9}")
        print("-" * 48)
        for r in sorted(model_results, key=lambda x: -x["ari_operation"]):
            print(f"{r['model']:<28} {r['ari_operation']:>9.4f} {r['silhouette_operation']:>9.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    for ds in DIAGNOSTIC_SETS:
        eval_dataset(ds, device, dtype)


if __name__ == "__main__":
    main()