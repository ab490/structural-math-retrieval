"""
Compare eval_results.json files across all model smoke tests.

Usage:
    python src/compare_results.py
"""

import json
from pathlib import Path
from constants import PROJECT_ROOT

RUNS = {
    # Zero-shot baselines
    "RoBERTa zeroshot (cls)":                  PROJECT_ROOT / "results" / "roberta-base-zeroshot-cls"               / "eval_results.json",
    "RoBERTa zeroshot (mean)":                 PROJECT_ROOT / "results" / "roberta-base-zeroshot-mean"              / "eval_results.json",
    "RoBERTa zeroshot (last_token)":           PROJECT_ROOT / "results" / "roberta-base-zeroshot-last_token"        / "eval_results.json",
    "RoBERTa-large zeroshot (cls)":            PROJECT_ROOT / "results" / "roberta-large-zeroshot-cls"              / "eval_results.json",
    "RoBERTa-large zeroshot (mean)":           PROJECT_ROOT / "results" / "roberta-large-zeroshot-mean"             / "eval_results.json",
    "RoBERTa-large zeroshot (last_token)":     PROJECT_ROOT / "results" / "roberta-large-zeroshot-last_token"       / "eval_results.json",
    "Mamba-2 130M zeroshot (mean)":            PROJECT_ROOT / "results" / "mamba2-130m-zeroshot-mean"               / "eval_results.json",
    "Mamba-2 130M zeroshot (last_token)":      PROJECT_ROOT / "results" / "mamba2-130m-zeroshot-last_token"         / "eval_results.json",
    "DeBERTa-v3-large zeroshot (cls)":         PROJECT_ROOT / "results" / "deberta-v3-large-zeroshot-cls"           / "eval_results.json",
    "DeBERTa-v3-large zeroshot (mean)":        PROJECT_ROOT / "results" / "deberta-v3-large-zeroshot-mean"          / "eval_results.json",
    "DeBERTa-v3-large zeroshot (last_token)":  PROJECT_ROOT / "results" / "deberta-v3-large-zeroshot-last_token"    / "eval_results.json",
    "Mamba-2 370M zeroshot (mean)":            PROJECT_ROOT / "results" / "mamba2-370m-zeroshot-mean"               / "eval_results.json",
    "Mamba-2 370M zeroshot (last_token)":      PROJECT_ROOT / "results" / "mamba2-370m-zeroshot-last_token"         / "eval_results.json",
    "DeBERTa-v2-xxlarge zeroshot (cls)":       PROJECT_ROOT / "results" / "deberta-v2-xxlarge-zeroshot-cls"         / "eval_results.json",
    "DeBERTa-v2-xxlarge zeroshot (mean)":      PROJECT_ROOT / "results" / "deberta-v2-xxlarge-zeroshot-mean"        / "eval_results.json",
    "DeBERTa-v2-xxlarge zeroshot (last_token)":PROJECT_ROOT / "results" / "deberta-v2-xxlarge-zeroshot-last_token"  / "eval_results.json",
    "Mamba-2 1.3B zeroshot (mean)":            PROJECT_ROOT / "results" / "mamba2-1.3b-zeroshot-mean"               / "eval_results.json",
    "Mamba-2 1.3B zeroshot (last_token)":      PROJECT_ROOT / "results" / "mamba2-1.3b-zeroshot-last_token"         / "eval_results.json",
    # Fine-tuned
    "RoBERTa (cls)":                           PROJECT_ROOT / "results" / "roberta-base-cls"                        / "eval_results.json",
    "RoBERTa (mean)":                          PROJECT_ROOT / "results" / "roberta-base-mean"                       / "eval_results.json",
    "RoBERTa (last_token)":                    PROJECT_ROOT / "results" / "roberta-base-last_token"                 / "eval_results.json",
    "RoBERTa-large (cls)":                     PROJECT_ROOT / "results" / "roberta-large-cls"                       / "eval_results.json",
    "RoBERTa-large (mean)":                    PROJECT_ROOT / "results" / "roberta-large-mean"                      / "eval_results.json",
    "RoBERTa-large (last_token)":              PROJECT_ROOT / "results" / "roberta-large-last_token"                / "eval_results.json",
    "Mamba-2 130M (mean)":                     PROJECT_ROOT / "results" / "mamba2-130m-mean"                        / "eval_results.json",
    "Mamba-2 130M (last_token)":               PROJECT_ROOT / "results" / "mamba2-130m-last_token"                  / "eval_results.json",
    "DeBERTa-v3-large (cls)":                  PROJECT_ROOT / "results" / "deberta-v3-large-cls"                    / "eval_results.json",
    "DeBERTa-v3-large (mean)":                 PROJECT_ROOT / "results" / "deberta-v3-large-mean"                   / "eval_results.json",
    "DeBERTa-v3-large (last_token)":           PROJECT_ROOT / "results" / "deberta-v3-large-last_token"             / "eval_results.json",
    "Mamba-2 370M (mean)":                     PROJECT_ROOT / "results" / "mamba2-370m-mean"                        / "eval_results.json",
    "Mamba-2 370M (last_token)":               PROJECT_ROOT / "results" / "mamba2-370m-last_token"                  / "eval_results.json",
    "DeBERTa-v2-xxlarge (cls)":                PROJECT_ROOT / "results" / "deberta-v2-xxlarge-cls"                  / "eval_results.json",
    "DeBERTa-v2-xxlarge (mean)":               PROJECT_ROOT / "results" / "deberta-v2-xxlarge-mean"                 / "eval_results.json",
    "DeBERTa-v2-xxlarge (last_token)":         PROJECT_ROOT / "results" / "deberta-v2-xxlarge-last_token"           / "eval_results.json",
    "Mamba-2 1.3B (mean)":                     PROJECT_ROOT / "results" / "mamba2-1.3b-mean"                        / "eval_results.json",
    "Mamba-2 1.3B (last_token)":               PROJECT_ROOT / "results" / "mamba2-1.3b-last_token"                  / "eval_results.json",
}

METRICS = [
    ("pairwise_accuracy", "Pairwise Acc"),
    ("mean_positive_sim", "Mean Pos Sim"),
    ("mean_negative_sim", "Mean Neg Sim"),
    ("mean_margin",       "Mean Margin "),
    ("median_margin",     "Median Margin"),
    ("n_triplets",        "# Triplets  "),
]

RANKING_METRICS = [
    ("ndcg_at_10", "NDCG@10     "),
    ("mrr",        "MRR         "),
    ("corpus_size", "Corpus Size "),
]

LATENCY = [
    ("median_tokens_per_sec", "Median tok/s"),
    ("mean_ms_per_token",     "ms/token    "),
]

COL_W = max(len(m) for m in RUNS) + 2


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def print_table(title: str, rows: list[tuple[str, list]]) -> None:
    models = list(RUNS.keys())
    header = f"{'Metric':<18}" + "".join(f"{m:<{COL_W}}" for m in models)
    print(f"\n{'=' * len(header)}")
    print(title)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, vals in rows:
        row = f"{label:<18}" + "".join(
            f"{(fmt(v) if v is not None else 'N/A'):<{COL_W}}" for v in vals
        )
        print(row)
    print("=" * len(header))


def main() -> None:
    data = {name: load(path) for name, path in RUNS.items()}

    missing = [name for name, d in data.items() if d is None]
    if missing:
        print(f"\n[!] Missing results for: {', '.join(missing)}")

    # Retrieval metrics table
    retrieval_rows = []
    for key, label in METRICS:
        vals = []
        for name, d in data.items():
            if d is None:
                vals.append(None)
            else:
                vals.append(d.get("retrieval_metrics", {}).get(key))
        retrieval_rows.append((label, vals))

    print_table("PAIRWISE RETRIEVAL METRICS", retrieval_rows)

    # Ranking metrics table (NDCG@10, MRR)
    ranking_rows = []
    for key, label in RANKING_METRICS:
        vals = []
        for name, d in data.items():
            if d is None:
                vals.append(None)
            else:
                vals.append(d.get("ranking_metrics", {}).get(key))
        ranking_rows.append((label, vals))

    print_table("RANKING METRICS (NDCG@10, MRR)", ranking_rows)

    # Latency table
    latency_rows = []
    for key, label in LATENCY:
        vals = []
        for name, d in data.items():
            if d is None:
                vals.append(None)
            else:
                vals.append(d.get("latency", {}).get(key))
        latency_rows.append((label, vals))

    print_table("LATENCY", latency_rows)

    # Metadata
    print("\nRun metadata:")
    for name, d in data.items():
        if d:
            print(f"  {name}: checkpoint={d.get('checkpoint', 'N/A')}, "
                  f"pooling={d.get('pooling', 'N/A')}, "
                  f"max_len={d.get('max_len', 'N/A')}")


if __name__ == "__main__":
    main()
