"""
Tokenization check: RoBERTa vs Mamba-2 (GPT-NeoX tokenizer) on LaTeX expressions.

Samples N LaTeX expressions from the `latex` column of the ARQMath questions and
answers parquet files, tokenizes with both tokenizers, and saves results to
results/tokenization_check.json.

The `latex` column stores pipe-separated expressions extracted from math-container
spans during parsing.
"""

import json
import random
import argparse
import statistics
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer, AutoTokenizer, DebertaV2Tokenizer

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from constants import DATA_DIR

QUESTIONS_PARQUET = DATA_DIR / "processed" / "arqmath_questions.parquet"
ANSWERS_PARQUET   = DATA_DIR / "processed" / "arqmath_answers.parquet"
OUT_DIR  = Path(__file__).resolve().parent.parent / "results"
OUT_FILE = OUT_DIR / "tokenization_check.json"

ROBERTA_MODEL  = "roberta-base"
DEBERTA_MODEL  = "microsoft/deberta-v3-base"
MAMBA_MODEL    = "EleutherAI/gpt-neox-20b"   # Mamba-2 uses GPT-NeoX tokenizer


def extract_latex_expressions(parquet_path: Path, min_chars: int) -> list[str]:
    """
    Loads the `latex` column from a parquet file and splits each row on ' | '
    to recover individual expressions.
    """
    df = pd.read_parquet(parquet_path, columns=["latex"])
    expressions = []
    for latex_str in df["latex"].dropna():
        for expr in latex_str.split(" | "):
            expr = expr.strip()
            if len(expr) >= min_chars:
                expressions.append(expr)
    return expressions


def tokenize_and_measure(expressions: list[str], roberta_tok, deberta_tok, mamba_tok) -> list[dict]:
    results = []
    for expr in tqdm(expressions, desc="Tokenizing", unit="expr"):
        n_chars   = len(expr)
        n_roberta = len(roberta_tok.encode(expr))
        n_deberta = len(deberta_tok.encode(expr))
        n_mamba   = len(mamba_tok.encode(expr))
        results.append({
            "expr":            expr,
            "chars":           n_chars,
            "roberta_tokens":  n_roberta,
            "deberta_tokens":  n_deberta,
            "mamba_tokens":    n_mamba,
            "roberta_ratio":   n_roberta / max(n_chars, 1),
            "deberta_ratio":   n_deberta / max(n_chars, 1),
            "mamba_ratio":     n_mamba   / max(n_chars, 1),
        })
    return results


def compute_summary(results: list[dict]) -> dict:
    rob_ratios     = [r["roberta_ratio"]  for r in results]
    deb_ratios     = [r["deberta_ratio"]  for r in results]
    mamba_ratios   = [r["mamba_ratio"]    for r in results]
    rob_tokens     = [r["roberta_tokens"] for r in results]
    deb_tokens     = [r["deberta_tokens"] for r in results]
    mamba_tokens   = [r["mamba_tokens"]   for r in results]
    chars          = [r["chars"]          for r in results]

    def pct(lst, p):
        s = sorted(lst)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    return {
        "n_expressions": len(results),
        "roberta": {
            "mean_ratio":    statistics.mean(rob_ratios),
            "median_ratio":  statistics.median(rob_ratios),
            "stdev_ratio":   statistics.stdev(rob_ratios) if len(rob_ratios) > 1 else 0.0,
            "p95_ratio":     pct(rob_ratios, 95),
            "mean_tokens":   statistics.mean(rob_tokens),
            "median_tokens": statistics.median(rob_tokens),
            "p95_tokens":    pct(rob_tokens, 95),
        },
        "deberta": {
            "mean_ratio":    statistics.mean(deb_ratios),
            "median_ratio":  statistics.median(deb_ratios),
            "stdev_ratio":   statistics.stdev(deb_ratios) if len(deb_ratios) > 1 else 0.0,
            "p95_ratio":     pct(deb_ratios, 95),
            "mean_tokens":   statistics.mean(deb_tokens),
            "median_tokens": statistics.median(deb_tokens),
            "p95_tokens":    pct(deb_tokens, 95),
        },
        "mamba": {
            "mean_ratio":    statistics.mean(mamba_ratios),
            "median_ratio":  statistics.median(mamba_ratios),
            "stdev_ratio":   statistics.stdev(mamba_ratios) if len(mamba_ratios) > 1 else 0.0,
            "p95_ratio":     pct(mamba_ratios, 95),
            "mean_tokens":   statistics.mean(mamba_tokens),
            "median_tokens": statistics.median(mamba_tokens),
            "p95_tokens":    pct(mamba_tokens, 95),
        },
        "ratio_comparison": {
            "mamba_over_roberta_mean":    statistics.mean(mamba_ratios) / max(statistics.mean(rob_ratios), 1e-9),
            "mamba_over_roberta_median":  statistics.median(mamba_ratios) / max(statistics.median(rob_ratios), 1e-9),
            "deberta_over_roberta_mean":  statistics.mean(deb_ratios) / max(statistics.mean(rob_ratios), 1e-9),
            "deberta_over_roberta_median":statistics.median(deb_ratios) / max(statistics.median(rob_ratios), 1e-9),
        },
        "expression_char_stats": {
            "mean":   statistics.mean(chars),
            "median": statistics.median(chars),
            "min":    min(chars),
            "max":    max(chars),
            "p95":    pct(chars, 95),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Tokenization audit for LaTeX expressions")
    parser.add_argument("--n",         type=int, default=500, help="Expressions to sample (default: 500)")
    parser.add_argument("--min-chars", type=int, default=10,  help="Min expression length in chars (default: 10)")
    parser.add_argument("--seed",      type=int, default=42,  help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Extracting LaTeX expressions ...")
    expressions = []
    for path, label in [(QUESTIONS_PARQUET, "questions"), (ANSWERS_PARQUET, "answers")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} parquet not found: {path}")
        exprs = extract_latex_expressions(path, min_chars=args.min_chars)
        print(f"  {label}: {len(exprs):,} expressions")
        expressions.extend(exprs)

    expressions = list(set(expressions))
    print(f"  {len(expressions):,} unique after dedup")

    if len(expressions) < args.n:
        print(f"  Warning: only {len(expressions)} available, using all")
        sampled = expressions
    else:
        sampled = random.sample(expressions, args.n)
    print(f"  Sampled {len(sampled)}")

    print(f"\nLoading tokenizers ...")
    roberta_tok  = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    deberta_tok  = DebertaV2Tokenizer.from_pretrained(DEBERTA_MODEL)
    mamba_tok    = AutoTokenizer.from_pretrained(MAMBA_MODEL)

    results = tokenize_and_measure(sampled, roberta_tok, deberta_tok, mamba_tok)
    summary = compute_summary(results)

    print("\n--- Summary ---")
    print(f"RoBERTa   mean token/char ratio : {summary['roberta']['mean_ratio']:.4f}")
    print(f"DeBERTa-v3 mean token/char ratio: {summary['deberta']['mean_ratio']:.4f}")
    print(f"Mamba-2   mean token/char ratio : {summary['mamba']['mean_ratio']:.4f}")
    print(f"Mamba/RoBERTa ratio (mean)      : {summary['ratio_comparison']['mamba_over_roberta_mean']:.2f}x")
    print(f"Mamba/RoBERTa ratio (median)    : {summary['ratio_comparison']['mamba_over_roberta_median']:.2f}x")
    print(f"DeBERTa/RoBERTa ratio (mean)    : {summary['ratio_comparison']['deberta_over_roberta_mean']:.2f}x")
    print(f"DeBERTa/RoBERTa ratio (median)  : {summary['ratio_comparison']['deberta_over_roberta_median']:.2f}x")

    if summary["ratio_comparison"]["mamba_over_roberta_mean"] >= 2.0:
        print("\n[FLAG] Mamba produces >=2x more tokens per expression on average.")
        print("       Acknowledge this as a tokenization confound in the paper.")

    print("\n--- Top 10 most divergent expressions (highest Mamba/RoBERTa token ratio) ---")
    top10 = sorted(results, key=lambda r: r["mamba_tokens"] / max(r["roberta_tokens"], 1), reverse=True)[:10]
    for r in top10:
        ratio = r["mamba_tokens"] / max(r["roberta_tokens"], 1)
        expr_display = r["expr"][:60].replace("\n", " ")
        if len(r["expr"]) > 60:
            expr_display += "..."
        print(f"  rob={r['roberta_tokens']:3d}  deb={r['deberta_tokens']:3d}  mamba={r['mamba_tokens']:3d}  ({ratio:.2f}x)  {expr_display}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "questions_parquet": str(QUESTIONS_PARQUET),
            "answers_parquet":   str(ANSWERS_PARQUET),
            "n_sampled":         len(sampled),
            "min_chars":         args.min_chars,
            "seed":              args.seed,
            "roberta_model":     ROBERTA_MODEL,
            "deberta_model":     DEBERTA_MODEL,
            "mamba_model":       MAMBA_MODEL,
        },
        "summary": summary,
        "per_expression": results,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to {OUT_FILE}")


if __name__ == "__main__":
    main()