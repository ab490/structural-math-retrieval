"""
Builds both Phase 2 diagnostic sets for operation clustering analysis.

Set 1 — Broad (diagnostic_set.jsonl):
  Real MSE anchor texts sampled from the test set, grouped by (topic, operation).
  Topics: calculus, analysis. Operations: differentiation, integration, limit, summation.
  15 samples per (topic, operation) cell → 120 total.

Set 2 — Tight Synthetic (diagnostic_set_tight_synthetic.jsonl):
  Synthetic controlled problems. Each of 30 expression groups (e.g. x^2, sin(x), e^x)
  has exactly one problem per operation — only the operation changes, the expression
  is held constant. All problems use "Compute" as the verb so the only signal is
  the mathematical notation. 30 groups × 4 operations = 120 total.

Usage:
    python src/build_diagnostic_set.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from constants import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

TARGET_TOPICS = {"calculus", "analysis"}
TARGET_OPS    = {"differentiation", "integration", "limit", "summation"}
N_PER_CELL    = 15
SEED          = 42

# ---------------------------------------------------------------------------
# Tight set: 30 expression groups × 4 operations
# ---------------------------------------------------------------------------

GROUPS = [
    {
        "expression": "x^2",
        "differentiation": r"Compute $\frac{d}{dx}\left[x^2\right]$.",
        "integration":     r"Compute $\int x^2 \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{x^2}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} k^2$.",
    },
    {
        "expression": "x^3",
        "differentiation": r"Compute $\frac{d}{dx}\left[x^3\right]$.",
        "integration":     r"Compute $\int x^3 \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{x^3}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} k^3$.",
    },
    {
        "expression": "sin(x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[\sin(x)\right]$.",
        "integration":     r"Compute $\int \sin(x) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{\sin(x)}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} \sin(k)$.",
    },
    {
        "expression": "e^x",
        "differentiation": r"Compute $\frac{d}{dx}\left[e^x\right]$.",
        "integration":     r"Compute $\int e^x \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{e^x - 1}{x}$.",
        "summation":       r"Compute $\sum_{k=0}^{n} e^k$.",
    },
    {
        "expression": "1/x",
        "differentiation": r"Compute $\frac{d}{dx}\left[\frac{1}{x}\right]$.",
        "integration":     r"Compute $\int \frac{1}{x} \, dx$.",
        "limit":           r"Compute $\lim_{x \to \infty} \frac{1}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} \frac{1}{k}$.",
    },
    {
        "expression": "x^2 - 4",
        "differentiation": r"Compute $\frac{d}{dx}\left[x^2 - 4\right]$.",
        "integration":     r"Compute $\int (x^2 - 4) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 2} \frac{x^2 - 4}{x - 2}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} (k^2 - 4)$.",
    },
    {
        "expression": "ln(x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[\ln(x)\right]$.",
        "integration":     r"Compute $\int \ln(x) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0^+} x \ln(x)$.",
        "summation":       r"Compute $\sum_{k=1}^{n} \ln(k)$.",
    },
    {
        "expression": "1/(x^2+1)",
        "differentiation": r"Compute $\frac{d}{dx}\left[\frac{1}{x^2+1}\right]$.",
        "integration":     r"Compute $\int \frac{1}{x^2+1} \, dx$.",
        "limit":           r"Compute $\lim_{x \to \infty} \frac{1}{x^2+1}$.",
        "summation":       r"Compute $\sum_{k=1}^{\infty} \frac{1}{k^2+1}$.",
    },
    {
        "expression": "x * e^x",
        "differentiation": r"Compute $\frac{d}{dx}\left[x e^x\right]$.",
        "integration":     r"Compute $\int x e^x \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{x e^x}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} k e^k$.",
    },
    {
        "expression": "cos(x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[\cos(x)\right]$.",
        "integration":     r"Compute $\int \cos(x) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{1 - \cos(x)}{x^2}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} \cos(k)$.",
    },
    {
        "expression": "x^4 - x^2",
        "differentiation": r"Compute $\frac{d}{dx}\left[x^4 - x^2\right]$.",
        "integration":     r"Compute $\int (x^4 - x^2) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 1} \frac{x^4 - x^2}{x - 1}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} (k^4 - k^2)$.",
    },
    {
        "expression": "sqrt(x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[\sqrt{x}\right]$.",
        "integration":     r"Compute $\int \sqrt{x} \, dx$.",
        "limit":           r"Compute $\lim_{x \to 4} \frac{\sqrt{x} - 2}{x - 4}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} \sqrt{k}$.",
    },
    {
        "expression": "1/(x(x+1))",
        "differentiation": r"Compute $\frac{d}{dx}\left[\frac{1}{x(x+1)}\right]$.",
        "integration":     r"Compute $\int \frac{1}{x(x+1)} \, dx$.",
        "limit":           r"Compute $\lim_{x \to \infty} \frac{1}{x(x+1)}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} \frac{1}{k(k+1)}$.",
    },
    {
        "expression": "e^(2x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[e^{2x}\right]$.",
        "integration":     r"Compute $\int e^{2x} \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{e^{2x} - 1}{x}$.",
        "summation":       r"Compute $\sum_{k=0}^{n} e^{2k}$.",
    },
    {
        "expression": "x^3 + 3x^2",
        "differentiation": r"Compute $\frac{d}{dx}\left[x^3 + 3x^2\right]$.",
        "integration":     r"Compute $\int (x^3 + 3x^2) \, dx$.",
        "limit":           r"Compute $\lim_{x \to -3} \frac{x^3 + 3x^2}{x + 3}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} (k^3 + 3k^2)$.",
    },
    {
        "expression": "x^2 + 3x",
        "differentiation": r"Compute $\frac{d}{dx}\left[x^2 + 3x\right]$.",
        "integration":     r"Compute $\int (x^2 + 3x) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{x^2 + 3x}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} (k^2 + 3k)$.",
    },
    {
        "expression": "tan(x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[\tan(x)\right]$.",
        "integration":     r"Compute $\int \tan(x) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{\tan(x)}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} \tan(k)$.",
    },
    {
        "expression": "x * sin(x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[x \sin(x)\right]$.",
        "integration":     r"Compute $\int x \sin(x) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{x \sin(x)}{x^2}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} k \sin(k)$.",
    },
    {
        "expression": "1/x^2",
        "differentiation": r"Compute $\frac{d}{dx}\left[\frac{1}{x^2}\right]$.",
        "integration":     r"Compute $\int \frac{1}{x^2} \, dx$.",
        "limit":           r"Compute $\lim_{x \to \infty} \frac{1}{x^2}$.",
        "summation":       r"Compute $\sum_{k=1}^{\infty} \frac{1}{k^2}$.",
    },
    {
        "expression": "x^3 - x",
        "differentiation": r"Compute $\frac{d}{dx}\left[x^3 - x\right]$.",
        "integration":     r"Compute $\int (x^3 - x) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 1} \frac{x^3 - x}{x - 1}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} (k^3 - k)$.",
    },
    {
        "expression": "e^(-x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[e^{-x}\right]$.",
        "integration":     r"Compute $\int e^{-x} \, dx$.",
        "limit":           r"Compute $\lim_{x \to \infty} e^{-x}$.",
        "summation":       r"Compute $\sum_{k=0}^{n} e^{-k}$.",
    },
    {
        "expression": "x * cos(x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[x \cos(x)\right]$.",
        "integration":     r"Compute $\int x \cos(x) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{x \cos(x)}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} k \cos(k)$.",
    },
    {
        "expression": "ln(x+1)",
        "differentiation": r"Compute $\frac{d}{dx}\left[\ln(x+1)\right]$.",
        "integration":     r"Compute $\int \ln(x+1) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{\ln(x+1)}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} \ln(k+1)$.",
    },
    {
        "expression": "1/(x+1)",
        "differentiation": r"Compute $\frac{d}{dx}\left[\frac{1}{x+1}\right]$.",
        "integration":     r"Compute $\int \frac{1}{x+1} \, dx$.",
        "limit":           r"Compute $\lim_{x \to \infty} \frac{1}{x+1}$.",
        "summation":       r"Compute $\sum_{k=0}^{n} \frac{1}{k+1}$.",
    },
    {
        "expression": "x^4",
        "differentiation": r"Compute $\frac{d}{dx}\left[x^4\right]$.",
        "integration":     r"Compute $\int x^4 \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{x^4}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} k^4$.",
    },
    {
        "expression": "sin^2(x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[\sin^2(x)\right]$.",
        "integration":     r"Compute $\int \sin^2(x) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{\sin^2(x)}{x^2}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} \sin^2(k)$.",
    },
    {
        "expression": "x/(x+1)",
        "differentiation": r"Compute $\frac{d}{dx}\left[\frac{x}{x+1}\right]$.",
        "integration":     r"Compute $\int \frac{x}{x+1} \, dx$.",
        "limit":           r"Compute $\lim_{x \to \infty} \frac{x}{x+1}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} \frac{k}{k+1}$.",
    },
    {
        "expression": "1/x^3",
        "differentiation": r"Compute $\frac{d}{dx}\left[\frac{1}{x^3}\right]$.",
        "integration":     r"Compute $\int \frac{1}{x^3} \, dx$.",
        "limit":           r"Compute $\lim_{x \to \infty} \frac{1}{x^3}$.",
        "summation":       r"Compute $\sum_{k=1}^{\infty} \frac{1}{k^3}$.",
    },
    {
        "expression": "e^x * cos(x)",
        "differentiation": r"Compute $\frac{d}{dx}\left[e^x \cos(x)\right]$.",
        "integration":     r"Compute $\int e^x \cos(x) \, dx$.",
        "limit":           r"Compute $\lim_{x \to 0} \frac{e^x \cos(x) - 1}{x}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} e^k \cos(k)$.",
    },
    {
        "expression": "x^2 + 1",
        "differentiation": r"Compute $\frac{d}{dx}\left[x^2 + 1\right]$.",
        "integration":     r"Compute $\int (x^2 + 1) \, dx$.",
        "limit":           r"Compute $\lim_{x \to \infty} \frac{x^2 + 1}{x^2}$.",
        "summation":       r"Compute $\sum_{k=1}^{n} (k^2 + 1)$.",
    },
]

OPERATIONS = ["differentiation", "integration", "limit", "summation"]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_broad(test_file: Path) -> list[dict]:
    random.seed(SEED)
    records = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    groups: dict[tuple, list] = defaultdict(list)
    for rec in records:
        topic = rec.get("topic", "")
        op = rec.get("anchor_op", "")
        if topic in TARGET_TOPICS and op in TARGET_OPS:
            groups[(topic, op)].append({
                "text":      rec["anchor"],
                "topic":     topic,
                "operation": op,
                "qid":       rec.get("anchor_qid", -1),
            })

    diagnostic: list[dict] = []
    for topic in sorted(TARGET_TOPICS):
        for op in sorted(TARGET_OPS):
            key = (topic, op)
            pool = groups[key][:]
            random.shuffle(pool)
            diagnostic.extend(pool[:N_PER_CELL])

    return diagnostic


def build_tight() -> list[dict]:
    records = []
    for group_id, group in enumerate(GROUPS):
        for op in OPERATIONS:
            records.append({
                "text":       group[op],
                "operation":  op,
                "group_id":   group_id,
                "expression": group["expression"],
            })
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "processed"
    test_file = data_dir / "triplets_test.jsonl"

    # --- Broad set ---
    broad = build_broad(test_file)
    broad_path = data_dir / "diagnostic_set.jsonl"
    with open(broad_path, "w") as f:
        for item in broad:
            f.write(json.dumps(item) + "\n")
    op_counts = Counter(d["operation"] for d in broad)
    print(f"Broad set  : {len(broad)} samples — {dict(sorted(op_counts.items()))}")
    print(f"  → {broad_path}")

    # --- Tight set ---
    tight = build_tight()
    tight_path = data_dir / "diagnostic_set_tight_synthetic.jsonl"
    with open(tight_path, "w") as f:
        for item in tight:
            f.write(json.dumps(item) + "\n")
    op_counts_tight = Counter(d["operation"] for d in tight)
    print(f"Tight set  : {len(tight)} samples — {dict(sorted(op_counts_tight.items()))}")
    print(f"  → {tight_path}")


if __name__ == "__main__":
    main()