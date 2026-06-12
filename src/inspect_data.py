"""
Inspect ARQMath tag distribution.

Counts all tags from the parsed questions parquet, sorted by frequency.
Use this output to decide which tags to include in TOPIC_NORMALIZATION.

Outputs:
  results/tag_counts.json  - all tags sorted by frequency

Usage:
    python src/inspect_data.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from constants import DATA_DIR

RESULTS_DIR = DATA_DIR.parent / "results"
QUESTIONS_PATH = DATA_DIR / "processed" / "arqmath_questions.parquet"
OUT_PATH = RESULTS_DIR / "tag_counts.json"


def main():
    print("Loading questions parquet ...")
    questions = pd.read_parquet(QUESTIONS_PATH, columns=["Tags"])
    print(f"  {len(questions):,} questions loaded")

    tag_counter = Counter()
    for tags in tqdm(questions["Tags"].dropna(), desc="Counting tags", unit="question"):
        for tag in tags.split("|"):
            tag = tag.strip().lower()
            if tag:
                tag_counter[tag] += 1

    all_tags = dict(tag_counter.most_common())

    print(f"\n  Total unique tags : {len(all_tags):,}")
    print(f"\nTop 50 tags by frequency:")
    for tag, count in list(all_tags.items())[:50]:
        print(f"  {tag:<35} {count:,}")

    with open(OUT_PATH, "w") as f:
        json.dump(all_tags, f, indent=2)

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()