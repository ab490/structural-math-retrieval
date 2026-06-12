"""
Construct ARQMath triplets for training.

Triplet format:
- anchor   = question text
- positive = that question's accepted answer, or best-scoring answer if no accepted answer exists
- negative = another question's accepted/best answer from the same topic but a different operation

e.g.
{
  "anchor":      "How do I find the derivative of x^2 + 3x? I know the power rule but I'm not sure how to apply it term by term...",
  "positive":    "To differentiate x^2 + 3x apply the power rule to each term. For x^2 bring down the exponent giving 2x...",
  "negative":    "To integrate x^2 + 3x we reverse the power rule. The antiderivative of x^2 is x^3/3...",
  "topic":       "calculus",
  "anchor_op": "differentiation",
  "negative_op": "integration"
}

Important:
- Split is done at the QUESTION level first to avoid leakage.
- Negatives for each split are sampled only from that same split.

Outputs:
- triplets_train.jsonl
- triplets_val.jsonl
- triplets_test.jsonl
- triplets_report.json
"""

import re
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from constants import DATA_DIR, TOPIC_NORMALIZATION

ANSWERS_PATH   = DATA_DIR / "processed" / "arqmath_answers.parquet"
QUESTIONS_PATH = DATA_DIR / "processed" / "arqmath_questions.parquet"

OUT_TRAIN = DATA_DIR / "processed" / "triplets_train.jsonl"
OUT_VAL = DATA_DIR / "processed" / "triplets_val.jsonl"
OUT_TEST = DATA_DIR / "processed" / "triplets_test.jsonl"
OUT_REPORT = DATA_DIR / "processed" / "triplets_report.json"

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

RANDOM_SEED = 42

assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-9, "Splits must sum to 1"


OPERATION_PATTERNS = {
    "differentiation": [
        r"\\frac\{d",
        r"\\frac\s*\{\s*d\s*\}\s*\{\s*d[a-z]\s*\}",
        r"\\partial",
        r"\bderivative\b",
        r"\bderivatives\b",
        r"\bdifferentiate\b",
        r"\bdifferentiating\b",
        r"\bdifferentiation\b",
        r"\bd/dx\b",
        r"\bd/d[a-z]\b",
        r"\\nabla",
        r"\bgradient\b",
    ],
    "integration": [
        r"\\int",
        r"\\iint",
        r"\\iiint",
        r"\\oint",
        r"\bintegral\b",
        r"\bintegrals\b",
        r"\bintegrate\b",
        r"\bintegrating\b",
        r"\bintegration\b",
        r"\bantiderivative\b",
        r"\bantiderivatives\b",
    ],
    "limit": [
        r"\\lim",
        r"\blimit\b",
        r"\blimits\b",
        r"\btends to\b",
        r"\bas\s+[a-z]\s*->",
        r"\bapproaches\b",
        r"\bl['\u2019]?h[oô]pital\b",
    ],
    "summation": [
        r"\\sum",
        r"\bsummation\b",
        r"\bpartial sum\b",
        r"\btelescoping\b",
    ],
    "matrix": [
        r"\\det",
        r"\\begin\{[bBpvV]?matrix\}",
        r"\bmatrix\b",
        r"\bmatrices\b",
        r"\beigenvalue\b",
        r"\beigenvalues\b",
        r"\beigenvector\b",
        r"\beigenvectors\b",
        r"\bdeterminant\b",
        r"\bdeterminants\b",
        r"\\mathrm\{rank\}",
        r"\\operatorname\{rank\}",
        r"\btrace\b",
        r"\\operatorname\{tr\}",
        r"\btranspose\b",
        r"\\top\b",
        r"\bnull space\b",
        r"\bsingular value\b",
        r"\bsingular values\b",
    ],
    "probability": [
        r"\\mathbb\{[Pp]\}\s*[\(\[]",
        r"\\pr\b",                      
        r"\bprobability\b",
        r"\bprobabilities\b",
        r"\\mathbb\{[Ee]\}\s*[\(\[\{]",
        r"\bexpected value\b",
        r"\bexpectation\b",
        r"\bvariance\b",
        r"\brandom variable\b",
        r"\brandom variables\b",
        r"\bbayes\b",                   
        r"\bpdf\b",                     
        r"\bcdf\b",                     
        r"\bpmf\b",                     
    ],
    "statistics": [
        r"\bhypothesis test\b",
        r"\bhypothesis testing\b",
        r"\bp-value\b",
        r"\bconfidence interval\b",
        r"\bconfidence intervals\b",
        r"\bregression\b",
        r"\bleast squares\b",
        r"\bmaximum likelihood\b",
        r"\bmle\b",                     
        r"\bt-test\b",
        r"\bchi-square\b",
        r"\bchi-squared\b",             
        r"\bchi squared\b",
        r"\banova\b",                   
        r"\bstandard deviation\b",
        r"\bsample mean\b",
        r"\bsample variance\b",
    ],
}


def label_operation(text: str):
    if not isinstance(text, str) or not text.strip():
        return None

    text = text.lower()
    scores = {}

    for op, patterns in OPERATION_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            score += len(matches)
        if score > 0:
            scores[op] = score
    if not scores:
        return None

    max_score = max(scores.values())
    best_ops = [op for op, score in scores.items() if score == max_score]
    
    OPERATION_PRIORITY = ["differentiation", "integration", "limit", "summation", "matrix", "statistics", "probability"]
    
    if len(best_ops) == 1:
        return best_ops[0]
    
    # Break tie by priority order
    for op in OPERATION_PRIORITY:
        if op in best_ops:
            return op
    
    return best_ops[0]


def save_jsonl(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            

def get_primary_topic(tags):
    if not isinstance(tags, str) or not tags.strip():
        return None

    post_tags = [t.strip().lower() for t in tags.split("|") if t.strip()]

    for tag in post_tags:
        if tag in TOPIC_NORMALIZATION:
            return TOPIC_NORMALIZATION[tag]

    return None


def choose_best_answer(group: pd.DataFrame):
    if group.empty:
        return None
    # Highest score first, tie-break by lower Id (earlier answer)
    group = group.sort_values(["Score", "Id"], ascending=[False, True])
    return group.iloc[0]


def build_triplets_for_split(split_df: pd.DataFrame, split_name: str):
    print(f"\nBuilding {split_name} triplets ...")

    neg_index = defaultdict(lambda: defaultdict(list))
    topic_op_counts = defaultdict(dict)

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Indexing {split_name}", unit="question"):
        topic = row["topic"]
        op = row["anchor_op"]
        neg_index[topic][op].append({
            "qid": int(row["anchor_qid"]),
            "aid": int(row["positive_aid"]),
            "text": row["positive"],
        })

    for topic, op_map in neg_index.items():
        for op, items in op_map.items():
            topic_op_counts[topic][op] = len(items)

    triplets = []
    skipped_no_negative = 0

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Building {split_name} triplets", unit="question"):
        topic = row["topic"]
        pos_op = row["anchor_op"]

        available_ops = [
            op for op, items in neg_index[topic].items()
            if op != pos_op and len(items) > 0
        ]
        if not available_ops:
            skipped_no_negative += 1
            continue

        neg_op = random.choice(available_ops)
        neg_candidates = [
            x for x in neg_index[topic][neg_op]
            if x["qid"] != int(row["anchor_qid"])
        ]
        if not neg_candidates:
            skipped_no_negative += 1
            continue

        negative = random.choice(neg_candidates)

        triplets.append({
            "anchor_qid": int(row["anchor_qid"]),
            "anchor": row["anchor"],
            "topic": topic,
            "anchor_op": pos_op,
            "positive_aid": int(row["positive_aid"]),
            "positive": row["positive"],
            "negative_qid": int(negative["qid"]),
            "negative_aid": int(negative["aid"]),
            "negative": negative["text"],
            "negative_op": neg_op,
        })

    raw_triplets = len(triplets)
    random.shuffle(triplets)

    print(f"  Raw triplets        : {raw_triplets:,}")
    print(f"  Final triplets      : {len(triplets):,}")
    print(f"  Skipped no negative : {skipped_no_negative:,}")

    stats = {
        "questions_in_split": int(len(split_df)),
        "raw_triplets": int(raw_triplets),
        "final_triplets": int(len(triplets)),
        "skipped_no_negative": int(skipped_no_negative),
        "operation_distribution": {
            k: int(v) for k, v in split_df["anchor_op"].value_counts().to_dict().items()
        },
        "topic_operation_distribution": {
            topic: {op: int(count) for op, count in op_map.items()}
            for topic, op_map in topic_op_counts.items()
        },
        "skipped_no_negative_pct": round(skipped_no_negative / len(split_df) * 100, 2) if len(split_df) else 0        
    }
    return triplets, stats


def compute_length_stats(df: pd.DataFrame, text_cols=("anchor", "positive")):
    stats = {}

    for col in text_cols:
        texts = df[col].astype(str)
        token_lengths = texts.apply(lambda x: len(x.split()))
        char_lengths = texts.apply(len)

        stats[col] = {
            "mean_tokens": float(token_lengths.mean()),
            "median_tokens": float(token_lengths.median()),
            "max_tokens": int(token_lengths.max()),
            "p95_tokens": float(token_lengths.quantile(0.95)),
            "p99_tokens": float(token_lengths.quantile(0.99)),

            "mean_chars": float(char_lengths.mean()),
            "max_chars": int(char_lengths.max()),
            "p95_chars": float(char_lengths.quantile(0.95)),
        }

    return stats


def main():
    random.seed(RANDOM_SEED)

    print("Loading Parquet files ...")
    answers = pd.read_parquet(
        ANSWERS_PATH,
        columns=["Id", "ParentId", "Score", "plain_text", "latex"]
    )
    questions = pd.read_parquet(
        QUESTIONS_PATH,
        columns=["Id", "Title", "Tags", "AcceptedAnswerId", "plain_text", "latex"]
    )

    print(f"  Answers   : {len(answers):,}")
    print(f"  Questions : {len(questions):,}")
    
    answers["Id"] = pd.to_numeric(answers["Id"], errors="coerce")
    answers["ParentId"] = pd.to_numeric(answers["ParentId"], errors="coerce")
    answers["Score"] = pd.to_numeric(answers["Score"], errors="coerce").fillna(0)

    questions["Id"] = pd.to_numeric(questions["Id"], errors="coerce")
    questions["AcceptedAnswerId"] = pd.to_numeric(questions["AcceptedAnswerId"], errors="coerce")

    answers = answers.dropna(subset=["Id", "ParentId"]).copy()
    questions = questions.dropna(subset=["Id"]).copy()

    answers["Id"] = answers["Id"].astype(int)
    answers["ParentId"] = answers["ParentId"].astype(int)
    questions["Id"] = questions["Id"].astype(int)
    
    n_loaded_questions = len(questions)
    n_with_latex_questions = int(questions["latex"].fillna("").str.strip().ne("").sum())    

    print("\nDetecting operation from question text ...")
    questions["operation_input"] = (
        questions["Title"].fillna("") + " " +
        questions["plain_text"].fillna("")
    ).str.strip()

    questions["operation"] = questions["operation_input"].apply(label_operation)
    op_counts_all = questions["operation"].value_counts(dropna=True).to_dict()

    questions = questions[questions["operation"].notna()].copy()
    print(f"  Questions with operation label: {len(questions):,}")

    print("\nSelecting one positive answer per question ...")
    answers_by_id = answers.set_index("Id", drop=False)
    answers_by_parent = {qid: grp.copy() for qid, grp in answers.groupby("ParentId")}

    selected_rows = []
    skipped_no_answer = 0
    skipped_missing_accepted = 0
    skipped_empty_anchor = 0
    skipped_empty_positive = 0    

    for _, qrow in tqdm(questions.iterrows(),  total=len(questions), desc="Selecting positives", unit="question"):
        qid = int(qrow["Id"])
        accepted_aid = qrow["AcceptedAnswerId"]

        candidate_answers = answers_by_parent.get(qid)
        if candidate_answers is None or candidate_answers.empty:
            skipped_no_answer += 1
            continue

        chosen_answer = None
        positive_source = None

        if pd.notna(accepted_aid):
            accepted_aid = int(accepted_aid)
            if accepted_aid in answers_by_id.index:
                accepted_row = answers_by_id.loc[accepted_aid]
                if isinstance(accepted_row, pd.DataFrame):
                    accepted_row = accepted_row.iloc[0]  # take first if duplicates                
                if int(accepted_row["ParentId"]) == qid:
                    chosen_answer = accepted_row
                    positive_source = "accepted"
                else:
                    skipped_missing_accepted += 1
            else:
                skipped_missing_accepted += 1

        if chosen_answer is None:
            chosen_answer = choose_best_answer(candidate_answers)
            positive_source = "best_score"

        if chosen_answer is None:
            skipped_no_answer += 1
            continue

        anchor_text = (
            f"{qrow['Title'] if pd.notna(qrow['Title']) else ''} "
            f"{qrow['plain_text'] if pd.notna(qrow['plain_text']) else ''}"
        ).strip()

        if not anchor_text:
            skipped_empty_anchor += 1
            continue

        positive_text = str(chosen_answer["plain_text"]).strip()
        if not positive_text:
            skipped_empty_positive += 1
            continue

        selected_rows.append({
            "anchor_qid": qid,
            "anchor": anchor_text,
            "raw_tags": qrow["Tags"],
            "topic": get_primary_topic(qrow["Tags"]),
            "anchor_op": qrow["operation"],
            "positive_aid": int(chosen_answer["Id"]),
            "positive": positive_text,
            "positive_source": positive_source,
        })
        
    dataset = pd.DataFrame(selected_rows)
    skipped_no_topic = int(dataset["topic"].isna().sum())
    dataset = dataset[dataset["topic"].notna()].copy()

    print(f"  Skipped no topic    : {skipped_no_topic:,}")
    print("\nComputing length stats for full dataset...")
    length_stats_full = compute_length_stats(dataset)

    print(f"  Questions with selected positive + topic: {len(dataset):,}")

    before_topic_filter = len(dataset)
    valid_topics = (dataset.groupby("topic")["anchor_op"].nunique().loc[lambda x: x >= 2].index)
    dataset = dataset[dataset["topic"].isin(valid_topics)].copy()
    dropped_single_op_topics = before_topic_filter - len(dataset)
    
    print("\nOperation distribution:")
    op_counts_selected = dataset["anchor_op"].value_counts().to_dict()
    for op, count in op_counts_selected.items():
        print(f"  {op:<18} {count:,}")
        
    print("\nTopic distribution:")
    topic_counts = dataset["topic"].value_counts().to_dict()
    for topic, count in topic_counts.items():
        print(f"  {topic:<20} {count:,}")        
    
    # Stratify by topic + operation to reduce skew across splits
    dataset["stratify_label"] = dataset["topic"].astype(str) + "||" + dataset["anchor_op"].astype(str)

    # Very small groups are kept, but excluded from stratified splitting
    MIN_COUNT = 5
    label_counts = dataset["stratify_label"].value_counts()

    common_mask = dataset["stratify_label"].map(label_counts) >= MIN_COUNT
    common_df = dataset[common_mask].copy()
    rare_df = dataset[~common_mask].copy()

    # Split only common groups with stratification
    train_df, temp_df = train_test_split(
        common_df,
        test_size=(1.0 - TRAIN_SPLIT),
        random_state=RANDOM_SEED,
        stratify=common_df["stratify_label"]
    )

    # Split remaining temp into val and test
    val_relative = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)

    # Re-check label counts in temp_df for second stratified split
    temp_label_counts = temp_df["stratify_label"].value_counts()
    temp_common_mask = temp_df["stratify_label"].map(temp_label_counts) >= 2

    temp_common_df = temp_df[temp_common_mask].copy()
    temp_rare_df = temp_df[~temp_common_mask].copy()

    if len(temp_common_df) > 0:
        val_df, test_df = train_test_split(
            temp_common_df,
            test_size=(1.0 - val_relative),
            random_state=RANDOM_SEED,
            stratify=temp_common_df["stratify_label"]
        )
    else:
        val_df = pd.DataFrame(columns=temp_df.columns)
        test_df = pd.DataFrame(columns=temp_df.columns)

    # Randomly assign original rare examples to train/val/test
    if len(rare_df) > 0:
        rng = np.random.default_rng(RANDOM_SEED)
        rare_df = rare_df.copy()
        rare_df["split"] = rng.choice(
            ["train", "val", "test"],
            size=len(rare_df),
            p=[TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT]
        )

        train_df = pd.concat([train_df, rare_df[rare_df["split"] == "train"]], ignore_index=True)
        val_df   = pd.concat([val_df,   rare_df[rare_df["split"] == "val"]], ignore_index=True)
        test_df  = pd.concat([test_df,  rare_df[rare_df["split"] == "test"]], ignore_index=True)

    # Randomly assign too-small temp labels to val/test
    if len(temp_rare_df) > 0:
        rng = np.random.default_rng(RANDOM_SEED + 1)
        temp_rare_df = temp_rare_df.copy()
        temp_rare_df["split"] = rng.choice(
            ["val", "test"],
            size=len(temp_rare_df),
            p=[val_relative, 1.0 - val_relative]
        )

        val_df  = pd.concat([val_df,  temp_rare_df[temp_rare_df["split"] == "val"]], ignore_index=True)
        test_df = pd.concat([test_df, temp_rare_df[temp_rare_df["split"] == "test"]], ignore_index=True)

    # Cleanup
    train_df = train_df.drop(columns=["stratify_label", "split"], errors="ignore").reset_index(drop=True)
    val_df   = val_df.drop(columns=["stratify_label", "split"], errors="ignore").reset_index(drop=True)
    test_df  = test_df.drop(columns=["stratify_label", "split"], errors="ignore").reset_index(drop=True)   

    print("\nQuestion-level split:")
    print(f"  Train questions: {len(train_df):,}")
    print(f"  Val questions  : {len(val_df):,}")
    print(f"  Test questions : {len(test_df):,}")
    
    print("\nComputing length stats per split...")

    length_stats_train = compute_length_stats(train_df)
    length_stats_val = compute_length_stats(val_df)
    length_stats_test = compute_length_stats(test_df)    

    train_triplets, train_stats = build_triplets_for_split(train_df, "train")
    val_triplets, val_stats = build_triplets_for_split(val_df, "val")
    test_triplets, test_stats = build_triplets_for_split(test_df, "test")

    save_jsonl(train_triplets, OUT_TRAIN)
    save_jsonl(val_triplets, OUT_VAL)
    save_jsonl(test_triplets, OUT_TEST)  
    
    print(f"\nSaved:")
    print(f"  Train : {len(train_triplets):,} -> {OUT_TRAIN}")
    print(f"  Val   : {len(val_triplets):,}   -> {OUT_VAL}")
    print(f"  Test  : {len(test_triplets):,}  -> {OUT_TEST}")
    
    report = {
        "questions": {
            "loaded_raw": int(n_loaded_questions),
            "with_latex": int(n_with_latex_questions),
            "with_operation_label": int(len(questions)),
            "with_operation_label_pct": round(len(questions) / n_loaded_questions * 100, 2) if n_loaded_questions else 0,
            "final_dataset_questions": int(len(dataset)),
            "final_dataset_pct": round(len(dataset) / n_loaded_questions * 100, 2) if n_loaded_questions else 0,
        },
        "splits": {
            "train_questions": int(len(train_df)),
            "val_questions": int(len(val_df)),
            "test_questions": int(len(test_df)),
        },
        "triplets": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "skipped": {
            "no_answer_for_question": int(skipped_no_answer),
            "accepted_answer_missing_or_mismatched": int(skipped_missing_accepted),
            "empty_anchor": int(skipped_empty_anchor),
            "empty_positive": int(skipped_empty_positive),
            "no_topic_match": int(skipped_no_topic),
            "no_topic_match_pct": round(skipped_no_topic / len(questions) * 100, 2) if len(questions) else 0,
            "dropped_single_op_topics": int(dropped_single_op_topics),
            "dropped_single_op_topics_pct": round(dropped_single_op_topics / before_topic_filter * 100, 2) if before_topic_filter else 0
        },
        "topic_distribution": {
            k: int(v) for k, v in dataset["topic"].value_counts().to_dict().items()
        },          
        "operation_distribution": {
            "questions_labeled": {k: int(v) for k, v in op_counts_all.items()},
            "selected_dataset": {k: int(v) for k, v in op_counts_selected.items()},
        },
        "positive_source_distribution": {
            k: int(v) for k, v in dataset["positive_source"].value_counts().to_dict().items()
        },
        "settings": {
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "test_split": TEST_SPLIT,
            "random_seed": RANDOM_SEED,
            "operation_patterns": OPERATION_PATTERNS,
        },
        "length_statistics": {
            "full_dataset": length_stats_full,
            "train": length_stats_train,
            "val": length_stats_val,
            "test": length_stats_test,
        }       
    }
    
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  Report: {OUT_REPORT}")
    print("\nDone.")


if __name__ == "__main__":
    main()              