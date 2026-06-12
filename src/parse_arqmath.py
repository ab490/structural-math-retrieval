import csv
import sys
import json
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from constants import DATA_DIR

RAW_DIR   = DATA_DIR / "raw" / "arqmath"
POSTS_XML = RAW_DIR / "Posts.V1.3.xml"

OUT_ANSWERS   = DATA_DIR / "processed" / "arqmath_answers.parquet"
OUT_QUESTIONS = DATA_DIR / "processed" / "arqmath_questions.parquet"

MIN_SCORE = 0              # filter out posts with more downvotes (low quality posts)
MIN_TEXT_LENGTH = 50       # filter out posts with very short text (not useful for training)

FIELDS_ANSWERS   = ["Id", "ParentId", "Score", "plain_text", "latex"]
FIELDS_QUESTIONS = ["Id", "Score", "Title", "Tags", "AcceptedAnswerId", "plain_text", "latex"]

def parse_body(html_body):
    """
    Returns:
        plain_text        - full text with HTML stripped
        latex             - pipe-separated LaTeX strings from math-container spans
    """    
    soup = BeautifulSoup(html_body, "html.parser")
    # Extract LaTeX from math-container spans before stripping HTML
    latex = [span.get_text(strip=True) for span in soup.find_all("span", class_="math-container")]
    plain_text = " ".join(soup.get_text(separator=" ").split())
    latex_str  = " | ".join(latex)
    return plain_text, latex_str


def parse_tags(raw_tags):
    """
    Converts '<calculus><integration><derivatives>'
    to       'calculus|integration|derivatives'
    """    
    if not raw_tags:
        return ""
    return raw_tags.replace("><", "|").strip("<>")


def count_rows(xml_path):
    print("Counting rows ...")
    count = sum(1 for _, elem in ET.iterparse(xml_path, events=("end",)) if elem.tag == "row")
    print(f"  {count:,} rows found")
    return count


def main():
    if not POSTS_XML.exists():
        raise FileNotFoundError(f"Posts XML not found at {POSTS_XML}")

    OUT_ANSWERS.parent.mkdir(parents=True, exist_ok=True)

    # counters
    n_total = 0
    n_answers = 0
    n_questions = 0
    n_skipped_score = 0
    n_skipped_latex = 0
    n_skipped_length = 0
    
    answers_rows = []
    questions_rows = []    

    total = count_rows(POSTS_XML)
    
    with tqdm(total=total, desc="Parsing posts", unit="post", dynamic_ncols=True) as pbar:
        for _, elem in ET.iterparse(POSTS_XML, events=("end",)):
            if elem.tag != "row":
                continue

            n_total += 1
            pbar.update(1)

            post_type = elem.attrib.get("PostTypeId", "")
            try:
                score = int(elem.attrib.get("Score", 0))
            except ValueError:
                score = 0

            body = elem.attrib.get("Body", "")    

            # Filter 1: score must be above threshold
            if score < MIN_SCORE:
                n_skipped_score += 1
                elem.clear()
                pbar.set_postfix(kept=n_answers+n_questions, skipped=n_total-(n_answers+n_questions), refresh=False)
                continue

            plain_text, latex_str = parse_body(body)

            # Filter 2: must have at least one LaTeX expression 
            if not latex_str.strip():
                n_skipped_latex += 1
                elem.clear()
                pbar.set_postfix(kept=n_answers+n_questions, skipped=n_total-(n_answers+n_questions), refresh=False)
                continue

            # Filter 3: plain text must be long enough
            if len(plain_text) < MIN_TEXT_LENGTH:
                n_skipped_length += 1
                elem.clear()
                pbar.set_postfix(kept=n_answers+n_questions, skipped=n_total-(n_answers+n_questions), refresh=False)
                continue

            if post_type == "2":      # answer
                answers_rows.append({
                    "Id":                elem.attrib.get("Id", ""),
                    "ParentId":          elem.attrib.get("ParentId", ""),
                    "Score":             score,
                    "plain_text":        plain_text,
                    "latex":             latex_str,
                })
                n_answers += 1
                
            elif post_type == "1":     # question
                raw_title = elem.attrib.get("Title", "")
                clean_title = BeautifulSoup(raw_title, "html.parser").get_text(separator=" ").strip() if raw_title else ""                
                questions_rows.append({
                    "Id":               elem.attrib.get("Id", ""),
                    "Score":            score,
                    "Title":            clean_title,
                    "Tags":             parse_tags(elem.attrib.get("Tags", "")),
                    "AcceptedAnswerId": elem.attrib.get("AcceptedAnswerId", ""),
                    "plain_text":       plain_text,
                    "latex":            latex_str,
                })
                n_questions += 1

            elem.clear()
            pbar.set_postfix(q=n_questions, a=n_answers, skipped=n_total-(n_answers+n_questions), refresh=False)
            
            
    print("\nConverting to DataFrames ...")
    answers_df = pd.DataFrame(answers_rows, columns=FIELDS_ANSWERS)
    questions_df = pd.DataFrame(questions_rows, columns=FIELDS_QUESTIONS)

    print("Saving Parquet files ...")
    answers_df.to_parquet(OUT_ANSWERS, index=False)
    questions_df.to_parquet(OUT_QUESTIONS, index=False)

    # final report
    kept = n_answers + n_questions
    n_skipped = n_skipped_score + n_skipped_latex + n_skipped_length
                
    report = {
        "timestamp": datetime.now().isoformat(),
        "source": str(POSTS_XML),
        "total_rows_seen": n_total,
        "kept": {
            "total": kept,
            "questions": n_questions,
            "answers": n_answers,
            "pct_of_total": round(kept / n_total * 100, 2) if n_total else 0
        },
        "skipped": {
            "total": n_skipped,
            "low_score": n_skipped_score,
            "no_latex": n_skipped_latex,
            "too_short": n_skipped_length,
            "pct_of_total": round(n_skipped / n_total * 100, 2) if n_total else 0
        },
        "filters_used": {
            "min_score": MIN_SCORE,
            "min_text_length": MIN_TEXT_LENGTH,
            "requires_latex": True
        },
        "outputs": {
            "answers": str(OUT_ANSWERS),
            "questions": str(OUT_QUESTIONS)
        }
    }

    report_path = DATA_DIR / "processed" / "parse_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nParse complete.")
    print(f"Answers saved to:   {OUT_ANSWERS}")
    print(f"Questions saved to: {OUT_QUESTIONS}")
    print(f"Report saved to:    {report_path}")


if __name__ == "__main__":
    main()