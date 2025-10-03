import pandas as pd
from pathlib import Path
from typing import List

from config import SUPPLY_CHAIN_KEYWORDS, STRATEGIC_KEYWORDS, RISK_CATEGORIES


DATASET_DIR = Path("Dataset")


def compute_risk_score(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0

    content = text.lower()

    supply_hits = sum(1 for kw in SUPPLY_CHAIN_KEYWORDS if kw in content)
    strategic_hits = sum(1 for kw in STRATEGIC_KEYWORDS if kw in content)

    category_hits = 0
    for domain in RISK_CATEGORIES.values():
        for keywords in domain.values():
            category_hits += sum(1 for kw in keywords if kw.lower() in content)

    raw_score = supply_hits + strategic_hits + 0.5 * category_hits

    if raw_score <= 0:
        return 0.0

    # Normalize to 0..1 with a soft cap
    normalized = min(raw_score / 10.0, 1.0)
    return float(round(normalized, 4))


def add_labels(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    if df.empty:
        return df

    def row_text(row) -> str:
        parts = []
        for col in text_columns:
            val = row.get(col, "") if hasattr(row, "get") else row[col] if col in row else ""
            if isinstance(val, str):
                parts.append(val)
        return " ".join(parts)

    scores = []
    for _, row in df.iterrows():
        text = row_text(row)
        score = compute_risk_score(text)
        scores.append(score)

    df = df.copy()
    df["risk_score"] = scores
    df["risk_flag"] = (df["risk_score"] >= 0.3).astype(int)
    return df


def label_file(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Try common text fields present in provided datasets
    candidate_cols = [
        "Clean_summary",  # provided in your CSVs
        "Summary",
        "Title",
        "title",
        "content",
        "description",
    ]
    text_cols = [c for c in candidate_cols if c in df.columns]
    if not text_cols:
        # Fallback to all object dtype columns
        text_cols = [c for c in df.columns if df[c].dtype == object]

    labeled = add_labels(df, text_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)
    print(f"Saved labeled dataset -> {output_path}")


if __name__ == "__main__":
    train_in = DATASET_DIR / "train_combined.csv"
    test_in = DATASET_DIR / "test_combined.csv"

    train_out = DATASET_DIR / "train_combined_labeled.csv"
    test_out = DATASET_DIR / "test_combined_labeled.csv"

    label_file(train_in, train_out)
    label_file(test_in, test_out)


