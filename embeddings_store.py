import json
from pathlib import Path
from typing import List, Dict, Any

import hashlib
import pandas as pd


DATASET_DIR = Path("Dataset")
EMB_DIR = Path("embeddings")
EMB_DIR.mkdir(exist_ok=True)


def cheap_hash_embedding(text: str, dim: int = 64) -> List[float]:
    vec = [0] * dim
    if not isinstance(text, str):
        return [0.0] * dim
    tokens = text.lower().split()
    for tok in tokens:
        h = int(hashlib.sha1(tok.encode("utf-8", errors="ignore")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1
    norm = max(1.0, sum(abs(v) for v in vec))
    return [round(v / norm, 6) for v in vec]


def index_articles(enriched_csv: Path, out_jsonl: Path) -> None:
    df = pd.read_csv(enriched_csv)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = f"{row.get('title', '')} {row.get('text', '')}"
            emb = cheap_hash_embedding(text)
            rec: Dict[str, Any] = {
                'id': row.get('id', ''),
                'type': 'article',
                'text': text[:1000],
                'embedding': emb,
                'metadata': {
                    'source': row.get('source', ''),
                    'published_at': row.get('published_at', ''),
                    'risk_labels': row.get('risk_labels', '[]'),
                    'severity': row.get('severity_0_10', 0.0),
                }
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Indexed articles -> {out_jsonl}")


def index_events(events_csv: Path, out_jsonl: Path) -> None:
    df = pd.read_csv(events_csv)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = f"{row.get('headline', '')} | labels: {row.get('risk_labels', '[]')} | nodes: {row.get('impacted_nodes', '[]')}"
            emb = cheap_hash_embedding(text)
            rec: Dict[str, Any] = {
                'id': row.get('event_id', ''),
                'type': 'event',
                'text': text[:1000],
                'embedding': emb,
                'metadata': {
                    'detected_at': row.get('detected_at', ''),
                    'severity': row.get('severity', 0.0),
                    'confidence': row.get('confidence', 0.5),
                }
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Indexed events -> {out_jsonl}")


if __name__ == "__main__":
    index_articles(DATASET_DIR / "train_combined_enriched.csv", EMB_DIR / "articles_train.jsonl")
    index_articles(DATASET_DIR / "test_combined_enriched.csv", EMB_DIR / "articles_test.jsonl")
    index_events(DATASET_DIR / "train_impact_events.csv", EMB_DIR / "events_train.jsonl")
    index_events(DATASET_DIR / "test_impact_events.csv", EMB_DIR / "events_test.jsonl")



