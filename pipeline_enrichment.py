import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None

from config import SUPPLY_CHAIN_KEYWORDS, STRATEGIC_KEYWORDS, RISK_CATEGORIES


DATASET_DIR = Path("Dataset")


@dataclass
class CanonicalArticle:
    id: str
    source: str
    published_at: str
    title: str
    text: str
    url: str
    tags: List[str]
    score: float = 0.0  # optional social score
    subreddit: str = ""  # optional subreddit


def normalize_schema(row: pd.Series) -> CanonicalArticle:
    title_val = row.get("Title") or row.get("title") or ""
    text_val = row.get("Clean_summary") or row.get("Summary") or row.get("content") or row.get("description") or ""
    source_val = row.get("Source") or row.get("source") or "Unknown"
    url_val = row.get("url") or row.get("URL") or ""
    published_val = row.get("Published") or row.get("published_at") or row.get("publishedAt") or ""

    title = str(title_val) if pd.notna(title_val) else ""
    text = str(text_val) if pd.notna(text_val) else ""
    source = str(source_val) if pd.notna(source_val) else "Unknown"
    url = str(url_val) if pd.notna(url_val) else ""
    published = str(published_val) if pd.notna(published_val) else ""

    raw_id = f"{source}|{published}|{title[:128]}|{text[:256]}"
    stable_id = hashlib.sha1(raw_id.encode("utf-8", errors="ignore")).hexdigest()

    tags: List[str] = []
    if "Category" in row and pd.notna(row["Category"]):
        cat = str(row["Category"]).strip()
        if cat:
            tags.append(cat)

    score = float(row.get("score", 0.0)) if pd.notna(row.get("score", None)) else 0.0
    subreddit = str(row.get("subreddit", "")) if pd.notna(row.get("subreddit", None)) else ""

    return CanonicalArticle(
        id=stable_id,
        source=source,
        published_at=published,
        title=title,
        text=text,
        url=url,
        tags=tags,
        score=score,
        subreddit=subreddit,
    )


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    return t


def heuristic_source_reliability(source: str) -> float:
    if not source:
        return 0.3
    trusted = {"bbc news", "reuters", "bloomberg", "the times of india", "businessline", "forbes"}
    mid = {"yahoo entertainment", "autoblog", "gizmodo.com", "cna", "just-auto.com"}
    s = source.strip().lower()
    if s in trusted:
        return 0.9
    if s in mid:
        return 0.7
    return 0.5


def dedup_hash(title: str, text: str) -> str:
    return hashlib.md5((title + "|" + text).encode("utf-8", errors="ignore")).hexdigest()


def load_nlp() -> Any:
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


MATERIAL_TERMS = {"lithium", "cobalt", "nickel", "graphite", "manganese", "steel", "aluminum", "rubber"}
COMPONENT_TERMS = {"semiconductor", "chip", "soc", "mcu", "battery", "cell", "anode", "cathode"}
EVENT_TERMS = {
    "strike": "PORT_STRIKE",
    "port strike": "PORT_STRIKE",
    "embargo": "LOGISTICS_DISRUPTION",
    "shortage": "CHIP_SHORTAGE",
    "chip shortage": "CHIP_SHORTAGE",
    "policy": "POLICY_CHANGE",
    "subsidy": "POLICY_CHANGE",
    "tariff": "POLICY_CHANGE",
    "shutdown": "LOGISTICS_DISRUPTION",
    "cyberattack": "LOGISTICS_DISRUPTION",
    "cyber attack": "LOGISTICS_DISRUPTION",
    "price spike": "RAW_MATERIAL_PRICE_SPIKE",
    "price surge": "RAW_MATERIAL_PRICE_SPIKE",
}
POLICY_TERMS = {"ev policy", "regulation", "policy", "subsidy", "emission", "incentive", "tariff"}
SUPPLYNODE_TERMS = {"plant", "factory", "assembly", "port", "warehouse", "logistics hub"}


def extract_entities(nlp, text: str) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    if not text:
        return entities
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            label = ent.label_
            mapped = None
            if label in {"ORG"}:
                mapped = "ORGANIZATION"
            elif label in {"GPE", "LOC"}:
                mapped = "LOCATION"
            elif label in {"LAW"}:
                mapped = "POLICY"
            elif label in {"DATE"}:
                mapped = "DATE"
            if mapped:
                entities.append({"text": ent.text, "label": mapped, "start": ent.start_char, "end": ent.end_char, "confidence": 0.7})

    # Keyword-boosted domain entities
    for term in MATERIAL_TERMS:
        if term in text:
            entities.append({"text": term, "label": "MATERIAL", "start": text.find(term), "end": text.find(term) + len(term), "confidence": 0.6})
    for term in COMPONENT_TERMS:
        if term in text:
            entities.append({"text": term, "label": "COMPONENT", "start": text.find(term), "end": text.find(term) + len(term), "confidence": 0.6})
    for term in POLICY_TERMS:
        if term in text:
            entities.append({"text": term, "label": "POLICY", "start": text.find(term), "end": text.find(term) + len(term), "confidence": 0.55})
    for term in SUPPLYNODE_TERMS:
        if term in text:
            entities.append({"text": term, "label": "SUPPLYNODE", "start": text.find(term), "end": text.find(term) + len(term), "confidence": 0.55})

    return entities


def detect_events(text: str) -> List[str]:
    events: List[str] = []
    if not text:
        return events
    for k, v in EVENT_TERMS.items():
        if k in text:
            if v not in events:
                events.append(v)
    return events


def categorize_multi_label(text: str, entities: List[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    if any(term in text for term in ["lithium", "cobalt", "nickel", "raw material", "shortage"]):
        labels.append("SupplyChain:RawMaterials")
    if any(term in text for term in ["port", "shipping", "logistics", "freight", "embargo", "strike"]):
        labels.append("SupplyChain:Logistics")
    if any(term in text for term in ["supplier", "vendor", "plant", "factory"]):
        labels.append("SupplyChain:Vendors")
    if any(term in text for term in ["policy", "regulation", "tariff", "subsidy", "emission"]):
        labels.append("Strategic:Regulatory")
    if any(term in text for term in ["competition", "market share", "price war"]):
        labels.append("Strategic:Competition")
    if any(term in text for term in ["sustainability", "climate", "carbon"]):
        labels.append("Strategic:Sustainability")
    if any(term in text for term in ["ev", "electric vehicle", "battery"]):
        labels.append("Market:EV_Adoption")
    return list(dict.fromkeys(labels))


def temporal_urgency(text: str) -> float:
    if any(tok in text for tok in ["will", "planned", "expects", "may", "could"]):
        return 0.4  # future → lower urgency
    if any(tok in text for tok in ["halted", "disrupted", "impacts", "hit", "shutdown", "strike"]):
        return 1.0  # ongoing → high urgency
    return 0.7


def location_proximity(text: str) -> float:
    india_terms = ["india", "pune", "mumbai", "jamshedpur", "gujarat"]
    uk_terms = ["uk", "solihull", "halewood", "wolverhampton", "birmingham"]
    if any(t in text for t in india_terms):
        return 1.0
    if any(t in text for t in uk_terms):
        return 0.8
    return 0.6


def severity_score(row: Dict[str, Any]) -> float:
    base = float(row.get("risk_score", 0.0)) if pd.notna(row.get("risk_score", 0.0)) else 0.0
    rel = float(row.get("source_reliability", 0.5))
    social = float(row.get("social_score", 0.0))
    urgency = float(row.get("temporal_urgency", 0.7))
    proximity = float(row.get("location_proximity", 0.6))
    event_boost = 0.2 * len(row.get("events", []))
    label_boost = 0.1 * len(row.get("risk_labels", []))

    score_0_1 = min(1.0, base * 0.5 + rel * 0.15 + (social / 100.0) * 0.1 + urgency * 0.15 + proximity * 0.1 + event_boost + label_boost)
    return round(score_0_1 * 10.0, 2)  # 0..10


def enrich_dataframe(df: pd.DataFrame, nlp) -> pd.DataFrame:
    if df.empty:
        return df

    # Build canonical fields
    records: List[Dict[str, Any]] = []
    seen_hashes = set()
    for _, row in df.iterrows():
        article = normalize_schema(row)
        text_prep = preprocess_text(article.text)
        dupe_key = dedup_hash(article.title.lower(), text_prep)
        if dupe_key in seen_hashes:
            continue
        seen_hashes.add(dupe_key)

        source_reliability = heuristic_source_reliability(article.source)
        social_score = article.score

        ents = extract_entities(nlp, text_prep)
        events = detect_events(text_prep)
        risk_labels = categorize_multi_label(text_prep, ents)

        urgency = temporal_urgency(text_prep)
        proximity = location_proximity(text_prep)

        rec: Dict[str, Any] = row.to_dict()
        rec.update({
            "id": article.id,
            "source": article.source,
            "published_at": article.published_at,
            "title": article.title,
            "text": article.text,
            "url": article.url,
            "tags": ",".join(article.tags),
            "source_reliability": source_reliability,
            "social_score": social_score,
            "entities": json.dumps({"entities": ents}, ensure_ascii=False),
            "events": json.dumps(events),
            "risk_labels": json.dumps(risk_labels),
            "temporal_urgency": urgency,
            "location_proximity": proximity,
        })
        rec["severity_0_10"] = severity_score(rec)
        records.append(rec)

    return pd.DataFrame.from_records(records)


def run_pipeline(input_csv: Path, output_csv: Path, output_jsonl: Path) -> None:
    if not input_csv.exists():
        print(f"Missing input {input_csv}")
        return
    df = pd.read_csv(input_csv)
    nlp = load_nlp()
    enriched = enrich_dataframe(df, nlp)
    enriched.to_csv(output_csv, index=False)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for _, row in enriched.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"Saved enriched CSV -> {output_csv}")
    print(f"Saved enriched JSONL -> {output_jsonl}")


if __name__ == "__main__":
    train_in = DATASET_DIR / "train_combined_labeled.csv"
    test_in = DATASET_DIR / "test_combined_labeled.csv"

    run_pipeline(train_in, DATASET_DIR / "train_combined_enriched.csv", DATASET_DIR / "train_combined_enriched.jsonl")
    run_pipeline(test_in, DATASET_DIR / "test_combined_enriched.csv", DATASET_DIR / "test_combined_enriched.jsonl")


