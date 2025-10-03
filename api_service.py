from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import json

from rag_retriever import explain as rag_explain
import os
import google.generativeai as genai  # type: ignore
from inference_local import predict as local_predict
from data_ingestion import NewsFetcher, NewsProcessor


DATASET_DIR = Path("Dataset")
KB_DIR = Path("supply_kb")


class ArticleIn(BaseModel):
    id: str
    source: str
    published_at: str
    title: str
    text: str
    url: str
    tags: List[str] | None = None


app = FastAPI(title="Risk Intelligence API")


class AckIn(BaseModel):
    user: str | None = None
    comment: str | None = None


class AssignIn(BaseModel):
    assignee: str
    comment: str | None = None


@app.post("/ingest")
def ingest_article(article: ArticleIn) -> Dict[str, Any]:
    # Prototype stub: append to a staging file
    staging = DATASET_DIR / "incoming_articles.jsonl"
    with staging.open("a", encoding="utf-8") as f:
        f.write(json.dumps(article.dict()) + "\n")
    return {"status": "queued", "id": article.id}


@app.post("/predict")
def predict_article(body: ArticleIn) -> Dict[str, Any]:
    res = local_predict(body.title, body.text, source=body.source)
    return {"id": body.id, **res}


@app.get("/news/search")
def news_search(q: str = "Tata Motors", from_date: str | None = None, to_date: str | None = None) -> List[Dict[str, Any]]:
    fetcher = NewsFetcher()
    processor = NewsProcessor()
    arts = fetcher.fetch_news_api(q, from_date=from_date, to_date=to_date)
    df = processor.process_articles(arts)
    results: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        results.append({
            'id': r.get('url', ''),
            'source': r.get('source', ''),
            'published_at': r.get('published_at', ''),
            'title': r.get('title', ''),
            'text': r.get('content', ''),
            'url': r.get('url', ''),
            'tags': list(filter(None, [t.strip() for t in str(r.get('supply_chain_keywords','')).split(',') + str(r.get('strategic_keywords','')).split(',')]))
        })
    return results


class CreateEventIn(BaseModel):
    event_id: str
    title: str
    severity: float
    risk_labels: list[str]
    impacted_nodes: list[str]
    evidence: list[str]
    confidence: float | None = 0.7


@app.post("/events/create")
def events_create(body: CreateEventIn) -> Dict[str, Any]:
    ev_path = DATASET_DIR / "live_events.csv"
    import pandas as pd
    rec = {
        'event_id': body.event_id,
        'detected_at': pd.Timestamp.utcnow().isoformat(),
        'headline': body.title,
        'risk_labels': json.dumps(body.risk_labels),
        'severity': body.severity,
        'impacted_nodes': json.dumps(body.impacted_nodes),
        'evidence': json.dumps(body.evidence),
        'confidence': body.confidence or 0.7,
    }
    if ev_path.exists():
        df = pd.read_csv(ev_path)
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    else:
        df = pd.DataFrame([rec])
    df.to_csv(ev_path, index=False)
    return {"status": "created", "event_id": body.event_id}


@app.get("/events")
def get_events(limit: int = 100) -> List[Dict[str, Any]]:
    df = pd.read_csv(DATASET_DIR / "test_impact_events.csv")
    df = df.sort_values("severity", ascending=False).head(limit)
    return df.to_dict(orient="records")


@app.get("/events/{event_id}/explain")
def explain_event(event_id: str) -> Dict[str, Any]:
    df = pd.read_csv(DATASET_DIR / "test_impact_events.csv")
    row = df[df["event_id"] == event_id]
    if row.empty:
        return {"error": "not_found"}
    title = str(row.iloc[0].get("headline", ""))
    # Prefer backend RAG with Gemini if available
    summary = rag_explain(title, top_k=5)
    return {"event_id": event_id, "rag_summary": summary}


@app.get("/nodes")
def get_nodes() -> List[Dict[str, Any]]:
    nodes = pd.read_csv(KB_DIR / "nodes.csv").to_dict(orient="records")
    # Attach latest severity per node if available via events
    try:
        ev = pd.read_csv(DATASET_DIR / "test_impact_events.csv")
        ev["impacted_nodes_list"] = ev["impacted_nodes"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
        node_to_sev: Dict[str, float] = {}
        for _, r in ev.iterrows():
            for nid in r["impacted_nodes_list"]:
                node_to_sev[nid] = max(node_to_sev.get(nid, 0.0), float(r.get("severity", 0.0)))
        for n in nodes:
            n["severity"] = node_to_sev.get(n["node_id"], 0.0)
    except Exception:
        for n in nodes:
            n["severity"] = 0.0
    return nodes


@app.post("/events/{event_id}/ack")
def acknowledge_event(event_id: str, body: AckIn) -> Dict[str, Any]:
    log_path = DATASET_DIR / "events_actions.jsonl"
    rec = {"event_id": event_id, "action": "acknowledge", "user": body.user, "comment": body.comment}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return {"status": "ok", "event_id": event_id}


@app.post("/events/{event_id}/assign")
def assign_event(event_id: str, body: AssignIn) -> Dict[str, Any]:
    log_path = DATASET_DIR / "events_actions.jsonl"
    rec = {"event_id": event_id, "action": "assign", "assignee": body.assignee, "comment": body.comment}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return {"status": "ok", "event_id": event_id, "assignee": body.assignee}


