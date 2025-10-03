from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import json

from rag_retriever import explain as rag_explain
import os
import google.generativeai as genai  # type: ignore


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


