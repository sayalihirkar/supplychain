import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

EMB_DIR = Path("embeddings")

import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def retrieve(query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
    # Use same cheap hash embedding as store to keep compatibility
    from embeddings_store import cheap_hash_embedding

    q_emb = cheap_hash_embedding(query)
    corpora = [
        load_jsonl(EMB_DIR / "articles_train.jsonl"),
        load_jsonl(EMB_DIR / "articles_test.jsonl"),
        load_jsonl(EMB_DIR / "events_train.jsonl"),
        load_jsonl(EMB_DIR / "events_test.jsonl"),
    ]
    results: List[Tuple[float, Dict[str, Any]]] = []
    for items in corpora:
        for item in items:
            sim = cosine(q_emb, item.get("embedding", []))
            results.append((sim, item))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def explain(query: str, top_k: int = 3) -> str:
    hits = retrieve(query, top_k)
    if not GEMINI_API_KEY:
        lines = [f"### Event Explanation", "", f"**Query**: {query}", "", "**Top Evidence:**"]
        for score, item in hits:
            meta = item.get("metadata", {})
            lines.append(
                f"- **{item.get('type')}** (sim {score:.2f}) — {item.get('text','')[:180]}..."
            )
        lines.extend([
            "",
            "**Suggested actions**:",
            "- Validate impacted nodes",
            "- Contact critical suppliers",
            "- Monitor policy channels",
        ])
        return "\n".join(lines)

    # Gemini-backed summary
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        context_items = []
        for score, item in hits:
            context_items.append({
                'text': item.get('text', '')[:1200],
                'meta': item.get('metadata', {}),
                'score': round(float(score), 3)
            })
        prompt = (
            "You are a risk analyst assistant. Summarize the risk event based on retrieved evidence.\n"
            "Output as clean Markdown with sections: \n"
            "### Summary\n### Impacted Nodes\n### Risk Labels\n### Severity Rationale\n### Immediate Actions\n"
            "Use short bullet points.\n"
            f"Query: {query}\nEvidence: {context_items}"
        )
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text.startswith("### Summary"):
            # Wrap into standard sections if model returned free text
            bullets = [
                "### Summary",
                text,
                "",
                "### Immediate Actions",
                "- Validate impacted nodes",
                "- Contact critical suppliers",
                "- Monitor policy channels",
            ]
            return "\n".join(bullets)
        return text
    except Exception as ex:
        # Fallback to local explanation if Gemini errors
        lines = [f"### Event Explanation", "", f"**Query**: {query}", f"_Gemini error: {ex}_", "", "**Top Evidence:**"]
        for score, item in hits:
            meta = item.get("metadata", {})
            lines.append(
                f"- **{item.get('type')}** (sim {score:.2f}) — {item.get('text','')[:180]}..."
            )
        return "\n".join(lines)


if __name__ == "__main__":
    print(explain("chip shortage hits port kandla battery lines at pune"))

