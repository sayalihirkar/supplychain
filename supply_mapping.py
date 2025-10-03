import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd


KB_DIR = Path("supply_kb")
DATASET_DIR = Path("Dataset")


def load_kb() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(KB_DIR / "nodes.csv")
    edges = pd.read_csv(KB_DIR / "edges.csv")
    return nodes, edges


def simple_similarity(a: str, b: str) -> float:
    a_l = a.lower()
    b_l = b.lower()
    if a_l in b_l or b_l in a_l:
        return 1.0
    overlap = len(set(a_l.split()) & set(b_l.split()))
    denom = max(1, len(set(a_l.split())) + len(set(b_l.split())))
    return overlap * 2.0 / denom


def link_entities_to_nodes(entities: List[Dict[str, Any]], nodes_df: pd.DataFrame) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for ent in entities:
        text = ent.get("text", "")
        best: Tuple[str, float] = ("", 0.0)
        for _, node in nodes_df.iterrows():
            cand = f"{node['node_id']} {node['location']} {node.get('materials_handled', '')}"
            sim = simple_similarity(text, cand)
            if sim > best[1]:
                best = (node['node_id'], sim)
        if best[1] >= 0.6:
            links.append({"entity": text, "node_id": best[0], "confidence": round(best[1], 2)})
    return links


def propagate_impacts(seed_nodes: List[str], edges_df: pd.DataFrame, depth: int = 2) -> List[Tuple[str, float]]:
    impacts: Dict[str, float] = {}
    frontier = [(n, 1.0) for n in seed_nodes]
    for _ in range(depth):
        next_frontier: List[Tuple[str, float]] = []
        for node_id, strength in frontier:
            for _, e in edges_df.iterrows():
                if e['from_node'] == node_id:
                    weight = float(e.get('weight', 0.5))
                    new_strength = strength * weight
                    tgt = e['to_node']
                    if new_strength > impacts.get(tgt, 0.0):
                        impacts[tgt] = new_strength
                    next_frontier.append((tgt, new_strength))
        frontier = next_frontier
    return sorted(impacts.items(), key=lambda x: x[1], reverse=True)


def build_impact_events(enriched_csv: Path, out_csv: Path) -> None:
    nodes, edges = load_kb()
    df = pd.read_csv(enriched_csv)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        ents = []
        try:
            ents_obj = json.loads(row.get('entities', '{}'))
            ents = ents_obj.get('entities', []) if isinstance(ents_obj, dict) else []
        except Exception:
            pass
        links = link_entities_to_nodes(ents, nodes)
        linked_node_ids = [l['node_id'] for l in links]

        # Also link via materials keywords
        text = str(row.get('text', '')).lower()
        if 'lithium' in text:
            linked_node_ids.append('material_lithium')
        if 'cobalt' in text:
            linked_node_ids.append('material_cobalt')

        # Propagate impacts
        propagated = propagate_impacts(list(set(linked_node_ids)), edges, depth=2)

        impact_nodes = [n for n, s in propagated if s >= 0.3] or linked_node_ids

        rec = {
            'event_id': row.get('id', ''),
            'detected_at': row.get('published_at', ''),
            'headline': row.get('Title', row.get('title', '')),
            'risk_labels': row.get('risk_labels', '[]'),
            'severity': row.get('severity_0_10', 0.0),
            'impacted_nodes': json.dumps(impact_nodes),
            'evidence': json.dumps([row.get('id', '')]),
            'confidence': min(0.95, 0.5 + 0.05 * len(impact_nodes)),
        }
        records.append(rec)
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved impact events -> {out_csv}")


if __name__ == "__main__":
    build_impact_events(DATASET_DIR / "train_combined_enriched.csv", DATASET_DIR / "train_impact_events.csv")
    build_impact_events(DATASET_DIR / "test_combined_enriched.csv", DATASET_DIR / "test_impact_events.csv")



