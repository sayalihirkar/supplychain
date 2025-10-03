from pathlib import Path
from typing import Dict, Any, List

import json
import numpy as np
import joblib


MODELS_DIR = Path("models")


_clf_bundle = None
_reg_bundle = None


def _load_models() -> None:
    global _clf_bundle, _reg_bundle
    if _clf_bundle is None:
        _clf_bundle = joblib.load(MODELS_DIR / 'risk_classifier.joblib')
    if _reg_bundle is None:
        _reg_bundle = joblib.load(MODELS_DIR / 'severity_regressor.joblib')


def predict(title: str, text: str, source: str = "") -> Dict[str, Any]:
    _load_models()
    vectorizer = _clf_bundle['vectorizer']
    mlb = _clf_bundle['mlb']
    clf = _clf_bundle['clf']
    reg = _reg_bundle['regressor']

    combined = f"{title or ''} {text or ''}"
    X = vectorizer.transform([combined])
    y_pred = clf.predict(X)
    y_proba = None
    try:
        y_proba = clf.decision_function(X)
    except Exception:
        pass

    labels: List[str] = mlb.inverse_transform(y_pred)[0] if y_pred is not None else []
    label_probs: Dict[str, float] = {}
    if y_proba is not None and y_proba.shape[1] == len(mlb.classes_):
        probs = 1.0 / (1.0 + np.exp(-y_proba[0]))
        for cls, p in zip(mlb.classes_, probs):
            label_probs[cls] = float(p)

    # Derive heuristic features similar to pipeline_enrichment
    from label_dataset import compute_risk_score
    from pipeline_enrichment import (
        heuristic_source_reliability,
        temporal_urgency,
        location_proximity,
        detect_events,
        extract_entities,
        load_nlp,
        categorize_multi_label,
    )
    txt = (combined or "").lower()
    risk_score = float(compute_risk_score(txt))
    src_rel = float(heuristic_source_reliability(source))
    urg = float(temporal_urgency(txt))
    prox = float(location_proximity(txt))
    events = detect_events(txt)
    nlp = load_nlp()
    ents = extract_entities(nlp, txt)
    auto_labels = categorize_multi_label(txt, ents)
    # Merge heuristic labels with classifier labels for display
    merged_labels = list(dict.fromkeys(list(labels) + list(auto_labels)))

    # Features for regressor match engineer_features in training
    feat = np.asarray([[risk_score,  # risk_score
                        0.0,         # severity_0_10 placeholder (not used)
                        src_rel,     # source_reliability
                        0.0,         # social_score (unknown live)
                        urg,         # temporal_urgency
                        prox         # location_proximity
                        ]], dtype=float)
    sev = float(reg.predict(feat)[0])
    sev = max(0.0, min(10.0, sev))

    risk_level = 'low'
    if sev >= 8.0:
        risk_level = 'critical'
    elif sev >= 5.0:
        risk_level = 'high'
    elif sev >= 3.0:
        risk_level = 'medium'

    # Impacted nodes (lightweight): link entities to nodes
    try:
        from supply_mapping import load_kb, link_entities_to_nodes
        nodes_df, _ = load_kb()
        links = link_entities_to_nodes(ents, nodes_df)
        impacted_nodes = [l['node_id'] for l in links]
    except Exception:
        impacted_nodes = []

    return {
        'risk_labels': merged_labels,
        'label_probabilities': label_probs,
        'severity_0_10': round(sev, 2),
        'risk_level': risk_level,
        'events': events,
        'entities': ents,
        'impacted_nodes': impacted_nodes
    }


if __name__ == '__main__':
    demo = predict("Jaguar Land Rover hit by cyber attack", "Production halted; suppliers impacted")
    print(json.dumps(demo, indent=2))


