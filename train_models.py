import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor


DATASET_DIR = Path("Dataset")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
for d in [MODELS_DIR, REPORTS_DIR]:
    d.mkdir(exist_ok=True)


def load_enriched(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def parse_labels(col: pd.Series) -> List[List[str]]:
    labels: List[List[str]] = []
    for v in col.fillna("[]").tolist():
        try:
            parsed = json.loads(v)
            labels.append(parsed if isinstance(parsed, list) else [])
        except Exception:
            labels.append([])
    return labels


def build_text(df: pd.DataFrame) -> List[str]:
    texts: List[str] = []
    for _, r in df.iterrows():
        t = f"{r.get('title','')} {r.get('text','')}"
        texts.append(str(t))
    return texts


def engineer_features(df: pd.DataFrame) -> np.ndarray:
    cols = [
        'risk_score', 'severity_0_10', 'source_reliability', 'social_score',
        'temporal_urgency', 'location_proximity'
    ]
    feat = []
    for _, r in df.iterrows():
        feat.append([
            float(r.get(c, 0.0)) if pd.notna(r.get(c, 0.0)) else 0.0 for c in cols
        ])
    return np.asarray(feat, dtype=float)


def train_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    y_train_labels = parse_labels(train_df['risk_labels'])
    y_test_labels = parse_labels(test_df['risk_labels'])
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(y_train_labels)
    Y_test = mlb.transform(y_test_labels)

    X_train_text = build_text(train_df)
    X_test_text = build_text(test_df)
    tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1, 2))
    Xtr = tfidf.fit_transform(X_train_text)
    Xte = tfidf.transform(X_test_text)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
    clf.fit(Xtr, Y_train)
    Y_pred = clf.predict(Xte)
    Y_proba = np.clip(clf.decision_function(Xte), -10, 10)

    micro_f1 = f1_score(Y_test, Y_pred, average='micro')
    macro_f1 = f1_score(Y_test, Y_pred, average='macro')
    # AUC per label if possible
    aucs = []
    try:
        for j in range(Y_test.shape[1]):
            if len(np.unique(Y_test[:, j])) > 1:
                aucs.append(roc_auc_score(Y_test[:, j], Y_proba[:, j]))
    except Exception:
        pass

    metrics = {
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
        'mean_roc_auc': float(np.mean(aucs)) if aucs else None,
        'num_labels': int(Y_test.shape[1]),
    }

    # Save artifacts
    import joblib
    joblib.dump({'vectorizer': tfidf, 'mlb': mlb, 'clf': clf}, MODELS_DIR / 'risk_classifier.joblib')
    return metrics


def train_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    # Target: severity_0_10, fallback to risk_score*10
    y_train = train_df['severity_0_10'].fillna(train_df.get('risk_score', 0.0) * 10.0).astype(float)
    y_test = test_df['severity_0_10'].fillna(test_df.get('risk_score', 0.0) * 10.0).astype(float)

    X_train = engineer_features(train_df)
    X_test = engineer_features(test_df)

    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    # Spearman approx via ranking correlation
    try:
        from scipy.stats import spearmanr
        sp = float(spearmanr(y_test, y_pred).correlation)
    except Exception:
        sp = None

    metrics = {'rmse': rmse, 'spearman': sp}
    import joblib
    joblib.dump({'regressor': reg}, MODELS_DIR / 'severity_regressor.joblib')
    return metrics


def main():
    df = load_enriched(DATASET_DIR / 'train_combined_enriched.csv')
    # Time-aware split: use published_at if available, else random
    if 'published_at' in df.columns:
        df = df.sort_values('published_at')
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    clf_metrics = train_classifier(train_df, test_df)
    reg_metrics = train_regressor(train_df, test_df)

    report = {
        'NER_model': 'rule-assisted spaCy-disabled placeholder (to be replaced with transformer NER)',
        'Classifier': clf_metrics,
        'Regressor': reg_metrics,
        'splits': {'train': int(len(train_df)), 'test': int(len(test_df))},
        'notes': 'Weak supervision via heuristic labels in risk_labels; severity from pipeline + features.'
    }
    (REPORTS_DIR / 'model_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()


