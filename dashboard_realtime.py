import json
from pathlib import Path
import ast
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import plotly.graph_objects as go

DATASET_DIR = Path("Dataset")
KB_DIR = Path("supply_kb")

st.set_page_config(page_title="Risk Intelligence - Realtime", page_icon="📊", layout="wide")


def load_nodes():
    nodes = pd.read_csv(KB_DIR / "nodes.csv")
    # attach severity from latest events
    sev = {}
    try:
        ev = pd.read_csv(DATASET_DIR / "test_impact_events.csv")
        ev["impacted_nodes_list"] = ev["impacted_nodes"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
        for _, r in ev.iterrows():
            for nid in r["impacted_nodes_list"]:
                sev[nid] = max(sev.get(nid, 0.0), float(r.get("severity", 0.0)))
    except Exception:
        pass
    nodes["severity"] = nodes["node_id"].map(lambda x: sev.get(x, 0.0))
    return nodes


def overview_heatmap(nodes: pd.DataFrame):
    st.subheader("Overview — Node Severity Heatmap")
    # Simple scatter as world map proxy using location text; in real app use geocoding
    fig = px.scatter(
        nodes,
        x="location",
        y="node_type",
        size="severity",
        color="severity",
        color_continuous_scale="Reds",
        hover_data=["node_id", "materials_handled", "tier", "criticality_score"],
        title="Supply Nodes by Severity"
    )
    st.plotly_chart(fig, use_container_width=True)


def live_events_feed(limit: int = 20):
    st.subheader("Live Events")
    try:
        ev = pd.read_csv(DATASET_DIR / "test_impact_events.csv").sort_values("severity", ascending=False).head(limit)
    except Exception:
        st.info("No events.")
        return
    # Render each item within its own form to stabilize Streamlit widget state
    for idx, r in ev.reset_index(drop=True).iterrows():
        e_id = str(r.get('event_id',''))
        with st.form(f"evt_form_{e_id}"):
            cols = st.columns([5, 2, 2, 3])
            cols[0].markdown(f"**{r.get('headline','')}**")
            cols[1].markdown(f"Severity: `{r.get('severity',0):.1f}`")
            cols[2].markdown(f"Confidence: `{r.get('confidence',0):.2f}`")
            explain_clicked = cols[3].form_submit_button("Explain")
            a1, a2, a3 = st.columns([2, 3, 7])
            with a1:
                ack_clicked = st.form_submit_button("Acknowledge")
            with a2:
                assignee = st.text_input("Assign to", value="analyst@tata.com", key=f"assign_to_{e_id}")
                assign_clicked = st.form_submit_button("Assign")
            if explain_clicked:
                from rag_retriever import explain
                st.markdown(explain(str(r.get('headline','')), top_k=5))
            if ack_clicked:
                try:
                    resp = requests.post("http://localhost:8000/events/" + e_id + "/ack", json={"user": "analyst", "comment": "ack via UI"})
                    st.success("Acknowledged" if resp.status_code == 200 else f"Ack failed: {resp.status_code}")
                except Exception as ex:
                    st.error(f"Ack error: {ex}")
            if assign_clicked:
                try:
                    resp = requests.post("http://localhost:8000/events/" + e_id + "/assign", json={"assignee": assignee, "comment": "from UI"})
                    st.success("Assigned" if resp.status_code == 200 else f"Assign failed: {resp.status_code}")
                except Exception as ex:
                    st.error(f"Assign error: {ex}")


def risk_matrix():
    st.subheader("Risk Matrix — Probability vs Impact")
    try:
        ev = pd.read_csv(DATASET_DIR / "test_impact_events.csv")
    except Exception:
        st.info("No events.")
        return
    df = pd.DataFrame({
        'impact': ev['severity'],
        'probability': ev['confidence'],
        'title': ev['headline']
    })
    fig = px.scatter(df, x='probability', y='impact', hover_name='title', color='impact', color_continuous_scale='RdYlGn_r')
    fig.update_layout(xaxis_title='Probability', yaxis_title='Impact (Severity)')
    st.plotly_chart(fig, use_container_width=True)


def severity_scorecard(nodes: pd.DataFrame):
    st.subheader("Severity Scorecard")
    bins = pd.cut(nodes['severity'].fillna(0.0), bins=[-0.01, 3, 5, 8, 10], labels=['Low', 'Medium', 'High', 'Critical'])
    counts = bins.value_counts().reindex(['Critical', 'High', 'Medium', 'Low'], fill_value=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Critical Nodes", int(counts['Critical']))
    c2.metric("High Nodes", int(counts['High']))
    c3.metric("Medium Nodes", int(counts['Medium']))
    c4.metric("Low Nodes", int(counts['Low']))


def use_case_demo():
    st.subheader("Use-Case Demonstration")
    st.caption("Examples: chip shortages, lithium price spikes, EV subsidy changes")
    try:
        ev = pd.read_csv(DATASET_DIR / "test_impact_events.csv")
    except Exception:
        st.info("No events available.")
        return
    keywords = {
        'Chip Shortage': 'chip',
        'Lithium Spike': 'lithium',
        'EV Subsidy Change': 'subsidy'
    }
    for label, kw in keywords.items():
        sub = ev[ev['headline'].str.lower().str.contains(kw, na=False)].head(3)
        if sub.empty:
            continue
        with st.expander(f"{label}"):
            for _, r in sub.iterrows():
                st.markdown(f"- {r['headline']} | Severity `{r['severity']:.1f}` | Nodes: {r['impacted_nodes']}")
            from rag_retriever import explain
            st.markdown(explain(label + " news Tata Motors", top_k=5))


def drilldown_panel():
    st.subheader("Drilldown — Event + RAG")
    try:
        ev = pd.read_csv(DATASET_DIR / "test_impact_events.csv")
    except Exception:
        st.info("No events.")
        return
    options = ev['event_id'].tolist()
    if not options:
        st.info("No events.")
        return
    sel = st.selectbox("Select event", options)
    row = ev[ev['event_id'] == sel].iloc[0]
    st.markdown(f"**{row['headline']}**")
    st.markdown(f"Severity: `{row['severity']:.1f}` | Confidence: `{row['confidence']:.2f}`")
    st.markdown("Impacted nodes:")
    nodes = json.loads(row['impacted_nodes']) if isinstance(row['impacted_nodes'], str) else []
    st.write(nodes)
    from rag_retriever import explain
    st.markdown("Explanation (RAG):")
    st.code(explain(str(row['headline']), top_k=5))


def main():
    st.title("Risk Intelligence — Real-time Dashboard")
    st.markdown("Enter a headline and text to predict risk (model inference):")
    with st.form("predict_form"):
        title = st.text_input("Title", value="")
        text = st.text_area("Text", value="")
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            resp = requests.post("http://localhost:8000/predict", json={
                "id": "adhoc-1", "source": "ui", "published_at": "", "title": title, "text": text, "url": ""
            })
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Risk level: {data.get('risk_level','')} | Severity: {data.get('severity_0_10',0):.2f}")
                st.write({"risk_labels": data.get('risk_labels', []), "label_probabilities": data.get('label_probabilities', {})})
                st.write({"events": data.get('events', []), "impacted_nodes": data.get('impacted_nodes', [])})
            else:
                st.error(f"Prediction failed: {resp.status_code}")
        except Exception as ex:
            st.error(f"Prediction error: {ex}")
    nodes = load_nodes()
    severity_scorecard(nodes)
    overview_heatmap(nodes)
    col1, col2 = st.columns([2, 1])
    with col1:
        live_events_feed()
        risk_matrix()
    with col2:
        drilldown_panel()
    use_case_demo()

    st.markdown("---")
    st.subheader("Live News — Search, Review, Predict, Create Event")
    with st.form("news_search"):
        q = st.text_input("Query", value="Tata Motors supply chain")
        submitted = st.form_submit_button("Search News")
    if submitted:
        try:
            r = requests.get("http://localhost:8000/news/search", params={"q": q})
            items = r.json() if r.status_code == 200 else []
        except Exception:
            items = []
        if not items:
            st.info("No news found.")
        else:
            for it in items[:10]:
                with st.form(f"news_item_{hash(it['id'])}"):
                    st.markdown(f"**{it['title']}**  ")
                    st.caption(f"{it['source']} | {it['published_at']}")
                    st.write((it.get('text','') or '')[:300] + '...')
                    do_predict = st.form_submit_button("Predict Risk")
                    make_event = st.form_submit_button("Create Event")
                    if do_predict:
                        pr = requests.post("http://localhost:8000/predict", json={
                            "id": it['id'], "source": it['source'], "published_at": it['published_at'],
                            "title": it['title'], "text": it.get('text',''), "url": it['url']
                        })
                        if pr.status_code == 200:
                            data = pr.json()
                            st.success(f"Severity {data.get('severity_0_10',0):.2f} | {data.get('risk_level','')}")
                            st.write({"labels": data.get('risk_labels', []), "probs": data.get('label_probabilities', {})})
                    if make_event:
                        pr = requests.post("http://localhost:8000/predict", json={
                            "id": it['id'], "source": it['source'], "published_at": it['published_at'],
                            "title": it['title'], "text": it.get('text',''), "url": it['url']
                        })
                        sev = 0.0
                        labels = []
                        if pr.status_code == 200:
                            data = pr.json()
                            sev = float(data.get('severity_0_10', 0.0))
                            labels = data.get('risk_labels', [])
                        ce = requests.post("http://localhost:8000/events/create", json={
                            "event_id": f"live-{hash(it['id'])}",
                            "title": it['title'],
                            "severity": sev,
                            "risk_labels": labels,
                            "impacted_nodes": [],
                            "evidence": [it['id']],
                            "confidence": 0.7
                        })
                        if ce.status_code == 200:
                            st.success("Event created")


if __name__ == '__main__':
    main()


