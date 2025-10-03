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


def live_events_feed():
    st.subheader("Live Events")
    try:
        ev = pd.read_csv(DATASET_DIR / "test_impact_events.csv").sort_values("severity", ascending=False)
    except Exception:
        st.info("No events.")
        return
    for _, r in ev.iterrows():
        cols = st.columns([5, 2, 2, 3])
        cols[0].markdown(f"**{r.get('headline','')}**")
        cols[1].markdown(f"Severity: `{r.get('severity',0):.1f}`")
        cols[2].markdown(f"Confidence: `{r.get('confidence',0):.2f}`")
        e_id = str(r.get('event_id',''))
        btn_explain = cols[3].button("Explain", key=f"exp_{e_id}")
        if btn_explain:
            from rag_retriever import explain
            st.code(explain(str(r.get('headline','')), top_k=5))

        # Action row
        a1, a2, a3 = st.columns([2, 2, 6])
        with a1:
            if st.button("Acknowledge", key=f"ack_{e_id}"):
                try:
                    resp = requests.post("http://localhost:8000/events/" + e_id + "/ack", json={"user": "analyst", "comment": "ack via UI"})
                    st.success("Acknowledged" if resp.status_code == 200 else f"Ack failed: {resp.status_code}")
                except Exception as ex:
                    st.error(f"Ack error: {ex}")
        with a2:
            assignee = st.text_input("Assign to", key=f"assignee_{e_id}", value="analyst@tata.com")
            if st.button("Assign", key=f"assign_{e_id}"):
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
    nodes = load_nodes()
    overview_heatmap(nodes)
    col1, col2 = st.columns([2, 1])
    with col1:
        live_events_feed()
        risk_matrix()
    with col2:
        drilldown_panel()


if __name__ == '__main__':
    main()


