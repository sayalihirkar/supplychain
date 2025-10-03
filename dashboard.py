import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ast
import json
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Tata Motors Risk Intelligence Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1f4e79; margin-bottom: 0.5rem;}
    .risk-critical {color: #ff4b4b; font-weight: bold;}
    .risk-high {color: #ff9a3c; font-weight: bold;}
    .risk-medium {color: #ffc107; font-weight: bold;}
    .risk-low {color: #28a745; font-weight: bold;}
    .card {padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

def load_data() -> pd.DataFrame:
    """Load the risk analysis data"""
    try:
        df = pd.read_csv('data/risk_analysis.csv', parse_dates=['published_at'])
        
        # Convert string representations of dictionaries back to dictionaries
        if 'supply_chain_risk' in df.columns:
            df['supply_chain_risk'] = df['supply_chain_risk'].apply(
                lambda x: eval(x) if isinstance(x, str) else {}
            )
        if 'strategic_risk' in df.columns:
            df['strategic_risk'] = df['strategic_risk'].apply(
                lambda x: eval(x) if isinstance(x, str) else {}
            )
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def get_risk_color(level: str) -> str:
    """Get color based on risk level"""
    colors = {
        'critical': '#ff4b4b',
        'high': '#ff9a3c',
        'medium': '#ffc107',
        'low': '#28a745'
    }
    return colors.get(level.lower(), '#6c757d')

def display_risk_summary(df: pd.DataFrame):
    """Display risk summary metrics"""
    if df.empty:
        return
        
    # Calculate metrics
    total_articles = len(df)
    high_risk = len(df[df['risk_level'].isin(['high', 'critical'])])
    risk_percentage = (high_risk / total_articles * 100) if total_articles > 0 else 0
    
    # Risk distribution
    risk_dist = df['risk_level'].value_counts().reindex(
        ['critical', 'high', 'medium', 'low'], fill_value=0
    )
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Articles Analyzed", f"{total_articles:,}")
    
    with col2:
        st.metric("High/Critical Risk Articles", f"{high_risk:,}")
    
    with col3:
        st.metric("Risk Exposure", f"{risk_percentage:.1f}%")
    
    # Risk distribution chart
    st.subheader("Risk Distribution")
    fig = px.pie(
        names=risk_dist.index.str.title(),
        values=risk_dist.values,
        color=risk_dist.index,
        color_discrete_map={
            'Critical': '#ff4b4b',
            'High': '#ff9a3c',
            'Medium': '#ffc107',
            'Low': '#28a745'
        },
        hole=0.6
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

def display_risk_trends(df: pd.DataFrame):
    """Display risk trends over time"""
    if df.empty:
        return
        
    # Resample by week
    df_weekly = df.set_index('published_at').resample('W').agg({
        'risk_score': 'mean',
        'title': 'count'
    }).reset_index()
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add risk score line
    fig.add_trace(go.Scatter(
        x=df_weekly['published_at'],
        y=df_weekly['risk_score'],
        name='Average Risk Score',
        line=dict(color='#1f4e79', width=2),
        yaxis='y1'
    ))
    
    # Add article count bars
    fig.add_trace(go.Bar(
        x=df_weekly['published_at'],
        y=df_weekly['title'],
        name='Articles',
        marker_color='#6c757d',
        opacity=0.3,
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Risk Trends Over Time',
        xaxis_title='Date',
        yaxis=dict(
            title='Average Risk Score',
            range=[0, 1.1],
            showgrid=True,
            gridcolor='#f0f0f0'
        ),
        yaxis2=dict(
            title='Number of Articles',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(orientation='h', y=1.1, yanchor='bottom'),
        plot_bgcolor='white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_risk_categories(df: pd.DataFrame):
    """Display risk categories breakdown"""
    if df.empty:
        return
    
    # Extract category scores
    supply_chain_scores = {}
    strategic_scores = {}
    
    for _, row in df.iterrows():
        # Process supply chain risk categories
        if isinstance(row.get('supply_chain_risk'), dict) and 'categories' in row['supply_chain_risk']:
            for cat, score in row['supply_chain_risk']['categories'].items():
                supply_chain_scores[cat] = supply_chain_scores.get(cat, 0) + score
        
        # Process strategic risk categories
        if isinstance(row.get('strategic_risk'), dict) and 'categories' in row['strategic_risk']:
            for cat, score in row['strategic_risk']['categories'].items():
                strategic_scores[cat] = strategic_scores.get(cat, 0) + score
    
    # Create DataFrames for plotting
    sc_df = pd.DataFrame({
        'Category': list(supply_chain_scores.keys()),
        'Score': list(supply_chain_scores.values()),
        'Type': 'Supply Chain'
    })
    
    st_df = pd.DataFrame({
        'Category': list(strategic_scores.keys()),
        'Score': list(strategic_scores.values()),
        'Type': 'Strategic'
    })
    
    # Combine and sort
    combined_df = pd.concat([sc_df, st_df])
    combined_df = combined_df.sort_values('Score', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        combined_df,
        x='Score',
        y='Category',
        color='Type',
        orientation='h',
        title='Risk Categories by Impact',
        color_discrete_map={
            'Supply Chain': '#1f4e79',
            'Strategic': '#6c757d'
        }
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_high_risk_articles(df: pd.DataFrame):
    """Display high-risk articles in an expandable section"""
    if df.empty:
        return
        
    high_risk = df[df['risk_level'].isin(['high', 'critical'])].sort_values(
        'risk_score', ascending=False
    )
    
    if not high_risk.empty:
        st.subheader("High/Critical Risk Articles")
        
        for _, row in high_risk.iterrows():
            risk_color = get_risk_color(row['risk_level'])
            
            with st.expander(f"{row['title']} - {row['source']} (Risk: {row['risk_level'].title()})"):
                st.markdown(f"**Published:** {row['published_at'].strftime('%Y-%m-%d')}")
                st.markdown(f"**Risk Score:** {row['risk_score']:.2f}")
                
                # Display content preview
                content = row.get('content', '')
                if len(content) > 300:
                    content = content[:300] + "..."
                st.markdown(f"**Preview:** {content}")
                
                # Display keywords
                sc_keywords = row.get('supply_chain_keywords', '')
                st_keywords = row.get('strategic_keywords', '')
                
                if sc_keywords or st_keywords:
                    col1, col2 = st.columns(2)
                    if sc_keywords:
                        with col1:
                            st.markdown("**Supply Chain Keywords:**")
                            st.write(", ".join([f"`{k.strip()}`" for k in sc_keywords.split(",") if k.strip()]))
                    if st_keywords:
                        with col2:
                            st.markdown("**Strategic Keywords:**")
                            st.write(", ".join([f"`{k.strip()}`" for k in st_keywords.split(",") if k.strip()]))
                
                st.markdown(f"[Read more]({row['url']})")

def main():
    # Title and description
    st.markdown("<h1 class='main-header'>🚗 Tata Motors Risk Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
        Monitor and analyze supply chain and strategic risks from news sources.
        This dashboard provides real-time insights into potential risks affecting Tata Motors.
    """)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("No risk analysis data found. Please run the data pipeline first.")
        if st.button("Run Data Pipeline"):
            with st.spinner("Running data pipeline..."):
                # This would call your data pipeline
                st.error("Data pipeline execution not implemented in this demo.")
        return
    
    # Display last updated time
    last_updated = df['published_at'].max()
    st.caption(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M')} | {len(df)} articles analyzed")
    
    # Main metrics and risk distribution
    display_risk_summary(df)
    
    # Two-column layout for trends and categories
    col1, col2 = st.columns(2)
    
    with col1:
        display_risk_trends(df)
    
    with col2:
        display_risk_categories(df)
    
    # High-risk articles
    display_high_risk_articles(df)
    
    # Data table (collapsed by default)
    with st.expander("View Raw Data"):
        st.dataframe(df.drop(columns=['supply_chain_risk', 'strategic_risk']), use_container_width=True)

if __name__ == "__main__":
    main()
