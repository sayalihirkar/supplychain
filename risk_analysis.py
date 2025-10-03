import pandas as pd
import spacy
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from config import RISK_CATEGORIES, SEVERITY_LEVELS, MODELS_DIR

class RiskAnalyzer:
    def __init__(self):
        # Load the English language model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If the model is not downloaded, download it
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
        self.risk_categories = RISK_CATEGORIES
        self.severity_levels = SEVERITY_LEVELS
        
    def analyze_risk(self, text: str) -> Dict:
        """Analyze text and return risk assessment"""
        doc = self.nlp(text.lower())
        
        # Initialize risk scores
        risk_scores = {
            'supply_chain': {'score': 0, 'categories': {}},
            'strategic': {'score': 0, 'categories': {}}
        }
        
        # Check for risk categories
        for risk_domain, categories in self.risk_categories.items():
            for category, keywords in categories.items():
                # Count keyword matches
                matches = sum(1 for keyword in keywords if keyword in text.lower())
                if matches > 0:
                    risk_scores[risk_domain]['categories'][category] = matches
                    risk_scores[risk_domain]['score'] += matches
        
        # Calculate overall risk score (normalized 0-1)
        total_keywords = sum(len(keywords) for domain in self.risk_categories.values() for keywords in domain.values())
        total_matches = sum(domain['score'] for domain in risk_scores.values())
        overall_risk = min(total_matches / 10, 1.0)  # Cap at 1.0
        
        # Determine risk level
        risk_level = self._get_risk_level(overall_risk)
        
        return {
            'risk_score': overall_risk,
            'risk_level': risk_level,
            'supply_chain_risk': risk_scores['supply_chain'],
            'strategic_risk': risk_scores['strategic']
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to level"""
        if score >= 0.8:
            return 'critical'
        elif score >= 0.5:
            return 'high'
        elif score >= 0.3:
            return 'medium'
        return 'low'
    
    def analyze_news_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze a DataFrame of news articles"""
        if df.empty:
            return df
            
        # Initialize new columns
        df['risk_score'] = 0.0
        df['risk_level'] = 'low'
        df['supply_chain_risk'] = None
        df['strategic_risk'] = None
        
        # Analyze each article
        for idx, row in df.iterrows():
            text = f"{row.get('title', '')} {row.get('content', '')}"
            analysis = self.analyze_risk(text)
            
            df.at[idx, 'risk_score'] = analysis['risk_score']
            df.at[idx, 'risk_level'] = analysis['risk_level']
            df.at[idx, 'supply_chain_risk'] = str(analysis['supply_chain_risk'])
            df.at[idx, 'strategic_risk'] = str(analysis['strategic_risk'])
        
        return df

def save_risk_analysis(df: pd.DataFrame, filename: str = "risk_analysis.csv"):
    """Save risk analysis results to CSV"""
    if not df.empty:
        filepath = Path("data") / filename
        df.to_csv(filepath, index=False)
        print(f"Risk analysis saved to {filepath}")

def load_risk_analysis(filename: str = "risk_analysis.csv") -> pd.DataFrame:
    """Load risk analysis results from CSV"""
    filepath = Path("data") / filename
    try:
        return pd.read_csv(filepath, parse_dates=['published_at'])
    except FileNotFoundError:
        print(f"No existing risk analysis found at {filepath}")
        return pd.DataFrame()

if __name__ == "__main__":
    from data_ingestion import load_news_data
    
    # Load the news data
    print("Loading news data...")
    news_df = load_news_data()
    
    if news_df.empty:
        print("No news data found. Please run data_ingestion.py first.")
    else:
        # Analyze the news
        print("Analyzing risks...")
        analyzer = RiskAnalyzer()
        analyzed_df = analyzer.analyze_news_dataframe(news_df)
        
        # Save the results
        save_risk_analysis(analyzed_df)
        
        # Print summary
        if not analyzed_df.empty:
            print("\nRisk Analysis Summary:")
            print(f"Total articles analyzed: {len(analyzed_df)}")
            print(f"High/Critical risk articles: {len(analyzed_df[analyzed_df['risk_level'].isin(['high', 'critical'])]):,}")
            
            # Print high-risk articles
            high_risk = analyzed_df[analyzed_df['risk_level'].isin(['high', 'critical'])]
            if not high_risk.empty:
                print("\nHigh/Critical Risk Articles:")
                for _, row in high_risk.iterrows():
                    print(f"\n- {row['title']} (Risk: {row['risk_level'].title()}, Score: {row['risk_score']:.2f})")
                    print(f"  Source: {row['source']}")
                    print(f"  URL: {row['url']}")
