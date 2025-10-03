import json
import requests
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

from config import (
    NEWS_API_KEY, NEWS_API_ENDPOINT, DATA_DIR,
    SUPPLY_CHAIN_KEYWORDS, STRATEGIC_KEYWORDS
)

class NewsFetcher:
    def __init__(self):
        self.news_api_key = NEWS_API_KEY
        self.base_url = NEWS_API_ENDPOINT
        
    def fetch_news_api(self, query: str, from_date: str = None, to_date: str = None) -> List[Dict]:
        """Fetch news from NewsAPI"""
        if not self.news_api_key or self.news_api_key == "your_newsapi_key_here":
            print("Warning: Using demo mode with limited data. Please set NEWS_API_KEY in .env for full functionality.")
            return self._get_sample_news()
            
        params = {
            'q': query,
            'apiKey': self.news_api_key,
            'pageSize': 100,  # Max allowed
            'language': 'en',
            'sortBy': 'publishedAt'
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
            
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('articles', [])
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            return []
    
    def fetch_rss_feeds(self, feed_urls: List[str]) -> List[Dict]:
        """Fetch news from RSS feeds"""
        articles = []
        for url in feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    article = {
                        'source': {'name': feed.feed.get('title', 'Unknown')},
                        'title': entry.get('title', ''),
                        'description': entry.get('description', ''),
                        'url': entry.get('link', ''),
                        'publishedAt': entry.get('published', ''),
                        'content': entry.get('content', [{}])[0].get('value', '') if 'content' in entry else ''
                    }
                    articles.append(article)
            except Exception as e:
                print(f"Error fetching RSS feed {url}: {e}")
        return articles
    
    def _get_sample_news(self) -> List[Dict]:
        """Return sample news data for demo purposes"""
        return [
            {
                'source': {'name': 'Sample News 1'},
                'title': 'Global Chip Shortage Impacts Automobile Production',
                'description': 'Major automakers including Tata Motors face production delays due to semiconductor shortage.',
                'url': 'https://example.com/chip-shortage',
                'publishedAt': datetime.utcnow().isoformat(),
                'content': 'The ongoing global semiconductor shortage is significantly impacting automobile manufacturers worldwide. Tata Motors has reported a 15% reduction in production output.'
            },
            {
                'source': {'name': 'Sample News 2'},
                'title': 'New EV Policy Boosts Electric Vehicle Adoption',
                'description': 'Government announces new subsidies for electric vehicle manufacturers.',
                'url': 'https://example.com/ev-policy',
                'publishedAt': (datetime.utcnow() - timedelta(days=1)).isoformat(),
                'content': 'The government has introduced new incentives for EV manufacturers, including Tata Motors, to accelerate the transition to electric mobility.'
            }
        ]

class NewsProcessor:
    def __init__(self):
        self.supply_chain_terms = SUPPLY_CHAIN_KEYWORDS
        self.strategic_terms = STRATEGIC_KEYWORDS
    
    def process_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process raw articles into a structured DataFrame"""
        processed = []
        
        for article in articles:
            content = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
            content_lower = content.lower()
            
            # Check for supply chain and strategic keywords
            supply_chain_matches = [term for term in self.supply_chain_terms if term in content_lower]
            strategic_matches = [term for term in self.strategic_terms if term in content_lower]
            
            # Only include articles with relevant keywords
            if supply_chain_matches or strategic_matches:
                processed.append({
                    'title': article.get('title', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'content': content,
                    'supply_chain_keywords': ", ".join(set(supply_chain_matches)),
                    'strategic_keywords': ", ".join(set(strategic_matches)),
                    'relevance_score': len(supply_chain_matches) + len(strategic_matches)
                })
        
        return pd.DataFrame(processed)

def save_news_data(df: pd.DataFrame, filename: str = "news_data.csv"):
    """Save processed news data to CSV"""
    filepath = DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"News data saved to {filepath}")

def load_news_data(filename: str = "news_data.csv") -> pd.DataFrame:
    """Load processed news data from CSV"""
    filepath = DATA_DIR / filename
    try:
        return pd.read_csv(filepath, parse_dates=['published_at'])
    except FileNotFoundError:
        print(f"No existing data found at {filepath}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    fetcher = NewsFetcher()
    processor = NewsProcessor()
    
    # Fetch news for the last 7 days
    to_date = datetime.utcnow()
    from_date = (to_date - timedelta(days=7)).strftime('%Y-%m-%d')
    
    print("Fetching news articles...")
    
    # Example queries
    queries = [
        "Tata Motors",
        "automobile supply chain",
        "EV battery production",
        "semiconductor shortage"
    ]
    
    all_articles = []
    for query in queries:
        print(f"Fetching news for: {query}")
        articles = fetcher.fetch_news_api(query, from_date=from_date)
        all_articles.extend(articles)
        time.sleep(1)  # Be nice to the API
    
    # Process and save the data
    if all_articles:
        df = processor.process_articles(all_articles)
        if not df.empty:
            save_news_data(df, "tata_motors_news.csv")
            print(f"Processed {len(df)} relevant articles")
        else:
            print("No relevant articles found matching the criteria.")
    else:
        print("No articles were fetched. Using sample data instead.")
        sample_articles = fetcher._get_sample_news()
        df = processor.process_articles(sample_articles)
        save_news_data(df, "sample_news_data.csv")
