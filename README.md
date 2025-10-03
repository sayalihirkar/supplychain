# Tata Motors Risk Intelligence Dashboard

A real-time supply chain and strategic risk monitoring system for Tata Motors that analyzes news articles to identify potential risks and their impact on operations.

## Features

- **News Ingestion**: Fetches news from various sources using NewsAPI and RSS feeds
- **Risk Analysis**: Uses NLP to analyze news content and identify risks
- **Risk Categorization**: Classifies risks into supply chain and strategic categories
- **Dashboard**: Interactive visualization of risk trends and alerts
- **Real-time Monitoring**: Continuously monitors for new risks

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- NewsAPI key (free tier available at [NewsAPI.org](https://newsapi.org/))

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd supplychain
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Create a `.env` file in the project root and add your NewsAPI key:
   ```
   NEWS_API_KEY=your_newsapi_key_here
   ```

## Usage

### 1. Run the data pipeline

To fetch and analyze news data:

```bash
python data_ingestion.py
python risk_analysis.py
```

### 2. Launch the dashboard

```bash
streamlit run dashboard.py
```

This will start a local web server and open the dashboard in your default web browser.

## Project Structure

```
supplychain/
├── data/                   # Directory for storing data files
├── models/                 # Directory for storing ML models
├── config.py              # Configuration settings and constants
├── data_ingestion.py      # News data collection and processing
├── risk_analysis.py       # Risk assessment and analysis
├── dashboard.py           # Streamlit dashboard
└── README.md              # This file
```

## Data Flow

1. **Data Collection**:
   - NewsAPI and RSS feeds are queried for relevant articles
   - Articles are processed and stored locally

2. **Risk Analysis**:
   - NLP techniques extract entities and assess risk levels
   - Articles are categorized by risk type and severity

3. **Visualization**:
   - Interactive dashboard displays risk metrics and trends
   - High-risk items are highlighted for immediate attention

## Customization

### Adding New Risk Categories

Edit the `RISK_CATEGORIES` dictionary in `config.py` to add or modify risk categories and their associated keywords.

### Adjusting Risk Scoring

Modify the `analyze_risk` method in `risk_analysis.py` to adjust how risk scores are calculated.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NewsAPI for providing news data
- Streamlit for the dashboard framework
- spaCy for natural language processing
