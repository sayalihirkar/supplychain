import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Directories
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# News API Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_newsapi_key_here")
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# Keywords for Tata Motors supply chain and strategic risks
SUPPLY_CHAIN_KEYWORDS = [
    "supply chain", "raw material", "lithium", "cobalt", "semiconductor",
    "chip shortage", "logistics", "shipping", "port", "freight", "vendor",
    "supplier", "manufacturing", "production", "inventory", "shortage"
]

STRATEGIC_KEYWORDS = [
    "EV policy", "electric vehicle", "regulation", "sustainability",
    "emission", "tariff", "trade war", "geopolitical", "competition",
    "market demand", "sales", "government policy", "subsidy", "incentive"
]

# Risk Categories
RISK_CATEGORIES = {
    "supply_chain": {
        "materials": ["lithium", "cobalt", "steel", "aluminum", "rubber"],
        "logistics": ["port", "shipping", "freight", "transport"],
        "production": ["manufacturing", "factory", "plant", "assembly"],
        "labor": ["strike", "labor", "union", "shortage"]
    },
    "strategic": {
        "regulatory": ["regulation", "policy", "law", "standard"],
        "market": ["demand", "sales", "competition", "market share"],
        "environmental": ["sustainability", "emission", "carbon", "climate"],
        "technology": ["EV", "battery", "autonomous", "innovation"]
    }
}

# Severity Scoring
SEVERITY_LEVELS = {
    "low": {"score": 1, "description": "Minimal impact"},
    "medium": {"score": 2, "description": "Moderate impact"},
    "high": {"score": 3, "description": "Significant impact"},
    "critical": {"score": 4, "description": "Severe impact"}
}
