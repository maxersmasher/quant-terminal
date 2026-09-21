import xml.etree.ElementTree as ET
import urllib.parse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from analytics import fetch_equity_analytics

app = FastAPI(
    title="Quantitative Market Analytics API",
    description="Engine computing volatility, dynamic Beta, Sharpe, and Sortino risk ratios.",
    version="2.0.0"
)

# Enable CORS for Vercel and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online", "service": "Quant Financial Engine"}

@app.get("/api/analytics/{ticker}")
def get_analytics(ticker: str, period: str = "1y"):
    try:
        data = fetch_equity_analytics(ticker=ticker, period=period)
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/news/{query}")
def get_syndicated_news(query: str):
    """
    Fetches real-time syndication feeds from Google News RSS.
    Does not scrape or store text; forwards primary attribution URLs.
    """
    try:
        encoded_query = urllib.parse.quote(query.strip())
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        res = requests.get(rss_url, headers=headers, timeout=6)
        if res.status_code != 200:
            return {"items": []}

        root = ET.fromstring(res.content)
        articles = []
        for item in root.findall("./channel/item")[:6]:
            title = item.find("title").text if item.find("title") is not None else "Market Update"
            link = item.find("link").text if item.find("link") is not None else "#"
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
            source_el = item.find("source")
            source = source_el.text if source_el is not None else "Wire Source"

            articles.append({
                "title": title,
                "link": link,
                "pubDate": pub_date,
                "source": source
            })
        return {"items": articles}
    except Exception:
        return {"items": []}