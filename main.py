import urllib.parse
import xml.etree.ElementTree as ET
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from analytics import fetch_equity_analytics

app = FastAPI(title="Quant Market Terminal Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POPULAR_STOCKS = [
    {"name": "Reliance Industries", "symbol": "RELIANCE.NS", "exchange": "NSE"},
    {"name": "Tata Motors", "symbol": "TATAMOTORS.NS", "exchange": "NSE"},
    {"name": "Tata Consultancy Services", "symbol": "TCS.NS", "exchange": "NSE"},
    {"name": "HDFC Bank", "symbol": "HDFCBANK.NS", "exchange": "NSE"},
    {"name": "Infosys Limited", "symbol": "INFY.NS", "exchange": "NSE"},
    {"name": "State Bank of India", "symbol": "SBIN.NS", "exchange": "NSE"},
    {"name": "ITC Limited", "symbol": "ITC.NS", "exchange": "NSE"},
    {"name": "Apple Inc.", "symbol": "AAPL", "exchange": "NASDAQ"},
    {"name": "Microsoft Corporation", "symbol": "MSFT", "exchange": "NASDAQ"},
    {"name": "NVIDIA Corporation", "symbol": "NVDA", "exchange": "NASDAQ"},
    {"name": "Alphabet Inc. (Google)", "symbol": "GOOGL", "exchange": "NASDAQ"},
    {"name": "Tesla Inc.", "symbol": "TSLA", "exchange": "NASDAQ"},
    {"name": "Amazon.com Inc.", "symbol": "AMZN", "exchange": "NASDAQ"}
]

@app.get("/")
def root():
    return {"status": "online", "message": "Quant Terminal Backend"}

@app.get("/api/search")
def search_stocks(q: str = ""):
    query = q.strip().lower()
    if not query:
        return {"results": []}
    matches = [
        s for s in POPULAR_STOCKS 
        if query in s["name"].lower() or query in s["symbol"].lower()
    ]
    return {"results": matches[:6]}

@app.get("/api/analytics/{symbol}")
def get_analytics(symbol: str):
    try:
        data = fetch_equity_analytics(symbol, period="6mo")
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/news/{query}")
def get_news(query: str):
    try:
        encoded_query = urllib.parse.quote(query.strip())
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(rss_url, headers=headers, timeout=5)
        if res.status_code != 200:
            return {"articles": []}

        root = ET.fromstring(res.content)
        articles = []
        for item in root.findall("./channel/item")[:6]:
            title = item.find("title").text if item.find("title") is not None else "Market News"
            url = item.find("link").text if item.find("link") is not None else "#"
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
            source_el = item.find("source")
            source = source_el.text if source_el is not None else "News Source"

            date_str = " ".join(pub_date.split(" ")[:4]) if pub_date else "Recent"
            articles.append({
                "title": title,
                "url": url,
                "source": source,
                "date": date_str
            })
        return {"articles": articles}
    except Exception:
        return {"articles": []}