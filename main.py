from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from analytics import compute_financial_metrics

app = FastAPI(title="Global Quant Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BENCHMARKS = {
    "INDIA": {"symbol": "^NSEI", "name": "NIFTY 50"},
    "GLOBAL": {"symbol": "^GSPC", "name": "S&P 500"}
}

# 1. Company Name Autocomplete Search
@app.get("/api/search")
def search_companies(q: str = Query(..., min_length=1)):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    params = {'q': q, 'quotesCount': 6, 'newsCount': 0}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        data = resp.json()
        matches = []
        for quote in data.get('quotes', []):
            if quote.get('quoteType') in ['EQUITY', 'ETF']:
                matches.append({
                    "symbol": quote.get('symbol'),
                    "name": quote.get('shortname') or quote.get('longname') or quote.get('symbol'),
                    "exchange": quote.get('exchange', 'GLOBAL')
                })
        return {"results": matches}
    except Exception:
        return {"results": []}

# 2. Live News Feed via Google News RSS (Headlines + Direct Redirect Links)
@app.get("/api/news/{company_query}")
def get_company_news(company_query: str):
    """
    Fetches real-time RSS news headlines. 
    Returns headline, source publisher, published time, and direct redirect link.
    """
    clean_query = urllib.parse.quote(company_query + " stock news")
    rss_url = f"https://news.google.com/rss/search?q={clean_query}&hl=en-IN&gl=IN&ceid=IN:en"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    try:
        resp = requests.get(rss_url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return {"articles": []}

        root = ET.fromstring(resp.content)
        articles = []

        # Parse top 6 RSS news items
        for item in root.findall('.//item')[:6]:
            title = item.find('title').text if item.find('title') is not None else "Market Update"
            link = item.find('link').text if item.find('link') is not None else "#"
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            source_el = item.find('source')
            source = source_el.text if source_el is not None else "News Source"

            # Format the title cleanly if it has a trailing source name
            clean_title = title.rsplit(' - ', 1)[0] if ' - ' in title else title

            # Format pub_date to shorter string (e.g. "20 Sep 2026")
            formatted_date = " ".join(pub_date.split()[:4]) if pub_date else "Recent"

            articles.append({
                "title": clean_title,
                "source": source,
                "url": link,
                "date": formatted_date
            })

        return {"articles": articles}
    except Exception:
        return {"articles": []}

# 3. Quantitative Risk & Return Analytics
@app.get("/api/analytics/{ticker}")
def get_stock_analytics(ticker: str, period: str = Query("6mo")):
    symbol = ticker.strip().upper()

    try:
        is_indian = symbol.endswith((".NS", ".BO"))
        bench_key = "INDIA" if is_indian else "GLOBAL"
        bench_info = BENCHMARKS[bench_key]

        stock_data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        bench_data = yf.download(bench_info["symbol"], period=period, auto_adjust=True, progress=False)

        if hasattr(stock_data.columns, 'levels'):
            stock_data = stock_data.xs(symbol, level=1, axis=1)
        if hasattr(bench_data.columns, 'levels'):
            bench_data = bench_data.xs(bench_info["symbol"], level=1, axis=1)

        if stock_data.empty or len(stock_data.dropna(subset=['Close'])) < 15:
            raise HTTPException(status_code=404, detail=f"Insufficient price history found for '{symbol}'.")

        results = compute_financial_metrics(stock_data, bench_data, bench_info["name"])

        return {
            "symbol": symbol,
            "benchmark": bench_info["name"],
            "metrics": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)