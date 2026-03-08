# examples/finance_tools.py

from agno.tools import tool
import yfinance as yf
from tavily import TavilyClient
import os


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


@tool
def get_current_stock_price(symbol: str) -> dict:
    """Return the latest available stock price snapshot."""
    symbol = _normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")

        if hist is None or hist.empty:
            return {"symbol": symbol, "ok": False, "error": "No price history available."}

        latest = hist.iloc[-1]
        previous_close = hist["Close"].iloc[-2] if len(hist) >= 2 else None

        return {
            "symbol": symbol,
            "ok": True,
            "price": float(latest["Close"]),
            "open": float(latest["Open"]) if "Open" in latest else None,
            "day_high": float(latest["High"]) if "High" in latest else None,
            "day_low": float(latest["Low"]) if "Low" in latest else None,
            "volume": int(latest["Volume"]) if "Volume" in latest else None,
            "previous_close": float(previous_close) if previous_close is not None else None,
        }
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)}


@tool
def get_analyst_recommendations(symbol: str) -> dict:
    """Return recent analyst recommendation records."""
    symbol = _normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(symbol)
        recs = ticker.recommendations

        if recs is None or recs.empty:
            return {"symbol": symbol, "ok": True, "analyst_recommendations": []}

        latest = recs.tail(10).reset_index().to_dict(orient="records")
        return {
            "symbol": symbol,
            "ok": True,
            "record_count": len(latest),
            "analyst_recommendations": latest,
        }
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)}


@tool
def get_company_info(symbol: str) -> dict:
    """Return a curated subset of company fundamentals and descriptive fields."""
    symbol = _normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        curated = {
            "longName": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "longBusinessSummary": info.get("longBusinessSummary"),
            "marketCap": info.get("marketCap"),
            "enterpriseValue": info.get("enterpriseValue"),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "totalRevenue": info.get("totalRevenue"),
            "netIncomeToCommon": info.get("netIncomeToCommon"),
            "trailingEps": info.get("trailingEps"),
            "forwardPE": info.get("forwardPE"),
            "trailingPE": info.get("trailingPE"),
            "priceToBook": info.get("priceToBook"),
            "currentRatio": info.get("currentRatio"),
            "quickRatio": info.get("quickRatio"),
            "totalCash": info.get("totalCash"),
            "totalDebt": info.get("totalDebt"),
            "operatingCashflow": info.get("operatingCashflow"),
            "freeCashflow": info.get("freeCashflow"),
            "grossMargins": info.get("grossMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "profitMargins": info.get("profitMargins"),
            "revenueGrowth": info.get("revenueGrowth"),
            "earningsGrowth": info.get("earningsGrowth"),
            "targetHighPrice": info.get("targetHighPrice"),
            "targetLowPrice": info.get("targetLowPrice"),
            "targetMeanPrice": info.get("targetMeanPrice"),
            "targetMedianPrice": info.get("targetMedianPrice"),
            "recommendationMean": info.get("recommendationMean"),
            "recommendationKey": info.get("recommendationKey"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
            "dividendRate": info.get("dividendRate"),
            "dividendYield": info.get("dividendYield"),
            "payoutRatio": info.get("payoutRatio"),
        }

        return {"symbol": symbol, "ok": True, "company_info": curated}
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)}


@tool
def get_company_news(symbol: str, num_stories: int = 10) -> dict:
    """Return normalized recent news headlines from yfinance."""
    symbol = _normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []

        normalized = []
        for item in news[:num_stories]:
            content = item.get("content") if isinstance(item, dict) else None

            if isinstance(content, dict):
                normalized.append({
                    "title": content.get("title"),
                    "publisher": content.get("provider", {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else content.get("publisher"),
                    "date": content.get("pubDate") or content.get("displayTime"),
                    "url": content.get("canonicalUrl", {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else content.get("clickThroughUrl", {}).get("url")
                    if isinstance(content.get("clickThroughUrl"), dict)
                    else content.get("url"),
                    "summary": content.get("summary"),
                })
            else:
                normalized.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "date": item.get("providerPublishTime") or item.get("pubDate"),
                    "url": item.get("link") or item.get("url"),
                    "summary": item.get("summary"),
                })

        return {"symbol": symbol, "ok": True, "news": normalized}
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)}


@tool
def get_company_news_tavily(symbol: str, company_name: str = "", num_stories: int = 5) -> dict:
    """Return normalized recent company news using Tavily."""
    symbol = _normalize_symbol(symbol)

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "symbol": symbol,
            "ok": False,
            "error": "TAVILY_API_KEY not set.",
        }

    query = f"{symbol} stock latest company news"
    if company_name and company_name.strip():
        query = f"{company_name.strip()} ({symbol}) latest company news"

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            topic="news",
            max_results=num_stories,
            include_raw_content=False,
        )

        results = response.get("results", []) if isinstance(response, dict) else []

        normalized = []
        for item in results[:num_stories]:
            url = item.get("url")
            publisher = None
            if isinstance(url, str) and "://" in url:
                publisher = url.split("/")[2]

            normalized.append({
                "title": item.get("title"),
                "publisher": publisher,
                "date": item.get("published_date") or item.get("publishedDate"),
                "url": url,
                "summary": item.get("content"),
                "score": item.get("score"),
            })

        return {
            "symbol": symbol,
            "ok": True,
            "query_used": query,
            "news": normalized,
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "ok": False,
            "error": str(e),
            "query_used": query,
        }
        