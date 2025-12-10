# examples/finance_tools.py

import yfinance as yf
from agno.tools import tool


@tool
def get_current_stock_price(symbol: str) -> dict:
    """Return real-time stock price."""
    ticker = yf.Ticker(symbol)
    price = ticker.history(period="1d")["Close"].iloc[-1]
    return {"symbol": symbol, "price": float(price)}


@tool
def get_analyst_recommendations(symbol: str) -> dict:
    """Return analyst recommendation summary."""
    ticker = yf.Ticker(symbol)
    recs = ticker.recommendations

    if recs is None or recs.empty:
        return {"symbol": symbol, "analyst_recommendations": None}

    latest = recs.tail(10).to_dict(orient="records")
    return {"symbol": symbol, "analyst_recommendations": latest}


@tool
def get_company_info(symbol: str) -> dict:
    """Return company fundamentals & key metrics."""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {"symbol": symbol, "company_info": info}


@tool
def get_company_news(symbol: str, num_stories: int = 10) -> dict:
    """Return recent news headlines."""
    ticker = yf.Ticker(symbol)
    news = ticker.news[:num_stories]
    return {"symbol": symbol, "news": news}
