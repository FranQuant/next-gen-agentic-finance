from datetime import datetime, timezone
import os
from urllib.parse import urlparse

from agno.tools import tool
from tavily import TavilyClient
import yfinance as yf


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _is_missing(value) -> bool:
    return value is None or value != value


def _clean_text(value) -> str | None:
    if value is None:
        return None

    text = value.strip() if isinstance(value, str) else str(value).strip()
    return text or None


def _to_builtin(value):
    if _is_missing(value):
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return value


def _to_float_or_none(value) -> float | None:
    value = _to_builtin(value)
    if _is_missing(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value) -> int | None:
    value = _to_builtin(value)
    if _is_missing(value):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_date(value) -> str | None:
    value = _to_builtin(value)
    if _is_missing(value):
        return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return str(value)

    return _clean_text(value)


def _publisher_from_url(url: str | None) -> str | None:
    if not url:
        return None

    hostname = urlparse(url).netloc.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname or None


def _normalize_publisher(publisher, url: str | None = None) -> str | None:
    if isinstance(publisher, dict):
        publisher = publisher.get("displayName") or publisher.get("name")

    return _clean_text(publisher) or _publisher_from_url(url)


@tool
def get_current_stock_price(symbol: str) -> dict:
    """Return the latest available Yahoo history-based market snapshot, not a guaranteed real-time price."""
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
            "price": _to_float_or_none(latest.get("Close")),
            "open": _to_float_or_none(latest.get("Open")),
            "day_high": _to_float_or_none(latest.get("High")),
            "day_low": _to_float_or_none(latest.get("Low")),
            "volume": _to_int_or_none(latest.get("Volume")),
            "previous_close": _to_float_or_none(previous_close),
            "as_of": _normalize_date(hist.index[-1] if len(hist.index) else None),
            "history_period": "5d",
            "source": "yfinance history snapshot",
        }
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)}


@tool
def get_analyst_recommendations(symbol: str) -> dict:
    """Return recent Yahoo analyst recommendation records, not a definitive consensus engine."""
    symbol = _normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(symbol)
        recs = ticker.recommendations

        if recs is None or recs.empty:
            return {
                "symbol": symbol,
                "ok": True,
                "analyst_recommendations": [],
                "record_count": 0,
                "source": "yfinance recommendation records",
            }

        latest = recs.tail(10).reset_index().to_dict(orient="records")
        normalized = []
        for row in latest:
            normalized.append({key: _to_builtin(value) for key, value in row.items()})

        return {
            "symbol": symbol,
            "ok": True,
            "analyst_recommendations": normalized,
            "record_count": len(normalized),
            "source": "yfinance recommendation records",
        }
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)}


@tool
def get_company_info(symbol: str) -> dict:
    """Return a curated Yahoo company snapshot, not a clean audited fundamentals API."""
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

        normalized = {key: _to_builtin(value) for key, value in curated.items()}

        return {
            "symbol": symbol,
            "ok": True,
            "company_info": normalized,
            "source": "yfinance info",
        }
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)}


@tool
def get_company_news(symbol: str, num_stories: int = 10) -> dict:
    """Return normalized recent Yahoo news items for a symbol."""
    symbol = _normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []

        normalized = []
        for item in news:
            if not isinstance(item, dict):
                continue

            content = item.get("content") if isinstance(item, dict) else None

            if isinstance(content, dict):
                title = _clean_text(content.get("title"))
                url = _clean_text(
                    content.get("canonicalUrl", {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else content.get("clickThroughUrl", {}).get("url")
                    if isinstance(content.get("clickThroughUrl"), dict)
                    else content.get("url")
                )
                publisher = _normalize_publisher(
                    content.get("provider") if isinstance(content.get("provider"), dict) else content.get("publisher"),
                    url=url,
                )
                date = _normalize_date(content.get("pubDate") or content.get("displayTime"))
                snippet = _clean_text(content.get("summary"))
            else:
                title = _clean_text(item.get("title"))
                url = _clean_text(item.get("link") or item.get("url"))
                publisher = _normalize_publisher(item.get("publisher"), url=url)
                date = _normalize_date(item.get("providerPublishTime") or item.get("pubDate"))
                snippet = _clean_text(item.get("summary"))

            if not title or not url:
                continue

            normalized.append(
                {
                    "title": title,
                    "publisher": publisher,
                    "date": date,
                    "url": url,
                    "snippet": snippet,
                }
            )

            if len(normalized) >= num_stories:
                break

        return {
            "symbol": symbol,
            "ok": True,
            "news": normalized,
            "returned_count": len(normalized),
            "source": "yfinance news",
        }
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)}


@tool
def get_company_news_tavily(symbol: str, company_name: str = "", num_stories: int = 5) -> dict:
    """Return normalized recent company news search results from Tavily."""
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
        for item in results:
            if not isinstance(item, dict):
                continue

            title = _clean_text(item.get("title"))
            url = _clean_text(item.get("url"))
            if not title or not url:
                continue

            normalized.append(
                {
                    "title": title,
                    "publisher": _normalize_publisher(item.get("source"), url=url),
                    "date": _normalize_date(item.get("published_date") or item.get("publishedDate")),
                    "url": url,
                    "snippet": _clean_text(item.get("content")),
                    "score": _to_float_or_none(item.get("score")),
                }
            )

            if len(normalized) >= num_stories:
                break

        return {
            "symbol": symbol,
            "ok": True,
            "news": normalized,
            "query_used": query,
            "returned_count": len(normalized),
            "source": "Tavily news search",
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "ok": False,
            "error": str(e),
            "query_used": query,
        }
        
