from datetime import datetime, timezone
import os
from urllib.parse import urlparse

from agno.tools import tool
from tavily import TavilyClient
import yfinance as yf

from news_filter import (
    NEWS_FILTER_POLICY_VERSION,
    build_company_terms,
    parse_news_datetime,
    score_tavily_news_item,
    select_diverse_news_items,
    select_preferred_news_item,
)

NEWS_QUERY_WINDOW_DAYS = 30
NEWS_QUERY_MIN_RESULTS_PER_QUERY = 2
NEWS_QUERY_MAX_RESULTS_PER_QUERY = 3


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


def _normalize_url_key(url: str | None) -> str | None:
    clean_url = _clean_text(url)
    if not clean_url:
        return None

    parsed = urlparse(clean_url)
    hostname = parsed.netloc.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    path = parsed.path.rstrip("/") or "/"
    return f"{hostname}{path}"


def _normalize_title_key(title: str | None) -> str | None:
    clean_title = _clean_text(title)
    if not clean_title:
        return None

    return " ".join(clean_title.lower().split())


@tool
def get_current_stock_price(symbol: str) -> dict:
    """Return the latest available Yahoo history-based market snapshot, not a guaranteed real-time price.

    Args:
        symbol: stock ticker symbol (e.g. AAPL)

    Returns:
        dict with keys: symbol, ok, price, open, day_high, day_low, volume,
        previous_close, as_of, history_period, source. On failure: symbol, ok, error.
    """
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
    """Return recent Yahoo analyst recommendation records, not a definitive consensus engine.

    Args:
        symbol: stock ticker symbol (e.g. AAPL)

    Returns:
        dict with keys: symbol, ok, analyst_recommendations (list of period-level count
        records), record_count, source. On failure: symbol, ok, error.
    """
    symbol = _normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(symbol)
        # Yahoo's recommendations surface is record-oriented and may be sparse,
        # delayed, or empty depending on the symbol.
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
    """Return a curated Yahoo company snapshot, not a clean audited fundamentals API.

    Args:
        symbol: stock ticker symbol (e.g. AAPL)

    Returns:
        dict with keys: symbol, ok, company_info (curated subset of yfinance .info
        fields covering identity, fundamentals, valuation, and targets). Yahoo's
        .info surface is best-effort and may disagree across symbols or sessions. On failure:
        symbol, ok, error.
    """
    symbol = _normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(symbol)
        # Yahoo's .info surface is a curated snapshot, not a schema-stable feed.
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
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
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
    company_name = _clean_text(company_name) or ""
    max_results_per_query = max(
        NEWS_QUERY_MIN_RESULTS_PER_QUERY,
        min(num_stories, NEWS_QUERY_MAX_RESULTS_PER_QUERY),
    )
    packet_metadata = {
        "policy_version": NEWS_FILTER_POLICY_VERSION,
        "query_window_days": NEWS_QUERY_WINDOW_DAYS,
        "max_results_per_query": max_results_per_query,
    }

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "symbol": symbol,
            "ok": False,
            "error": "TAVILY_API_KEY not set.",
            **packet_metadata,
        }

    company_label = f"{company_name} ({symbol})" if company_name else symbol
    queries = [
        {
            "query_category": "broad_company_news",
            "query": f"{company_label} latest company news",
        },
        {
            "query_category": "product_strategy",
            "query": f"{company_label} acquisition product launch partnership strategy",
        },
        {
            "query_category": "regulatory_legal",
            "query": f"{company_label} regulatory legal antitrust app store china",
        },
        {
            "query_category": "management_commentary",
            "query": f"{company_label} management commentary CEO interview guidance",
        },
    ]

    try:
        client = TavilyClient(api_key=api_key)
        company_terms, company_phrase = build_company_terms(symbol, company_name)

        collected = []
        excluded_count = 0
        deduped_count = 0
        query_failures = []

        for query_info in queries:
            try:
                response = client.search(
                    query=query_info["query"],
                    topic="news",
                    days=NEWS_QUERY_WINDOW_DAYS,
                    max_results=max_results_per_query,
                    include_raw_content=False,
                )
            except Exception as e:
                query_failures.append(
                    {
                        "query_category": query_info["query_category"],
                        "error": str(e),
                    }
                )
                continue

            results = response.get("results", []) if isinstance(response, dict) else []
            for item in results:
                if not isinstance(item, dict):
                    continue

                title = _clean_text(item.get("title"))
                url = _clean_text(item.get("url"))
                if not title or not url:
                    continue

                normalized = {
                    "title": title,
                    "publisher": _normalize_publisher(item.get("source"), url=url),
                    "date": _normalize_date(item.get("published_date") or item.get("publishedDate")),
                    "url": url,
                    "snippet": _clean_text(item.get("content")),
                    "score": _to_float_or_none(item.get("score")),
                    "query_category": query_info["query_category"],
                }

                scoring = score_tavily_news_item(
                    normalized,
                    symbol=symbol,
                    company_terms=company_terms,
                    company_phrase=company_phrase,
                )

                if scoring["exclusion_reason"] is not None:
                    excluded_count += 1
                    continue

                normalized["relevance_bucket"] = scoring["bucket"]
                normalized["_ranking_score"] = scoring["score"]
                if isinstance(scoring.get("reason_summary"), dict):
                    normalized["reason_summary"] = scoring["reason_summary"]
                collected.append(normalized)

        if not collected and len(query_failures) == len(queries):
            return {
                "symbol": symbol,
                "ok": False,
                "error": "All Tavily company-news queries failed.",
                "queries_used": queries,
                "query_used": queries[0]["query"],
                "query_failures": query_failures,
                "source": "Tavily news search (multi-query)",
                **packet_metadata,
            }

        deduped_items = {}
        title_index = {}
        for item in collected:
            url_key = _normalize_url_key(item.get("url"))
            title_key = _normalize_title_key(item.get("title"))

            existing_key = None
            if url_key and url_key in deduped_items:
                existing_key = url_key
            elif title_key and title_key in title_index:
                existing_key = title_index[title_key]

            if existing_key is not None:
                deduped_count += 1
                deduped_items[existing_key] = select_preferred_news_item(
                    deduped_items[existing_key],
                    item,
                )
                continue

            item_key = url_key or title_key or f"item-{len(deduped_items) + 1}"
            item["_item_key"] = item_key
            deduped_items[item_key] = item
            if title_key:
                title_index[title_key] = item_key

        ranked_items = sorted(
            deduped_items.values(),
            key=lambda item: (
                item.get("relevance_bucket") == "high_confidence_company_specific",
                item.get("relevance_bucket") == "broader_context",
                item.get("_ranking_score", 0.0),
                parse_news_datetime(item.get("date")) or datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )

        selected = select_diverse_news_items(ranked_items, num_stories, company_terms)

        high_confidence_count = sum(
            1 for item in selected if item.get("relevance_bucket") == "high_confidence_company_specific"
        )
        broader_context_count = sum(
            1 for item in selected if item.get("relevance_bucket") == "broader_context"
        )
        distinct_categories = len({item.get("query_category") for item in selected})

        if high_confidence_count >= 2:
            news_quality_note = "Strong company-specific coverage found."
        elif high_confidence_count >= 1 or broader_context_count >= 2:
            news_quality_note = "Mixed result set: company-specific items found, with some contextual coverage retained."
        elif selected:
            news_quality_note = "Weak result set: best available items retained, but company specificity is limited."
        else:
            news_quality_note = "No sufficiently relevant company-news items found."

        if query_failures and selected:
            news_quality_note = f"{news_quality_note} Partial query failures occurred."

        normalized = []
        for item in selected:
            entry = {
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "date": item.get("date"),
                "url": item.get("url"),
                "snippet": item.get("snippet"),
                "score": item.get("score"),
                "query_category": item.get("query_category"),
                "relevance_bucket": item.get("relevance_bucket"),
            }
            reason_summary = item.get("reason_summary")
            if isinstance(reason_summary, dict):
                entry["reason_summary"] = reason_summary
            normalized.append(entry)

        return {
            "symbol": symbol,
            "ok": True,
            "news": normalized,
            "queries_used": queries,
            "query_used": queries[0]["query"],
            "returned_count": len(normalized),
            "excluded_count": excluded_count,
            "deduped_count": deduped_count,
            "news_quality_note": news_quality_note,
            "event_diversity_note": f"Selected {len(normalized)} items across {distinct_categories} query categories.",
            "source": "Tavily news search (multi-query)",
            **packet_metadata,
            **({"query_failures": query_failures} if query_failures else {}),
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "ok": False,
            "error": str(e),
            "queries_used": queries,
            "query_used": queries[0]["query"],
            **packet_metadata,
        }
