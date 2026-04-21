from collections.abc import Sequence
import io
from datetime import datetime, timezone
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Any
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
DEFAULT_TAVILY_EXCLUDE_DOMAINS = (
    "247wallst.com",
    "americanbankingnews.com",
    "barchart.com",
    "defenseworld.net",
    "etfdailynews.com",
    "fool.com",
    "investorplace.com",
    "marketbeat.com",
    "markets.financialcontent.com",
    "mexc.com",
    "simplywall.st",
    "stockanalysis.com",
    "stocktitan.net",
    "tipranks.com",
)
ALLOWED_TAVILY_SEARCH_DEPTHS = {"basic", "advanced", "fast", "ultra-fast"}


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


def _normalize_domain_value(value: str | None) -> str | None:
    clean_value = _clean_text(value)
    if not clean_value:
        return None

    parsed = urlparse(clean_value if "://" in clean_value else f"https://{clean_value}")
    hostname = (parsed.netloc or parsed.path).lower().strip()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    hostname = hostname.split("/")[0].strip(".")
    return hostname or None


def _normalize_domain_list(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        domain = _normalize_domain_value(value)
        if not domain or domain in seen:
            continue
        seen.add(domain)
        normalized.append(domain)

    return normalized


def _domain_matches_any(domain: str | None, candidates: Sequence[str]) -> bool:
    if not domain:
        return False

    return any(domain == candidate or domain.endswith(f".{candidate}") for candidate in candidates)


def _normalize_search_depth(search_depth: str | None) -> str | None:
    clean_depth = _clean_text(search_depth)
    if not clean_depth:
        return None

    lowered = clean_depth.lower()
    return lowered if lowered in ALLOWED_TAVILY_SEARCH_DEPTHS else None


def _normalize_media_url(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, dict):
        for key in ("url", "src", "image", "href", "link"):
            candidate = _clean_text(value.get(key))
            if candidate:
                return candidate
        return None

    if isinstance(value, list):
        for entry in value:
            candidate = _normalize_media_url(entry)
            if candidate:
                return candidate
        return None

    return _clean_text(value)


def _normalize_media_description(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, dict):
        for key in ("description", "alt", "caption", "title", "text"):
            candidate = _clean_text(value.get(key))
            if candidate:
                return candidate
        return None

    if isinstance(value, list):
        for entry in value:
            candidate = _normalize_media_description(entry)
            if candidate:
                return candidate
        return None

    text = _clean_text(value)
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"...", "n/a", "na", "none", "null", "unknown", "unavailable"}:
        return None
    if not any(char.isalnum() for char in lowered):
        return None
    if "http://" in lowered or "https://" in lowered or "www." in lowered:
        return None
    if "placeholder" in lowered:
        return None
    if "..." in text or "…" in text:
        return None
    if lowered.startswith("image description") and any(
        token in lowered for token in ("missing", "unavailable", "n/a", "none", "unknown")
    ):
        return None

    return text


def _extract_tavily_media_fields(item: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    favicon = _normalize_media_url(
        item.get("favicon") or item.get("favicon_url") or item.get("faviconUrl") or item.get("site_icon")
    )
    image = _normalize_media_url(
        item.get("image")
        or item.get("image_url")
        or item.get("imageUrl")
        or item.get("images")
        or item.get("image_urls")
    )
    image_description = _normalize_media_description(
        item.get("image_description")
        or item.get("imageDescription")
        or item.get("image_alt")
        or item.get("imageAlt")
    )

    if image_description is None:
        image_description = _normalize_media_description(item.get("images"))

    return favicon, image, image_description


def _run_yfinance_safely(action):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        return action()


def _ranking_score_for_item(item: dict[str, Any]) -> float:
    value = item.get("ranking_score")
    if value is None:
        value = item.get("score")
    return _to_float_or_none(value) or 0.0


def _shortlist_priority(bucket: str | None) -> int:
    return {
        "high_confidence_company_specific": 3,
        "broader_context": 2,
        "weak_or_generic": 1,
    }.get(bucket or "", 0)


def _select_shortlisted_items(items: Sequence[dict[str, Any]], max_urls: int) -> list[dict[str, Any]]:
    capped_max_urls = max(1, min(int(max_urls), 3))
    deduped: list[dict[str, Any]] = []
    seen_url_keys: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        url_key = _normalize_url_key(item.get("url"))
        if not url_key or url_key in seen_url_keys:
            continue

        seen_url_keys.add(url_key)
        deduped.append(item)

    def sort_key(item: dict[str, Any]) -> tuple[int, float, datetime, str, str]:
        return (
            _shortlist_priority(item.get("relevance_bucket")),
            _ranking_score_for_item(item),
            parse_news_datetime(item.get("date")) or datetime.min.replace(tzinfo=timezone.utc),
            _clean_text(item.get("title")) or "",
            _clean_text(item.get("url")) or "",
        )

    return sorted(deduped, key=sort_key, reverse=True)[:capped_max_urls]


def _normalize_extraction_result(
    item: dict[str, Any],
    *,
    source_item: dict[str, Any] | None = None,
    extraction_status: str,
    extraction_error: str | None = None,
    selected_rank: int | None = None,
    extract_depth: str | None = None,
    extract_format: str | None = None,
) -> dict[str, Any]:
    source_item = source_item or {}
    url = _clean_text(item.get("url") or source_item.get("url"))
    title = _clean_text(item.get("title") or source_item.get("title"))
    publisher = _normalize_publisher(item.get("publisher") or source_item.get("publisher"), url=url)
    date = _normalize_date(item.get("date") or item.get("published_date") or source_item.get("date"))
    favicon, image, image_description = _extract_tavily_media_fields(item)
    if favicon is None or image is None or image_description is None:
        source_favicon, source_image, source_image_description = _extract_tavily_media_fields(source_item)
        favicon = favicon or source_favicon
        image = image or source_image
        image_description = image_description or source_image_description

    content = _clean_text(item.get("content") or item.get("raw_content") or item.get("text"))
    normalized: dict[str, Any] = {
        "title": title,
        "publisher": publisher,
        "date": date,
        "url": url,
        "favicon": favicon,
        "image": image,
        "image_description": image_description,
        "query_category": source_item.get("query_category"),
        "relevance_bucket": source_item.get("relevance_bucket"),
        "extracted": extraction_status == "success",
        "extraction_status": extraction_status,
        "source_type": "tavily_extraction",
        "caution": source_item.get("caution"),
        "ranking_score": _ranking_score_for_item(source_item),
        "snippet": source_item.get("snippet"),
        "content": content,
    }

    if selected_rank is not None:
        normalized["selected_rank"] = selected_rank
    if extract_depth is not None:
        normalized["extract_depth"] = extract_depth
    if extract_format is not None:
        normalized["extract_format"] = extract_format
    if extraction_error:
        normalized["extraction_error"] = extraction_error

    return normalized


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
        hist = _run_yfinance_safely(lambda: yf.Ticker(symbol).history(period="5d"))

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
        # Yahoo's recommendations surface is record-oriented and may be sparse,
        # delayed, or empty depending on the symbol.
        recs = _run_yfinance_safely(lambda: yf.Ticker(symbol).recommendations)

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
        # Yahoo's .info surface is a curated snapshot, not a schema-stable feed.
        info = _run_yfinance_safely(lambda: yf.Ticker(symbol).info or {})

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
        news = _run_yfinance_safely(lambda: yf.Ticker(symbol).news or [])

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
def get_company_news_tavily(
    symbol: str,
    company_name: str = "",
    num_stories: int = 5,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    search_depth: str | None = None,
) -> dict:
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

    normalized_include_domains = _normalize_domain_list(include_domains)
    normalized_exclude_domains = _normalize_domain_list(exclude_domains)
    search_exclude_domains = list(dict.fromkeys([*DEFAULT_TAVILY_EXCLUDE_DOMAINS, *normalized_exclude_domains]))
    normalized_search_depth = _normalize_search_depth(search_depth)

    try:
        client = TavilyClient(api_key=api_key)
        company_terms, company_phrase = build_company_terms(symbol, company_name)

        collected = []
        excluded_count = 0
        deduped_count = 0
        query_failures = []

        for query_info in queries:
            try:
                search_kwargs: dict[str, Any] = {
                    "query": query_info["query"],
                    "topic": "news",
                    "days": NEWS_QUERY_WINDOW_DAYS,
                    "max_results": max_results_per_query,
                    "include_raw_content": False,
                    "include_images": True,
                    "include_favicon": True,
                }
                if normalized_search_depth is not None:
                    search_kwargs["search_depth"] = normalized_search_depth
                if normalized_include_domains:
                    search_kwargs["include_domains"] = normalized_include_domains
                if search_exclude_domains:
                    search_kwargs["exclude_domains"] = search_exclude_domains

                response = client.search(**search_kwargs)
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

                favicon, image, image_description = _extract_tavily_media_fields(item)
                normalized = {
                    "title": title,
                    "publisher": _normalize_publisher(item.get("source"), url=url),
                    "date": _normalize_date(item.get("published_date") or item.get("publishedDate")),
                    "url": url,
                    "snippet": _clean_text(item.get("content")),
                    "score": _to_float_or_none(item.get("score")),
                    "query_category": query_info["query_category"],
                    "favicon": favicon,
                    "image": image,
                    "image_description": image_description,
                }

                scoring = score_tavily_news_item(
                    normalized,
                    symbol=symbol,
                    company_terms=company_terms,
                    company_phrase=company_phrase,
                )

                url_domain = _normalize_domain_value(url)
                if scoring["exclusion_reason"] is not None:
                    excluded_count += 1
                    continue
                if normalized_include_domains and not _domain_matches_any(url_domain, normalized_include_domains):
                    excluded_count += 1
                    continue
                if search_exclude_domains and _domain_matches_any(url_domain, search_exclude_domains):
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

        deduped_items: dict[str, dict[str, Any]] = {}
        title_index: dict[str, str] = {}
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
                "favicon": item.get("favicon"),
                "image": item.get("image"),
                "image_description": item.get("image_description"),
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


@tool
def selective_extract_shortlisted_urls_tavily(
    shortlisted_items: list[dict[str, Any]],
    query: str = "",
    max_urls: int = 3,
    extract_depth: str | None = "basic",
    format: str = "markdown",
    include_images: bool = True,
    include_favicon: bool = True,
    timeout: float = 30,
    chunks_per_source: int = 1,
) -> dict:
    """Extract a small shortlist of URLs after search-based ranking and filtering."""
    normalized_extract_depth = _clean_text(extract_depth)
    if normalized_extract_depth is not None:
        normalized_extract_depth = normalized_extract_depth.lower()
        if normalized_extract_depth not in {"basic", "advanced"}:
            normalized_extract_depth = "basic"
    else:
        normalized_extract_depth = "basic"

    normalized_format = _clean_text(format)
    if normalized_format is not None:
        normalized_format = normalized_format.lower()
        if normalized_format not in {"markdown", "text"}:
            normalized_format = "markdown"
    else:
        normalized_format = "markdown"

    selected_sources = _select_shortlisted_items(shortlisted_items, max_urls=max_urls)
    selected_urls = [source.get("url") for source in selected_sources if _clean_text(source.get("url"))]

    payload: dict[str, Any] = {
        "ok": True,
        "source": "Tavily selective extraction",
        "selection_strategy": "ranked_top_shortlist",
        "selected_count": len(selected_urls),
        "selected_urls": selected_urls,
        "query": _clean_text(query),
        "extract_depth": normalized_extract_depth,
        "format": normalized_format,
        "results": [],
        "failed_results": [],
    }

    if not selected_urls:
        return payload

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            **payload,
            "ok": False,
            "error": "TAVILY_API_KEY not set.",
        }

    try:
        client = TavilyClient(api_key=api_key)
        response = client.extract(
            urls=selected_urls,
            include_images=include_images,
            extract_depth=normalized_extract_depth,
            format=normalized_format,
            timeout=timeout,
            include_favicon=include_favicon,
            query=_clean_text(query),
            chunks_per_source=chunks_per_source,
        )
    except Exception as e:
        return {
            **payload,
            "ok": False,
            "error": str(e),
        }

    response_results = response.get("results", []) if isinstance(response, dict) else []
    response_failures = response.get("failed_results", []) if isinstance(response, dict) else []

    source_index = {
        _normalize_url_key(source.get("url")): source
        for source in selected_sources
        if _normalize_url_key(source.get("url"))
    }

    normalized_results: list[dict[str, Any]] = []
    for index, result in enumerate(response_results, start=1):
        if not isinstance(result, dict):
            continue

        url_key = _normalize_url_key(result.get("url"))
        source_item = source_index.get(url_key) if url_key else None
        normalized_results.append(
            _normalize_extraction_result(
                result,
                source_item=source_item,
                extraction_status="success",
                selected_rank=index,
                extract_depth=normalized_extract_depth,
                extract_format=normalized_format,
            )
        )

    normalized_failures: list[dict[str, Any]] = []
    for failure in response_failures:
        if not isinstance(failure, dict):
            continue

        url_key = _normalize_url_key(failure.get("url"))
        source_item = source_index.get(url_key) if url_key else None
        normalized_failures.append(
            _normalize_extraction_result(
                failure,
                source_item=source_item,
                extraction_status="failed",
                extraction_error=_clean_text(
                    failure.get("error") or failure.get("message") or failure.get("detail")
                ),
                extract_depth=normalized_extract_depth,
                extract_format=normalized_format,
            )
        )

    payload["results"] = normalized_results
    payload["failed_results"] = normalized_failures
    payload["returned_count"] = len(normalized_results)
    payload["failed_count"] = len(normalized_failures)

    return payload
