from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import os
import re
from urllib.parse import urlparse

from agno.tools import tool
from dotenv import load_dotenv
from tavily import TavilyClient
import yfinance as yf

load_dotenv()

_COMPANY_NAME_STOPWORDS = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
    "limited",
    "ltd",
    "group",
    "holdings",
    "sa",
    "ag",
    "nv",
    "the",
}

_PREFERRED_NEWS_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "apnews.com",
    "finance.yahoo.com",
    "sec.gov",
}

_EVENT_NEWS_DOMAINS = {
    "techcrunch.com",
    "theguardian.com",
    "theverge.com",
}

_LOW_VALUE_NEWS_DOMAINS = {
    "fool.com",
    "investorplace.com",
    "247wallst.com",
    "marketbeat.com",
    "etfdailynews.com",
    "defenseworld.net",
    "americanbankingnews.com",
    "stockanalysis.com",
    "tipranks.com",
    "barchart.com",
    "simplywall.st",
    "markets.financialcontent.com",
}

_COMMENTARY_NEWS_DOMAINS = {
    "seekingalpha.com",
    "tipranks.com",
    "fool.com",
    "investorplace.com",
    "marketbeat.com",
    "stockanalysis.com",
    "simplywall.st",
    "barchart.com",
}

_PR_AGGREGATOR_NEWS_DOMAINS = {
    "stocktitan.net",
    "mexc.com",
}

_MATERIAL_NEWS_HINTS = (
    "earnings",
    "guidance",
    "forecast",
    "outlook",
    "acquisition",
    "acquire",
    "merger",
    "partnership",
    "partner",
    "product launch",
    "launch",
    "announcement",
    "regulatory",
    "legal",
    "lawsuit",
    "antitrust",
    "investigation",
    "ceo",
    "cfo",
    "management",
    "interview",
    "commentary",
    "strategy",
    "strategic",
    "buyback",
    "capital return",
    "supply chain",
    "china",
    "commission",
    "acquires",
    "acquired",
)

_WEAK_NEWS_HINTS = (
    "is this stock a buy",
    "is this stock worth",
    "stock to buy",
    "top stocks",
    "best stocks",
    "price prediction",
    "institutional investors",
    "institutional investor",
    "hedge fund",
    "portfolio",
    "stake",
    "shares of",
    "market roundup",
    "market wrap",
    "premarket",
    "pre-market",
    "stocks to watch",
    "why this stock",
    "why the stock",
    "wall street analysts think",
    "analysts think",
    "dividend stock",
    "what to do now",
    "is it time to reassess",
    "valuation",
    "bullish",
    "bearish",
    "unwarranted",
    "shares sold by",
    "analyst calls",
    "good stock to buy",
)

_EXCLUDED_NEWS_HINTS = (
    "[poll]",
    "what are you most excited about",
    "most excited about",
    "looking forward to the most",
    "rumor roundup",
    "rumored",
    "rumours",
    "stock research article",
    "deep-dive:",
)

_QUERY_CATEGORY_WEIGHTS = {
    "broad_company_news": 0.6,
    "product_strategy": 1.1,
    "regulatory_legal": 1.1,
    "management_commentary": 1.0,
}

_EVENT_TOKEN_STOPWORDS = {
    "latest",
    "company",
    "companies",
    "stock",
    "stocks",
    "share",
    "shares",
    "news",
    "says",
    "say",
    "after",
    "amid",
    "over",
    "under",
    "from",
    "with",
    "without",
    "into",
    "year",
    "years",
    "this",
    "that",
    "today",
    "now",
    "report",
    "reports",
    "reported",
    "reportedly",
    "fend",
    "local",
}

_EVENT_TOKEN_ALIASES = {
    "price target": "price_target",
    "capital return": "capital_return",
    "product launch": "product_launch",
    "supply chain": "supply_chain",
}

_EVENT_CLUSTER_HINTS = {
    "commission",
    "china",
    "antitrust",
    "regulatory",
    "lawsuit",
    "guidance",
    "earnings",
    "acquisition",
    "acquired",
    "acquires",
    "partnership",
    "product_launch",
    "ceo",
    "cfo",
    "interview",
}

_MANAGEMENT_TOPIC_HINTS = {
    "interview": ("interview",),
    "anniversary": ("50th", "first 50 years", "anniversary"),
    "succession": ("retirement", "succession"),
    "tariffs": ("tariff", "refund"),
    "guidance": ("guidance", "forecast", "outlook"),
}


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


def _domain_key(url: str | None) -> str | None:
    if not url:
        return None

    hostname = urlparse(url).netloc.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname or None


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


def _source_preference_score(url: str | None) -> float:
    domain = _domain_key(url)
    if domain in _PREFERRED_NEWS_DOMAINS:
        return 2.0
    if domain in _EVENT_NEWS_DOMAINS:
        return 1.0
    if domain in _PR_AGGREGATOR_NEWS_DOMAINS:
        return -5.0
    if domain in _COMMENTARY_NEWS_DOMAINS:
        return -4.0
    if domain in _LOW_VALUE_NEWS_DOMAINS:
        return -2.5
    return 0.0


def _build_company_terms(symbol: str, company_name: str) -> tuple[set[str], str | None]:
    symbol = symbol.lower()
    company_terms = {symbol}

    clean_name = _clean_text(company_name)
    if not clean_name:
        return company_terms, None

    normalized_name = " ".join(re.findall(r"[a-z0-9]+", clean_name.lower()))
    for token in normalized_name.split():
        if len(token) < 3 or token in _COMPANY_NAME_STOPWORDS:
            continue
        company_terms.add(token)

    return company_terms, normalized_name or None


def _has_symbol(text: str, symbol: str) -> bool:
    return bool(re.search(rf"\b{re.escape(symbol.lower())}\b", text))


def _parse_news_datetime(value: str | None) -> datetime | None:
    clean_value = _clean_text(value)
    if not clean_value:
        return None

    candidates = [clean_value.replace("Z", "+00:00"), clean_value[:10]]
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue

    try:
        parsed = parsedate_to_datetime(clean_value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass

    return None


def _recency_bonus(date_value: str | None) -> float:
    parsed = _parse_news_datetime(date_value)
    if parsed is None:
        return 0.0

    age_days = max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400, 0.0)
    if age_days <= 2:
        return 1.5
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.5
    return 0.0


def _news_age_days(date_value: str | None) -> float | None:
    parsed = _parse_news_datetime(date_value)
    if parsed is None:
        return None

    return max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400, 0.0)


def _event_text_tokens(text: str, company_terms: set[str]) -> set[str]:
    normalized_text = text.lower()
    for phrase, alias in _EVENT_TOKEN_ALIASES.items():
        normalized_text = normalized_text.replace(phrase, alias)

    tokens = set()
    for token in re.findall(r"[a-z0-9_]+", normalized_text):
        if len(token) < 3:
            continue
        if token in company_terms or token in _COMPANY_NAME_STOPWORDS or token in _EVENT_TOKEN_STOPWORDS:
            continue
        tokens.add(token)

    return tokens


def _event_tokens_for_item(item: dict, company_terms: set[str]) -> set[str]:
    parts = [item.get("title") or ""]
    if item.get("snippet"):
        parts.append(str(item.get("snippet"))[:180])
    return _event_text_tokens(" ".join(parts), company_terms)


def _management_commentary_signature(item: dict) -> tuple[str | None, set[str]]:
    text = " ".join(
        part.lower()
        for part in [item.get("title") or "", item.get("snippet") or ""]
        if part
    )

    speaker = None
    if "ceo" in text:
        speaker = "ceo"
    elif "cfo" in text:
        speaker = "cfo"

    topics = {
        topic
        for topic, hints in _MANAGEMENT_TOPIC_HINTS.items()
        if any(hint in text for hint in hints)
    }

    return speaker, topics


def _is_similar_event(item: dict, other: dict, company_terms: set[str]) -> bool:
    if item.get("query_category") == other.get("query_category") == "management_commentary":
        speaker_a, topics_a = _management_commentary_signature(item)
        speaker_b, topics_b = _management_commentary_signature(other)
        same_domain = _domain_key(item.get("url")) == _domain_key(other.get("url"))

        if speaker_a and speaker_a == speaker_b:
            if same_domain:
                return True
            if topics_a & topics_b:
                return True

    tokens_a = _event_tokens_for_item(item, company_terms)
    tokens_b = _event_tokens_for_item(other, company_terms)
    if not tokens_a or not tokens_b:
        return False

    overlap = tokens_a & tokens_b
    if not overlap:
        return False

    overlap_ratio = len(overlap) / min(len(tokens_a), len(tokens_b))
    if overlap_ratio >= 0.6:
        return True

    event_hint_overlap = overlap & _EVENT_CLUSTER_HINTS
    if item.get("query_category") == other.get("query_category") and len(event_hint_overlap) >= 2:
        return True

    return len(event_hint_overlap) >= 2 and overlap_ratio >= 0.25


def _select_diverse_news_items(ranked_items: list[dict], num_stories: int, company_terms: set[str]) -> list[dict]:
    selected = []
    selected_keys = set()
    used_categories = set()

    non_weak_items = [item for item in ranked_items if item.get("relevance_bucket") != "weak_or_generic"]
    high_confidence_items = [
        item for item in non_weak_items if item.get("relevance_bucket") == "high_confidence_company_specific"
    ]
    broader_context_items = [
        item
        for item in non_weak_items
        if item.get("relevance_bucket") == "broader_context"
        and item.get("query_category") == "broad_company_news"
    ]
    minimum_fill = min(num_stories, 4)

    def can_add(
        item: dict,
        require_new_category: bool,
        avoid_similar_event: bool,
        allow_similar_cross_category: bool = False,
    ) -> bool:
        item_key = item.get("_item_key")
        if item_key in selected_keys:
            return False

        similar_to_selected = any(_is_similar_event(item, chosen, company_terms) for chosen in selected)
        same_category_items = [
            chosen for chosen in selected if chosen.get("query_category") == item.get("query_category")
        ]
        if require_new_category and item.get("query_category") in used_categories:
            return False
        if same_category_items and any(_is_similar_event(item, chosen, company_terms) for chosen in same_category_items):
            if item.get("query_category") == "management_commentary" and item.get("_ranking_score", 0.0) >= 1.0:
                return False
            return False
        if avoid_similar_event and similar_to_selected:
            return False
        if similar_to_selected and not avoid_similar_event and not allow_similar_cross_category:
            similar_categories = {
                chosen.get("query_category")
                for chosen in selected
                if _is_similar_event(item, chosen, company_terms)
            }
            if item.get("query_category") in similar_categories:
                return False

        return True

    def add_item(item: dict) -> None:
        selected.append(item)
        selected_keys.add(item.get("_item_key"))
        used_categories.add(item.get("query_category"))

    def try_add(
        items: list[dict],
        require_new_category: bool,
        avoid_similar_event: bool,
        allow_similar_cross_category: bool = False,
    ) -> None:
        for item in items:
            if not can_add(
                item,
                require_new_category=require_new_category,
                avoid_similar_event=avoid_similar_event,
                allow_similar_cross_category=allow_similar_cross_category,
            ):
                continue

            add_item(item)
            if len(selected) >= num_stories:
                return

    preferred_category_order = (
        "regulatory_legal",
        "product_strategy",
        "management_commentary",
        "broad_company_news",
    )

    for category in preferred_category_order:
        for item in high_confidence_items:
            if item.get("query_category") != category:
                continue
            if not can_add(item, require_new_category=False, avoid_similar_event=True):
                continue

            add_item(item)
            break

        if len(selected) >= num_stories:
            return selected[:num_stories]

    if len(selected) < num_stories:
        try_add(high_confidence_items, require_new_category=True, avoid_similar_event=True)
    if len(selected) < minimum_fill:
        for category in preferred_category_order:
            if category in used_categories:
                continue
            category_items = [item for item in high_confidence_items if item.get("query_category") == category]
            try_add(
                category_items,
                require_new_category=False,
                avoid_similar_event=False,
                allow_similar_cross_category=True,
            )
            if len(selected) >= minimum_fill:
                break
    if len(selected) < minimum_fill:
        try_add(high_confidence_items, require_new_category=False, avoid_similar_event=False)
    if len(selected) < num_stories:
        try_add(broader_context_items, require_new_category=False, avoid_similar_event=False)

    return selected[:num_stories]


def _score_tavily_news_item(
    item: dict,
    symbol: str,
    company_terms: set[str],
    company_phrase: str | None,
) -> tuple[float, str, str | None]:
    title = (item.get("title") or "").lower()
    snippet = (item.get("snippet") or "").lower()
    publisher = (item.get("publisher") or "").lower()
    domain = _domain_key(item.get("url"))
    text = " ".join(part for part in [title, snippet, publisher] if part)

    matched_terms = [
        term for term in company_terms if term != symbol.lower() and len(term) >= 3 and term in text
    ]
    symbol_match = _has_symbol(text, symbol)
    strong_name_match = bool(company_phrase and company_phrase in text)
    material_hits = sum(1 for hint in _MATERIAL_NEWS_HINTS if hint in text)
    weak_hits = sum(1 for hint in _WEAK_NEWS_HINTS if hint in text)
    weak_pattern = weak_hits > 0
    excluded_pattern = any(hint in text for hint in _EXCLUDED_NEWS_HINTS)
    domain_source_score = _source_preference_score(item.get("url"))
    commentary_domain = domain in _COMMENTARY_NEWS_DOMAINS
    pr_aggregator_domain = domain in _PR_AGGREGATOR_NEWS_DOMAINS
    age_days = _news_age_days(item.get("date"))

    score = _to_float_or_none(item.get("score")) or 0.0
    score += _QUERY_CATEGORY_WEIGHTS.get(item.get("query_category"), 0.0)
    score += _recency_bonus(item.get("date"))
    score += domain_source_score

    if strong_name_match:
        score += 3.0
    elif len(matched_terms) >= 2:
        score += 2.0
    elif len(matched_terms) == 1:
        score += 1.0

    if symbol_match:
        score += 1.5

    score += min(material_hits, 2) * 1.0
    if weak_pattern:
        score -= min(weak_hits, 2) * 3.0

    if not strong_name_match and not symbol_match and not matched_terms:
        return score - 4.0, "excluded", "not_company_specific"

    if excluded_pattern:
        return score, "excluded", "low_signal_format"

    if age_days is not None and age_days > 90:
        return score, "excluded", "stale_result"

    if pr_aggregator_domain:
        return score, "excluded", "pr_aggregator"

    if commentary_domain and weak_hits and material_hits <= 1:
        return score, "excluded", "commentary_opinion"

    if weak_hits >= 2 and material_hits == 0:
        return score, "excluded", "weak_generic"

    if weak_pattern and domain in _LOW_VALUE_NEWS_DOMAINS and material_hits <= 1:
        return score, "excluded", "weak_generic"

    if weak_pattern or commentary_domain or domain in _LOW_VALUE_NEWS_DOMAINS:
        bucket = "weak_or_generic"
    elif strong_name_match or symbol_match or len(matched_terms) >= 2 or material_hits:
        bucket = "high_confidence_company_specific"
        score += 1.0
    else:
        bucket = "broader_context"

    return score, bucket, None


def _select_preferred_news_item(existing: dict, candidate: dict) -> dict:
    existing_score = existing.get("_ranking_score", 0.0)
    candidate_score = candidate.get("_ranking_score", 0.0)
    if candidate_score > existing_score:
        return candidate
    if candidate_score < existing_score:
        return existing

    existing_date = _parse_news_datetime(existing.get("date"))
    candidate_date = _parse_news_datetime(candidate.get("date"))
    if existing_date and candidate_date and candidate_date > existing_date:
        return candidate

    return existing


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
    company_name = _clean_text(company_name) or ""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "symbol": symbol,
            "ok": False,
            "error": "TAVILY_API_KEY not set.",
        }

    company_label = f"{company_name} ({symbol})" if company_name else symbol
    max_results_per_query = max(2, min(num_stories, 3))
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
        company_terms, company_phrase = _build_company_terms(symbol, company_name)

        collected = []
        excluded_count = 0
        deduped_count = 0
        query_failures = []

        for query_info in queries:
            try:
                response = client.search(
                    query=query_info["query"],
                    topic="news",
                    days=30,
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

                ranking_score, relevance_bucket, exclusion_reason = _score_tavily_news_item(
                    normalized,
                    symbol=symbol,
                    company_terms=company_terms,
                    company_phrase=company_phrase,
                )

                if exclusion_reason is not None:
                    excluded_count += 1
                    continue

                normalized["relevance_bucket"] = relevance_bucket
                normalized["_ranking_score"] = ranking_score
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
                deduped_items[existing_key] = _select_preferred_news_item(
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
                _parse_news_datetime(item.get("date")) or datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )

        selected = _select_diverse_news_items(ranked_items, num_stories, company_terms)

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
            normalized.append(
                {
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "date": item.get("date"),
                    "url": item.get("url"),
                    "snippet": item.get("snippet"),
                    "score": item.get("score"),
                    "query_category": item.get("query_category"),
                    "relevance_bucket": item.get("relevance_bucket"),
                }
            )

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
            **({"query_failures": query_failures} if query_failures else {}),
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "ok": False,
            "error": str(e),
            "queries_used": queries,
            "query_used": queries[0]["query"],
        }
        
