from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from urllib.parse import urlparse

# Intentional policy surface for the evidence layer.
NEWS_FILTER_POLICY_VERSION = "v1"

# Company identity normalization policy.
COMPANY_NAME_STOPWORDS = {
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

# Source preference policy.
SOURCE_DOMAIN_PREFERENCES: dict[str, float] = {
    "reuters.com": 2.0,
    "bloomberg.com": 2.0,
    "wsj.com": 2.0,
    "ft.com": 2.0,
    "cnbc.com": 2.0,
    "apnews.com": 2.0,
    "finance.yahoo.com": 2.0,
    "sec.gov": 2.0,
    "techcrunch.com": 1.0,
    "theguardian.com": 1.0,
    "theverge.com": 1.0,
    "247wallst.com": -2.5,
    "etfdailynews.com": -2.5,
    "defenseworld.net": -2.5,
    "americanbankingnews.com": -2.5,
    "markets.financialcontent.com": -2.5,
    "seekingalpha.com": -4.0,
    "tipranks.com": -4.0,
    "fool.com": -4.0,
    "investorplace.com": -4.0,
    "marketbeat.com": -4.0,
    "stockanalysis.com": -4.0,
    "simplywall.st": -4.0,
    "barchart.com": -4.0,
    "stocktitan.net": -5.0,
    "mexc.com": -5.0,
}

# News text signal policy.
MATERIAL_NEWS_HINTS = (
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

WEAK_NEWS_HINTS = (
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

EXCLUDED_NEWS_HINTS = (
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

# Query routing policy.
QUERY_CATEGORY_WEIGHTS = {
    "broad_company_news": 0.6,
    "product_strategy": 1.1,
    "regulatory_legal": 1.1,
    "management_commentary": 1.0,
}

# Event clustering and specificity policy.
EVENT_TOKEN_STOPWORDS = {
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

EVENT_TOKEN_ALIASES = {
    "price target": "price_target",
    "capital return": "capital_return",
    "product launch": "product_launch",
    "supply chain": "supply_chain",
}

EVENT_CLUSTER_HINTS = {
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

MANAGEMENT_TOPIC_HINTS = {
    "interview": ("interview",),
    "anniversary": ("50th", "first 50 years", "anniversary"),
    "succession": ("retirement", "succession"),
    "tariffs": ("tariff", "refund"),
    "guidance": ("guidance", "forecast", "outlook"),
}

__all__ = [
    "NEWS_FILTER_POLICY_VERSION",
    "COMPANY_NAME_STOPWORDS",
    "SOURCE_DOMAIN_PREFERENCES",
    "MATERIAL_NEWS_HINTS",
    "WEAK_NEWS_HINTS",
    "EXCLUDED_NEWS_HINTS",
    "QUERY_CATEGORY_WEIGHTS",
    "EVENT_TOKEN_STOPWORDS",
    "EVENT_TOKEN_ALIASES",
    "EVENT_CLUSTER_HINTS",
    "MANAGEMENT_TOPIC_HINTS",
    "source_preference_score",
    "build_company_terms",
    "parse_news_datetime",
    "select_diverse_news_items",
    "score_tavily_news_item",
    "select_preferred_news_item",
]

# Backward-compatible aliases for existing internal imports and call sites.
_COMPANY_NAME_STOPWORDS = COMPANY_NAME_STOPWORDS
_DOMAIN_SCORES = SOURCE_DOMAIN_PREFERENCES
_MATERIAL_NEWS_HINTS = MATERIAL_NEWS_HINTS
_WEAK_NEWS_HINTS = WEAK_NEWS_HINTS
_EXCLUDED_NEWS_HINTS = EXCLUDED_NEWS_HINTS
_QUERY_CATEGORY_WEIGHTS = QUERY_CATEGORY_WEIGHTS
_EVENT_TOKEN_STOPWORDS = EVENT_TOKEN_STOPWORDS
_EVENT_TOKEN_ALIASES = EVENT_TOKEN_ALIASES
_EVENT_CLUSTER_HINTS = EVENT_CLUSTER_HINTS
_MANAGEMENT_TOPIC_HINTS = MANAGEMENT_TOPIC_HINTS


def _clean_text(value) -> str | None:
    if value is None:
        return None

    text = value.strip() if isinstance(value, str) else str(value).strip()
    return text or None


def _to_float_or_none(value) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _domain_key(url: str | None) -> str | None:
    if not url:
        return None

    hostname = urlparse(url).netloc.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname or None


def source_preference_score(url: str | None) -> float:
    domain = _domain_key(url)
    return SOURCE_DOMAIN_PREFERENCES.get(domain, 0.0)


def build_company_terms(symbol: str, company_name: str) -> tuple[set[str], str | None]:
    symbol = symbol.lower()
    company_terms = {symbol}

    clean_name = _clean_text(company_name)
    if not clean_name:
        return company_terms, None

    normalized_name = " ".join(re.findall(r"[a-z0-9]+", clean_name.lower()))
    for token in normalized_name.split():
        if len(token) < 3 or token in COMPANY_NAME_STOPWORDS:
            continue
        company_terms.add(token)

    return company_terms, normalized_name or None


def _has_symbol(text: str, symbol: str) -> bool:
    return bool(re.search(rf"\b{re.escape(symbol.lower())}\b", text))


def parse_news_datetime(value: str | None) -> datetime | None:
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
    parsed = parse_news_datetime(date_value)
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
    parsed = parse_news_datetime(date_value)
    if parsed is None:
        return None

    return max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400, 0.0)


def _event_text_tokens(text: str, company_terms: set[str]) -> set[str]:
    normalized_text = text.lower()
    for phrase, alias in EVENT_TOKEN_ALIASES.items():
        normalized_text = normalized_text.replace(phrase, alias)

    tokens = set()
    for token in re.findall(r"[a-z0-9_]+", normalized_text):
        if len(token) < 3:
            continue
        if token in company_terms or token in COMPANY_NAME_STOPWORDS or token in EVENT_TOKEN_STOPWORDS:
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
        for topic, hints in MANAGEMENT_TOPIC_HINTS.items()
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

    event_hint_overlap = overlap & EVENT_CLUSTER_HINTS
    if item.get("query_category") == other.get("query_category") and len(event_hint_overlap) >= 2:
        return True

    return len(event_hint_overlap) >= 2 and overlap_ratio >= 0.25


def select_diverse_news_items(ranked_items: list[dict], num_stories: int, company_terms: set[str]) -> list[dict]:
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


def _build_score_reason_summary(
    *,
    bucket: str,
    exclusion_reason: str | None,
    domain: str | None,
    domain_source_score: float,
    recency_bonus: float,
    age_days: float | None,
    matched_terms: int,
    symbol_match: bool,
    strong_name_match: bool,
    material_hits: int,
    weak_hits: int,
    commentary_domain: bool,
    pr_aggregator_domain: bool,
    excluded_pattern: bool,
) -> dict:
    flags = []
    if strong_name_match:
        flags.append("strong_name_match")
    if symbol_match:
        flags.append("symbol_match")
    if commentary_domain:
        flags.append("commentary_domain")
    if pr_aggregator_domain:
        flags.append("pr_aggregator_domain")
    if excluded_pattern:
        flags.append("excluded_pattern")

    return {
        "policy_version": NEWS_FILTER_POLICY_VERSION,
        "decision": exclusion_reason or bucket,
        "domain": domain,
        "domain_score": domain_source_score,
        "recency_bonus": recency_bonus,
        "age_days": round(age_days, 1) if age_days is not None else None,
        "matched_terms": matched_terms,
        "material_hits": material_hits,
        "weak_hits": weak_hits,
        "flags": flags,
        "exclusion_reason": exclusion_reason,
    }


def score_tavily_news_item(
    item: dict,
    symbol: str,
    company_terms: set[str],
    company_phrase: str | None,
) -> dict:
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
    material_hits = sum(1 for hint in MATERIAL_NEWS_HINTS if hint in text)
    weak_hits = sum(1 for hint in WEAK_NEWS_HINTS if hint in text)
    weak_pattern = weak_hits > 0
    excluded_pattern = any(hint in text for hint in EXCLUDED_NEWS_HINTS)
    domain_source_score = source_preference_score(item.get("url"))
    commentary_domain = SOURCE_DOMAIN_PREFERENCES.get(domain, 0.0) == -4.0
    pr_aggregator_domain = SOURCE_DOMAIN_PREFERENCES.get(domain, 0.0) == -5.0
    age_days = _news_age_days(item.get("date"))
    recency_bonus = _recency_bonus(item.get("date"))

    score = _to_float_or_none(item.get("score")) or 0.0
    score += QUERY_CATEGORY_WEIGHTS.get(item.get("query_category"), 0.0)
    score += recency_bonus
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

    bucket = ""
    exclusion_reason: str | None = None

    if not strong_name_match and not symbol_match and not matched_terms:
        score -= 4.0
        bucket = "excluded"
        exclusion_reason = "not_company_specific"

    elif excluded_pattern:
        bucket = "excluded"
        exclusion_reason = "low_signal_format"

    elif age_days is not None and age_days > 90:
        bucket = "excluded"
        exclusion_reason = "stale_result"

    elif pr_aggregator_domain:
        bucket = "excluded"
        exclusion_reason = "pr_aggregator"

    elif commentary_domain and weak_hits and material_hits <= 1:
        bucket = "excluded"
        exclusion_reason = "commentary_opinion"

    elif weak_hits >= 2 and material_hits == 0:
        bucket = "excluded"
        exclusion_reason = "weak_generic"

    elif weak_pattern and SOURCE_DOMAIN_PREFERENCES.get(domain, 0.0) == -2.5 and material_hits <= 1:
        bucket = "excluded"
        exclusion_reason = "weak_generic"

    elif weak_pattern or commentary_domain or SOURCE_DOMAIN_PREFERENCES.get(domain, 0.0) == -2.5:
        bucket = "weak_or_generic"
    elif strong_name_match or symbol_match or len(matched_terms) >= 2 or material_hits:
        bucket = "high_confidence_company_specific"
        score += 1.0
    else:
        bucket = "broader_context"

    return {
        "score": score,
        "bucket": bucket,
        "exclusion_reason": exclusion_reason,
        "reason_summary": _build_score_reason_summary(
            bucket=bucket,
            exclusion_reason=exclusion_reason,
            domain=domain,
            domain_source_score=domain_source_score,
            recency_bonus=recency_bonus,
            age_days=age_days,
            matched_terms=len(matched_terms),
            symbol_match=symbol_match,
            strong_name_match=strong_name_match,
            material_hits=material_hits,
            weak_hits=weak_hits,
            commentary_domain=commentary_domain,
            pr_aggregator_domain=pr_aggregator_domain,
            excluded_pattern=excluded_pattern,
        ),
    }


def select_preferred_news_item(existing: dict, candidate: dict) -> dict:
    existing_score = existing.get("_ranking_score", 0.0)
    candidate_score = candidate.get("_ranking_score", 0.0)
    if candidate_score > existing_score:
        return candidate
    if candidate_score < existing_score:
        return existing

    existing_date = parse_news_datetime(existing.get("date"))
    candidate_date = parse_news_datetime(candidate.get("date"))
    if existing_date and candidate_date and candidate_date > existing_date:
        return candidate

    return existing


def _score_tavily_news_item(
    item: dict,
    symbol: str,
    company_terms: set[str],
    company_phrase: str | None,
) -> tuple[float, str, str | None]:
    result = score_tavily_news_item(
        item,
        symbol=symbol,
        company_terms=company_terms,
        company_phrase=company_phrase,
    )
    return result["score"], result["bucket"], result["exclusion_reason"]


# Backward-compatible aliases for older internal imports and call sites.
_source_preference_score = source_preference_score
_build_company_terms = build_company_terms
_parse_news_datetime = parse_news_datetime
_select_diverse_news_items = select_diverse_news_items
_select_preferred_news_item = select_preferred_news_item
