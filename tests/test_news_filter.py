import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import news_filter


def _days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _news_item(**overrides):
    item = {
        "title": "Apple announces supply chain update after earnings",
        "snippet": "Apple said guidance remains intact and highlighted operational changes.",
        "publisher": "Reuters",
        "url": "https://www.reuters.com/markets/markets/apple-update/",
        "date": _days_ago(3),
        "score": 1.0,
        "query_category": "broad_company_news",
    }
    item.update(overrides)
    return item


def test_source_preference_score_prefers_higher_quality_sources():
    assert news_filter.source_preference_score("https://www.reuters.com/world") > news_filter.source_preference_score(
        "https://www.fool.com/investing"
    )
    assert news_filter.source_preference_score("https://www.reuters.com/world") > 0


def test_score_tavily_news_item_excludes_stale_results():
    company_terms, company_phrase = news_filter.build_company_terms("AAPL", "Apple Inc")
    result = news_filter.score_tavily_news_item(
        _news_item(
            date=_days_ago(120),
            title="Apple announces supply chain update after earnings",
            snippet="Apple said guidance remains intact and highlighted operational changes.",
        ),
        symbol="AAPL",
        company_terms=company_terms,
        company_phrase=company_phrase,
    )

    assert result["bucket"] == "excluded"
    assert result["exclusion_reason"] == "stale_result"
    assert result["reason_summary"]["policy_version"] == news_filter.NEWS_FILTER_POLICY_VERSION
    assert result["reason_summary"]["decision"] == "stale_result"


def test_score_tavily_news_item_excludes_weak_commentary():
    company_terms, company_phrase = news_filter.build_company_terms("AAPL", "Apple Inc")
    result = news_filter.score_tavily_news_item(
        _news_item(
            title="Is Apple a good stock to buy after earnings?",
            snippet="Analysts think the valuation still looks compelling.",
            publisher="Seeking Alpha",
            url="https://seekingalpha.com/article/123456",
        ),
        symbol="AAPL",
        company_terms=company_terms,
        company_phrase=company_phrase,
    )

    assert result["bucket"] == "excluded"
    assert result["exclusion_reason"] == "commentary_opinion"
    assert "commentary_domain" in result["reason_summary"]["flags"]


def test_score_tavily_news_item_checks_company_specificity():
    company_terms, company_phrase = news_filter.build_company_terms("AAPL", "Apple Inc")

    generic = news_filter.score_tavily_news_item(
        _news_item(
            title="Markets rally on macro data",
            snippet="Analysts discuss valuation and portfolio positioning.",
            url="https://www.reuters.com/markets/global-markets/",
        ),
        symbol="AAPL",
        company_terms=company_terms,
        company_phrase=company_phrase,
    )
    specific = news_filter.score_tavily_news_item(
        _news_item(
            title="Apple expands services revenue after earnings",
            snippet="Apple said guidance remains intact and the company is broadening services.",
            url="https://www.reuters.com/markets/apple-services/",
        ),
        symbol="AAPL",
        company_terms=company_terms,
        company_phrase=company_phrase,
    )

    assert generic["exclusion_reason"] == "not_company_specific"
    assert specific["bucket"] == "high_confidence_company_specific"
    assert specific["exclusion_reason"] is None
    assert specific["reason_summary"]["matched_terms"] >= 1


def test_select_preferred_news_item_prefers_higher_score_or_newer_duplicate():
    existing = {
        "_ranking_score": 3.0,
        "date": "2026-04-01T00:00:00+00:00",
    }
    higher_score = {
        "_ranking_score": 4.0,
        "date": "2026-03-01T00:00:00+00:00",
    }
    newer_same_score = {
        "_ranking_score": 3.0,
        "date": "2026-04-02T00:00:00+00:00",
    }

    assert news_filter.select_preferred_news_item(existing, higher_score) is higher_score
    assert news_filter.select_preferred_news_item(existing, newer_same_score) is newer_same_score


def test_select_diverse_news_items_suppresses_similar_management_commentary():
    company_terms, _ = news_filter.build_company_terms("AAPL", "Apple Inc")
    ranked_items = [
        {
            "_item_key": "item-1",
            "_ranking_score": 9.0,
            "title": "Apple CEO says guidance remains intact in interview",
            "snippet": "The CEO repeated the guidance view and discussed tariffs.",
            "url": "https://www.reuters.com/business/apple-ceo-interview/",
            "date": _days_ago(2),
            "query_category": "management_commentary",
            "relevance_bucket": "high_confidence_company_specific",
        },
        {
            "_item_key": "item-2",
            "_ranking_score": 8.5,
            "title": "Apple CEO discusses guidance and outlook again",
            "snippet": "The CEO repeated the same guidance message in a follow-up interview.",
            "url": "https://www.reuters.com/business/apple-ceo-followup/",
            "date": _days_ago(1),
            "query_category": "management_commentary",
            "relevance_bucket": "high_confidence_company_specific",
        },
    ]

    selected = news_filter.select_diverse_news_items(ranked_items, num_stories=2, company_terms=company_terms)

    assert len(selected) == 1
    assert selected[0]["_item_key"] == "item-1"
