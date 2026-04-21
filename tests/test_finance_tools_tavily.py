import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import finance_tools


class _FakeSearchClient:
    calls: list[dict] = []

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, **kwargs):
        self.__class__.calls.append(kwargs)
        return {
            "results": [
                {
                    "title": "Microsoft expands AI partnership",
                    "url": "https://www.reuters.com/business/microsoft-ai/",
                    "source": {"displayName": "Reuters"},
                    "published_date": "2026-04-18T00:00:00+00:00",
                    "content": "Microsoft expanded an AI partnership.",
                    "score": 9.1,
                    "favicon": "https://www.reuters.com/favicon.ico",
                    "image": "https://www.reuters.com/image.jpg",
                    "image_description": "Reuters photo",
                },
                {
                    "title": "Microsoft commentary piece",
                    "url": "https://bad.com/microsoft-commentary/",
                    "source": "Bad Source",
                    "published_date": "2026-04-17T00:00:00+00:00",
                    "content": "A low quality commentary item.",
                    "score": 7.2,
                    "favicon": "https://bad.com/favicon.ico",
                    "image": "https://bad.com/image.jpg",
                    "image_description": "Bad photo",
                },
                {
                    "title": "Microsoft junk aggregation",
                    "url": "https://www.stocktitan.net/microsoft-junk/",
                    "source": "StockTitan",
                    "published_date": "2026-04-16T00:00:00+00:00",
                    "content": "A junk aggregation item.",
                    "score": 6.5,
                },
            ]
        }


class _FakeExtractClient:
    calls: list[dict] = []

    def __init__(self, api_key: str):
        self.api_key = api_key

    def extract(self, **kwargs):
        self.__class__.calls.append(kwargs)
        return {
            "results": [
                {
                    "title": "Microsoft expands AI partnership",
                    "url": "https://www.reuters.com/business/microsoft-ai/",
                    "content": "Long extracted content about the partnership.",
                    "favicon": "https://www.reuters.com/favicon.ico",
                    "image": "https://www.reuters.com/image.jpg",
                    "image_description": "Reuters photo",
                },
                {
                    "title": "Microsoft context coverage",
                    "url": "https://www.bloomberg.com/news/microsoft-context/",
                    "content": "Context extracted content.",
                    "favicon": "https://www.bloomberg.com/favicon.ico",
                    "image": "https://www.bloomberg.com/image.jpg",
                    "image_description": "Bloomberg photo",
                },
            ],
            "failed_results": [
                {
                    "title": "Microsoft commentary piece",
                    "url": "https://bad.com/microsoft-commentary/",
                    "error": "timeout",
                }
            ],
        }


def test_get_company_news_tavily_honors_include_and_exclude_domains(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(finance_tools, "TavilyClient", _FakeSearchClient)
    monkeypatch.setattr(
        finance_tools,
        "score_tavily_news_item",
        lambda item, symbol, company_terms, company_phrase: {
            "score": item.get("score") or 0.0,
            "bucket": "high_confidence_company_specific",
            "exclusion_reason": None,
            "reason_summary": {"policy_version": finance_tools.NEWS_FILTER_POLICY_VERSION},
        },
    )

    _FakeSearchClient.calls.clear()
    result = finance_tools.get_company_news_tavily.entrypoint(
        "MSFT",
        company_name="Microsoft Corporation",
        num_stories=3,
        include_domains=["reuters.com"],
        exclude_domains=["bad.com"],
        search_depth="advanced",
    )

    assert result["ok"] is True
    assert result["returned_count"] == 1
    assert result["news"][0]["title"] == "Microsoft expands AI partnership"
    assert result["news"][0]["favicon"] == "https://www.reuters.com/favicon.ico"
    assert result["news"][0]["image"] == "https://www.reuters.com/image.jpg"
    assert result["news"][0]["image_description"] == "Reuters photo"
    assert len(_FakeSearchClient.calls) == 4
    for call in _FakeSearchClient.calls:
        assert call["search_depth"] == "advanced"
        assert call["include_domains"] == ["reuters.com"]
        assert "bad.com" in call["exclude_domains"]
        assert "stocktitan.net" in call["exclude_domains"]
        assert call["include_images"] is True
        assert call["include_favicon"] is True


def test_selective_extract_shortlisted_urls_tavily_normalizes_records(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(finance_tools, "TavilyClient", _FakeExtractClient)

    _FakeExtractClient.calls.clear()
    shortlisted_items = [
        {
            "title": "Microsoft expands AI partnership",
            "publisher": "Reuters",
            "date": "2026-04-18T00:00:00+00:00",
            "url": "https://www.reuters.com/business/microsoft-ai/",
            "score": 9.1,
            "query_category": "product_strategy",
            "relevance_bucket": "high_confidence_company_specific",
            "favicon": "https://www.reuters.com/favicon.ico",
        },
        {
            "title": "Microsoft context coverage",
            "publisher": "Bloomberg",
            "date": "2026-04-17T00:00:00+00:00",
            "url": "https://www.bloomberg.com/news/microsoft-context/",
            "score": 8.2,
            "query_category": "broad_company_news",
            "relevance_bucket": "broader_context",
            "favicon": "https://www.bloomberg.com/favicon.ico",
        },
        {
            "title": "Microsoft commentary piece",
            "publisher": "Bad Source",
            "date": "2026-04-16T00:00:00+00:00",
            "url": "https://bad.com/microsoft-commentary/",
            "score": 7.1,
            "query_category": "management_commentary",
            "relevance_bucket": "weak_or_generic",
            "favicon": "https://bad.com/favicon.ico",
        },
    ]

    result = finance_tools.selective_extract_shortlisted_urls_tavily.entrypoint(
        shortlisted_items,
        query="MSFT selective verification",
        max_urls=3,
        extract_depth="advanced",
    )

    assert result["ok"] is True
    assert result["selected_urls"] == [
        "https://www.reuters.com/business/microsoft-ai/",
        "https://www.bloomberg.com/news/microsoft-context/",
        "https://bad.com/microsoft-commentary/",
    ]
    assert len(_FakeExtractClient.calls) == 1
    assert _FakeExtractClient.calls[0]["urls"] == result["selected_urls"]
    assert _FakeExtractClient.calls[0]["extract_depth"] == "advanced"
    assert result["results"][0]["title"] == "Microsoft expands AI partnership"
    assert result["results"][0]["extracted"] is True
    assert result["results"][0]["extraction_status"] == "success"
    assert result["results"][0]["source_type"] == "tavily_extraction"
    assert result["results"][0]["ranking_score"] == 9.1
    assert result["results"][0]["content"] == "Long extracted content about the partnership."
    assert result["failed_results"][0]["extraction_status"] == "failed"
    assert result["failed_results"][0]["extraction_error"] == "timeout"
    assert result["failed_results"][0]["source_type"] == "tavily_extraction"


def test_extract_tavily_media_fields_ignores_junk_image_description():
    favicon, image, image_description = finance_tools._extract_tavily_media_fields(
        {
            "favicon": "https://www.reuters.com/favicon.ico",
            "image": "https://www.reuters.com/image.jpg",
            "image_description": "...",
        }
    )

    assert favicon == "https://www.reuters.com/favicon.ico"
    assert image == "https://www.reuters.com/image.jpg"
    assert image_description is None
