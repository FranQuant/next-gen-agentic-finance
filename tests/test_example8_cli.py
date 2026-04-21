import json
import sys
import types
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import example8


def _packet(labels: tuple[str, ...]) -> str:
    return "\n".join(f"- {label}: value" for label in labels)


def _handoff(evidence: str, research_read: str, open_questions: str, title_2: str = "OPEN QUESTIONS / GAPS") -> str:
    return "\n\n".join(
        [
            "EVIDENCE",
            evidence,
            "RESEARCH READ",
            research_read,
            title_2,
            open_questions,
        ]
    )


class _FakeRunOutput:
    def __init__(self, content: str):
        self.content = content
        self.run_id = "run-123"
        self.session_id = "session-456"
        self.status = types.SimpleNamespace(value="completed")


class _FakeTeam:
    def __init__(self, output: str):
        self.output = output
        self.name = "Fake Team"
        self.model = types.SimpleNamespace(id="gpt-5.4")
        self.members = [
            types.SimpleNamespace(name="member-a", model=types.SimpleNamespace(id="gpt-5.4")),
            types.SimpleNamespace(name="member-b", model=types.SimpleNamespace(id="gpt-5.4")),
        ]

    def print_response(self, prompt: str, stream: bool = True, **kwargs) -> None:
        sys.stdout.write(self.output)

    def get_last_run_output(self):
        return _FakeRunOutput(self.output)


def test_run_demo_default_prints_only_final_handoff(monkeypatch, capsys):
    evidence = _packet(example8.EVIDENCE_LABELS)
    research_read = _packet(example8.RESEARCH_READ_LABELS)
    open_questions = _packet(example8.OPEN_QUESTIONS_LABELS)
    final_text = _handoff(evidence, research_read, open_questions)

    monkeypatch.setattr(example8, "build_team", lambda: _FakeTeam(final_text))

    example8.run_demo(example8.DEMO_PROMPT)

    assert capsys.readouterr().out == final_text


def test_run_demo_expanded_sections_are_additive(monkeypatch, capsys):
    evidence = _packet(example8.EVIDENCE_LABELS)
    research_read = _packet(example8.RESEARCH_READ_LABELS)
    open_questions = _packet(example8.OPEN_QUESTIONS_LABELS)
    final_text = _handoff(evidence, research_read, open_questions)

    fake_team = _FakeTeam(final_text)
    monkeypatch.setattr(example8, "build_team", lambda: fake_team)
    monkeypatch.setattr(
        example8,
        "get_company_info",
        lambda symbol: {
            "ok": True,
            "company_info": {"longName": "Microsoft Corporation"},
            "source": "yfinance info",
        },
    )
    monkeypatch.setattr(
        example8,
        "get_company_news_tavily",
        lambda symbol, company_name="", num_stories=5: {
            "ok": True,
            "news": [
                {
                    "title": "Microsoft expands AI partnership",
                    "publisher": "Reuters",
                    "date": "2026-04-18T00:00:00+00:00",
                    "url": "https://www.reuters.com/business/microsoft-ai/",
                    "snippet": "Microsoft expanded an AI partnership.",
                    "score": 9.1,
                    "query_category": "product_strategy",
                    "relevance_bucket": "high_confidence_company_specific",
                    "favicon": "https://www.reuters.com/favicon.ico",
                    "image": "https://www.reuters.com/image.jpg",
                    "image_description": "Reuters photo",
                },
                {
                    "title": "Microsoft context coverage",
                    "publisher": "Bloomberg",
                    "date": "2026-04-17T00:00:00+00:00",
                    "url": "https://www.bloomberg.com/news/microsoft-context/",
                    "snippet": "A contextual market note.",
                    "score": 7.2,
                    "query_category": "broad_company_news",
                    "relevance_bucket": "broader_context",
                    "favicon": "https://www.bloomberg.com/favicon.ico",
                    "image": "https://www.bloomberg.com/image.jpg",
                    "image_description": "Bloomberg photo",
                },
                {
                    "title": "Microsoft commentary piece",
                    "publisher": "Example Wire",
                    "date": "2026-04-16T00:00:00+00:00",
                    "url": "https://www.example.com/microsoft-commentary/",
                    "snippet": "A weak commentary item.",
                    "score": 5.0,
                    "query_category": "management_commentary",
                    "relevance_bucket": "weak_or_generic",
                    "favicon": "https://www.example.com/favicon.ico",
                    "image": "https://www.example.com/image.jpg",
                    "image_description": "Example image",
                },
            ],
            "returned_count": 3,
            "news_quality_note": "Mixed result set: company-specific items found, with some contextual coverage retained.",
            "event_diversity_note": "Selected 3 items across 3 query categories.",
            "query_failures": [{"query_category": "management_commentary", "error": "timeout"}],
        },
    )
    monkeypatch.setattr(
        example8,
        "selective_extract_shortlisted_urls_tavily",
        lambda shortlisted_items, query="", max_urls=3: {
            "ok": True,
            "source": "Tavily selective extraction",
            "selected_count": 2,
            "selected_urls": [
                "https://www.reuters.com/business/microsoft-ai/",
                "https://www.bloomberg.com/news/microsoft-context/",
            ],
            "results": [
                {
                    "title": "Microsoft expands AI partnership",
                    "publisher": "Reuters",
                    "date": "2026-04-18T00:00:00+00:00",
                    "url": "https://www.reuters.com/business/microsoft-ai/",
                    "favicon": "https://www.reuters.com/favicon.ico",
                    "image": "https://www.reuters.com/image.jpg",
                    "image_description": "Reuters photo",
                    "query_category": "product_strategy",
                    "relevance_bucket": "high_confidence_company_specific",
                    "extracted": True,
                    "extraction_status": "success",
                    "source_type": "tavily_extraction",
                    "caution": None,
                    "ranking_score": 9.1,
                    "snippet": "Microsoft expanded an AI partnership.",
                    "content": "Microsoft expanded its partnership with a long-form explanation.",
                    "selected_rank": 1,
                    "extract_depth": "basic",
                    "extract_format": "markdown",
                }
            ],
            "failed_results": [
                {
                    "title": "Microsoft context coverage",
                    "publisher": "Bloomberg",
                    "date": "2026-04-17T00:00:00+00:00",
                    "url": "https://www.bloomberg.com/news/microsoft-context/",
                    "favicon": "https://www.bloomberg.com/favicon.ico",
                    "image": "https://www.bloomberg.com/image.jpg",
                    "image_description": "Bloomberg photo",
                    "query_category": "broad_company_news",
                    "relevance_bucket": "broader_context",
                    "extracted": False,
                    "extraction_status": "failed",
                    "source_type": "tavily_extraction",
                    "caution": "selective extraction failed; verify directly",
                    "ranking_score": 7.2,
                    "snippet": "A contextual market note.",
                    "extraction_error": "timeout",
                    "selected_rank": 2,
                    "extract_depth": "basic",
                    "extract_format": "markdown",
                }
            ],
        },
    )

    example8.run_demo(
        example8.DEMO_PROMPT,
        show_sources=True,
        show_weak_items=True,
        show_extractions=True,
    )

    output = capsys.readouterr().out
    assert output.startswith(final_text)
    assert output.index("SOURCE CARDS") < output.index("WEAK / LOW-CONFIDENCE ITEMS") < output.index("EXTRACTIONS") < output.index("QUERY FAILURES")
    assert "SOURCE CARDS" in output
    assert "WEAK / LOW-CONFIDENCE ITEMS" in output
    assert "EXTRACTIONS" in output
    assert "QUERY FAILURES" in output
    assert "[EXTRACTED]" in output
    assert "favicon: https://www.reuters.com/favicon.ico" in output
    assert "image: https://www.reuters.com/image.jpg" in output
    assert "Artifacts:" not in output


@pytest.mark.parametrize(
    "junk_description",
    [
        "",
        "...",
        "https://www.example.com/caption.jpg",
        "placeholder image",
        "image description unavailable",
    ],
)
def test_render_source_card_suppresses_junk_image_description(junk_description):
    rendered = example8._render_source_card(
        {
            "title": "Microsoft expands AI partnership",
            "publisher": "Reuters",
            "date": "2026-04-18T00:00:00+00:00",
            "url": "https://www.reuters.com/business/microsoft-ai/",
            "favicon": "https://www.reuters.com/favicon.ico",
            "image": "https://www.reuters.com/image.jpg",
            "image_description": junk_description,
            "query_category": "product_strategy",
            "relevance_bucket": "high_confidence_company_specific",
            "extracted": False,
            "extraction_status": "not_requested",
            "source_type": "tavily_news_search",
            "caution": None,
            "ranking_score": 9.1,
            "snippet": "Microsoft expanded an AI partnership.",
        }
    )

    assert "image_description:" not in rendered
    assert "favicon: https://www.reuters.com/favicon.ico" in rendered
    assert "image: https://www.reuters.com/image.jpg" in rendered


def test_run_demo_expanded_sections_suppress_auxiliary_noise(monkeypatch, capsys):
    evidence = _packet(example8.EVIDENCE_LABELS)
    research_read = _packet(example8.RESEARCH_READ_LABELS)
    open_questions = _packet(example8.OPEN_QUESTIONS_LABELS)
    final_text = _handoff(evidence, research_read, open_questions)

    class _NoisyTeam(_FakeTeam):
        def print_response(self, prompt: str, stream: bool = True, **kwargs) -> None:
            print("$MSFT: possibly delisted; no price data found")
            print("stderr noise from yfinance", file=sys.stderr)
            sys.stdout.write(self.output)

    noisy_team = _NoisyTeam(final_text)
    monkeypatch.setattr(example8, "build_team", lambda: noisy_team)

    def noisy_company_info(symbol: str):
        print(f"${symbol}: possibly delisted; no price data found")
        print("stderr noise from yfinance", file=sys.stderr)
        return {"symbol": symbol, "ok": False, "error": f"${symbol}: possibly delisted; no price data found"}

    monkeypatch.setattr(example8, "get_company_info", noisy_company_info)
    monkeypatch.setattr(
        example8,
        "get_company_news_tavily",
        lambda symbol, company_name="", num_stories=5: {
            "symbol": symbol,
            "ok": False,
            "news": [],
            "query_failures": [{"query_category": "management_commentary", "error": "timeout"}],
            "error": "tavily timeout",
        },
    )

    example8.run_demo(example8.DEMO_PROMPT, show_extractions=True)

    captured = capsys.readouterr()
    output = captured.out
    assert output.startswith(final_text)
    assert "$MSFT: possibly delisted; no price data found" not in output
    assert "stderr noise from yfinance" not in output
    assert captured.err == ""
    assert "EXTRACTIONS" in output
    assert "QUERY FAILURES" in output
    assert "query unavailable" in output
    assert "INSPECTION WARNINGS" in output
    assert "company_info: company info unavailable" in output
    assert "news_search: news search unavailable" in output


def test_write_run_artifacts_creates_sidecar_files(tmp_path):
    evidence = _packet(example8.EVIDENCE_LABELS)
    research_read = _packet(example8.RESEARCH_READ_LABELS)
    open_questions = _packet(example8.OPEN_QUESTIONS_LABELS)
    final_text = _handoff(evidence, research_read, open_questions)

    fake_team = _FakeTeam(final_text)
    source_inspection = {
        "symbol": "MSFT",
        "company_name": "Microsoft Corporation",
        "company_info_packet": {
            "ok": True,
            "company_info": {"longName": "Microsoft Corporation"},
            "source": "yfinance info",
        },
        "news_packet": {
            "ok": True,
            "news_quality_note": "Mixed result set: company-specific items found, with some contextual coverage retained.",
            "event_diversity_note": "Selected 2 items across 2 query categories.",
        },
        "source_cards": [
            {
                "title": "Microsoft expands AI partnership",
                "publisher": "Reuters",
                "date": "2026-04-18T00:00:00+00:00",
                "url": "https://www.reuters.com/business/microsoft-ai/",
                "favicon": "https://www.reuters.com/favicon.ico",
                "image": "https://www.reuters.com/image.jpg",
                "image_description": "Reuters photo",
                "query_category": "product_strategy",
                "relevance_bucket": "high_confidence_company_specific",
                "extracted": True,
                "extraction_status": "success",
                "source_type": "tavily_news_search",
                "caution": None,
                "ranking_score": 9.1,
                "snippet": "Microsoft expanded an AI partnership.",
            },
            {
                "title": "Microsoft context coverage",
                "publisher": "Bloomberg",
                "date": "2026-04-17T00:00:00+00:00",
                "url": "https://www.bloomberg.com/news/microsoft-context/",
                "favicon": "https://www.bloomberg.com/favicon.ico",
                "image": "https://www.bloomberg.com/image.jpg",
                "image_description": "Bloomberg photo",
                "query_category": "broad_company_news",
                "relevance_bucket": "broader_context",
                "extracted": False,
                "extraction_status": "not_requested",
                "source_type": "tavily_news_search",
                "caution": "contextual coverage; not a confirmed company-specific catalyst",
                "ranking_score": 7.2,
                "snippet": "A contextual market note.",
            },
        ],
        "weak_or_low_confidence_items": [],
        "query_failures": [{"query_category": "management_commentary", "error": "timeout"}],
        "selected_extraction_records": [
            {
                "title": "Microsoft expands AI partnership",
                "publisher": "Reuters",
                "date": "2026-04-18T00:00:00+00:00",
                "url": "https://www.reuters.com/business/microsoft-ai/",
                "favicon": "https://www.reuters.com/favicon.ico",
                "image": "https://www.reuters.com/image.jpg",
                "image_description": "Reuters photo",
                "query_category": "product_strategy",
                "relevance_bucket": "high_confidence_company_specific",
                "extracted": True,
                "extraction_status": "success",
                "source_type": "tavily_extraction",
                "caution": None,
                "ranking_score": 9.1,
                "snippet": "Microsoft expanded an AI partnership.",
                "content": "Microsoft expanded its partnership with a long-form explanation.",
                "selected_rank": 1,
            }
        ],
        "extraction_results": {
            "ok": True,
            "source": "Tavily selective extraction",
            "selected_count": 1,
            "selected_urls": ["https://www.reuters.com/business/microsoft-ai/"],
            "results": [
                {
                    "title": "Microsoft expands AI partnership",
                    "publisher": "Reuters",
                    "date": "2026-04-18T00:00:00+00:00",
                    "url": "https://www.reuters.com/business/microsoft-ai/",
                    "favicon": "https://www.reuters.com/favicon.ico",
                    "image": "https://www.reuters.com/image.jpg",
                    "image_description": "Reuters photo",
                    "query_category": "product_strategy",
                    "relevance_bucket": "high_confidence_company_specific",
                    "extracted": True,
                    "extraction_status": "success",
                    "source_type": "tavily_extraction",
                    "caution": None,
                    "ranking_score": 9.1,
                    "snippet": "Microsoft expanded an AI partnership.",
                    "content": "Microsoft expanded its partnership with a long-form explanation.",
                    "selected_rank": 1,
                    "extract_depth": "basic",
                    "extract_format": "markdown",
                }
            ],
            "failed_results": [],
        },
        "extraction_triggered": True,
        "selective_extraction_needed": True,
    }

    source_inspection["weak_or_low_confidence_items"] = [source_inspection["source_cards"][1]]

    run_dir = example8._write_run_artifacts(fake_team, example8.DEMO_PROMPT, source_inspection, artifact_root=tmp_path)

    assert run_dir is not None
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "source_cards.json").exists()
    assert (run_dir / "extraction_results.json").exists()

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    source_cards_artifact = json.loads((run_dir / "source_cards.json").read_text(encoding="utf-8"))
    extraction_artifact = json.loads((run_dir / "extraction_results.json").read_text(encoding="utf-8"))

    assert manifest["prompt"] == example8.DEMO_PROMPT
    assert manifest["validation_results"]["ok"] is True
    assert manifest["artifacts"]["source_cards"] == "source_cards.json"
    assert source_cards_artifact["symbol"] == "MSFT"
    assert len(source_cards_artifact["source_cards"]) == 2
    assert len(source_cards_artifact["weak_or_low_confidence_items"]) == 1
    assert len(source_cards_artifact["selected_extraction_records"]) == 1
    assert source_cards_artifact["query_failures"][0]["query_category"] == "management_commentary"
    assert extraction_artifact["results"][0]["content"].startswith("Microsoft expanded")
    assert extraction_artifact["results"][0]["favicon"] == "https://www.reuters.com/favicon.ico"
    assert extraction_artifact["failed_results"] == []
    assert {
        "title",
        "publisher",
        "date",
        "url",
        "favicon",
        "image",
        "image_description",
        "query_category",
        "relevance_bucket",
        "extracted",
        "extraction_status",
        "source_type",
        "caution",
        "ranking_score",
    }.issubset(source_cards_artifact["source_cards"][0].keys())
