import importlib
import sys
import types
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


@pytest.mark.parametrize(
    ("flag_name", "dependency_name", "expected_message"),
    [
        ("ENABLE_DB", "SqliteDb", "SQLite/SQLAlchemy support is unavailable"),
        ("ENABLE_REASONING", "ReasoningTools", "Reasoning-tool support is unavailable"),
    ],
)
def test_build_team_raises_when_optional_feature_dependency_is_missing(
    monkeypatch,
    tmp_path,
    flag_name,
    dependency_name,
    expected_message,
):
    openai_stub = types.ModuleType("agno.models.openai")
    tavily_stub = types.ModuleType("tavily")
    yfinance_stub = types.ModuleType("yfinance")

    class _DummyOpenAIResponses:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyTavilyClient:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyTicker:
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.OpenAIResponses = _DummyOpenAIResponses
    tavily_stub.TavilyClient = _DummyTavilyClient
    yfinance_stub.Ticker = _DummyTicker
    monkeypatch.setitem(sys.modules, "agno.models.openai", openai_stub)
    monkeypatch.setitem(sys.modules, "tavily", tavily_stub)
    monkeypatch.setitem(sys.modules, "yfinance", yfinance_stub)
    monkeypatch.delitem(sys.modules, "example7", raising=False)
    example7 = importlib.import_module("example7")

    monkeypatch.setattr(example7, "TEAM_DB_PATH", tmp_path / "example7_team.db")
    monkeypatch.setattr(example7, "ENABLE_DB", False)
    monkeypatch.setattr(example7, "ENABLE_REASONING", False)
    monkeypatch.setattr(example7, flag_name, True)
    monkeypatch.setattr(example7, dependency_name, None)

    with pytest.raises(RuntimeError, match=expected_message):
        example7.build_team()
