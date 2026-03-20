from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ResearchBrief:
    query: str
    objective: str
    topics: list[str]
    tickers: list[str]
    macro_indicators: list[str]
    timeframe: str = "3-12 months"
    constraints: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    source: str
    title: str
    summary: str
    url: str = ""
    timestamp: str = ""
    relevance: float = 0.5


@dataclass
class MarketSnapshot:
    tickers: list[str]
    as_of: str
    prices: dict[str, float]
    returns_20d: dict[str, float]
    vol_20d: dict[str, float]
    notes: list[str] = field(default_factory=list)


@dataclass
class MacroState:
    as_of: str
    indicators: dict[str, float]
    regime: str
    notes: list[str] = field(default_factory=list)


@dataclass
class TacticalView:
    signal: str
    conviction: float
    horizon: str
    risks: list[str] = field(default_factory=list)
    preferred_exposures: list[str] = field(default_factory=list)
    avoid_exposures: list[str] = field(default_factory=list)
    stance_basis: list[str] = field(default_factory=list)

    @property
    def thesis(self) -> list[str]:
        return self.stance_basis

    @property
    def allocations(self) -> dict[str, float]:
        return {}


PortfolioView = TacticalView


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    query: str
    stance_signal: str
    conviction: float
    notes: str = ""

    @property
    def portfolio_signal(self) -> str:
        return self.stance_signal


@dataclass
class ResearchPacket:
    run_id: str
    created_at: str
    query: str
    brief: ResearchBrief
    evidence: list[Evidence]
    web_view: dict[str, Any]
    market_snapshot: MarketSnapshot
    market_view: dict[str, Any]
    macro_state: MacroState
    macro_view: dict[str, Any]
    history: list[RunRecord]
    tactical_view: TacticalView
    tactical_state: dict[str, Any] = field(default_factory=dict)
    layer: str = "layer1-evidence-state"
    packet_kind: str = "tactical-view-packet"
    actionability: str = "tactical"
    report: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def portfolio_view(self) -> TacticalView:
        return self.tactical_view
