from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from core.adapters.fred_adapter import FredMacroAdapter
from core.adapters.oanda_adapter import OandaMarketDataAdapter
from core.domain.contracts import DataProvenance, ResearchPacket
from core.services.feature_builder import DeterministicFeatureBuilder
from core.services.regime_engine import DeterministicRegimeEngine


def _extract_rule_id_from_rationale(rationale: str) -> str:
    """Extract the leading ``rule_id`` token from a rationale string."""
    if not isinstance(rationale, str):
        raise ValueError("decision rationale must be a string.")

    prefix = "rule_id="
    if not rationale.startswith(prefix):
        raise ValueError("decision rationale must start with 'rule_id='.")

    first_segment = rationale.split(";", 1)[0]
    rule_id = first_segment[len(prefix) :]
    if not rule_id:
        raise ValueError("decision rationale is missing a rule_id value.")
    return rule_id


def run_macro_slice(instrument: str, start: datetime, end: datetime) -> ResearchPacket:
    """Run a deterministic macro regime slice for a single instrument and date window."""
    if not instrument:
        raise ValueError("instrument is required.")
    if start is None or end is None:
        raise ValueError("start and end datetimes are required.")
    if end < start:
        raise ValueError("end must be greater than or equal to start.")
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware UTC datetimes.")
    if start.tzinfo.utcoffset(start) != timedelta(0) or end.tzinfo.utcoffset(end) != timedelta(0):
        raise ValueError("start and end must be in UTC.")

    market = OandaMarketDataAdapter()
    macro = FredMacroAdapter()
    fb = DeterministicFeatureBuilder(market, macro)
    eng = DeterministicRegimeEngine()

    features = fb.build_features(instrument, start, end)
    decided_at = end
    decision = eng.decide(features, decided_at=decided_at)

    rule_id = _extract_rule_id_from_rationale(decision.rationale)
    packet_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    packet = ResearchPacket(
        packet_id=packet_id,
        as_of=decided_at,
        created_at=created_at,
        engine_version="v0",
        ruleset_version="v0",
        feature_values=features,
        regime=decision,
        input_provenance=[
            DataProvenance(
                source="OANDA",
                timestamp=decided_at,
                notes="tpqoa candles",
            ),
            DataProvenance(
                source="FRED:DGS10",
                timestamp=decided_at,
                notes="fredapi series",
            ),
        ],
        rule_trace=[rule_id],
    )
    return packet


if __name__ == "__main__":
    instrument = "SPX500_USD"
    start = datetime(2024, 1, 1)
    end = datetime(2026, 2, 17)
    packet = run_macro_slice(instrument=instrument, start=start, end=end)
    print(packet.model_dump())
