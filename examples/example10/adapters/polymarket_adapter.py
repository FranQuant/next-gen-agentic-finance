from __future__ import annotations

from datetime import datetime, timezone

try:
    from ..schemas import SentimentSnapshot
except ImportError:  # pragma: no cover
    from schemas import SentimentSnapshot


class PolymarketAdapter:
    def get_sentiment(self, topic: str) -> SentimentSnapshot:
        return SentimentSnapshot(
            topic=topic,
            as_of=datetime.now(timezone.utc).isoformat(),
            bullish_prob=None,
            bearish_prob=None,
            confidence=0.1,
            notes=[
                "Polymarket adapter is a stub in Example10.",
                "No live prediction market feed configured.",
            ],
        )
