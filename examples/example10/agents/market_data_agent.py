from __future__ import annotations

try:
    from ..schemas import MarketSnapshot, ResearchBrief
except ImportError:  # pragma: no cover
    from schemas import MarketSnapshot, ResearchBrief


class MarketDataAgent:
    def analyze(self, brief: ResearchBrief, snapshot: MarketSnapshot) -> dict:
        if not snapshot.prices:
            return {
                "summary": "Market snapshot unavailable.",
                "trend": "unknown",
                "leaders": [],
                "laggards": [],
                "high_volatility": [],
                "trend_score": 0.0,
            }

        returns = snapshot.returns_20d or {ticker: 0.0 for ticker in snapshot.tickers}
        avg_return = sum(returns.values()) / max(len(returns), 1)

        trend = "sideways"
        if avg_return > 0.02:
            trend = "uptrend"
        elif avg_return < -0.02:
            trend = "downtrend"

        ranked = sorted(returns.items(), key=lambda pair: pair[1], reverse=True)
        leaders = [f"{ticker}: {value:.2%}" for ticker, value in ranked[:2]]
        laggards = [f"{ticker}: {value:.2%}" for ticker, value in ranked[-2:]]

        high_volatility = [
            ticker for ticker, value in snapshot.vol_20d.items() if value >= 0.30
        ]

        return {
            "summary": f"Average 20-day return is {avg_return:.2%} ({trend}).",
            "trend": trend,
            "leaders": leaders,
            "laggards": laggards,
            "high_volatility": high_volatility,
            "trend_score": round(avg_return, 3),
            "notes": snapshot.notes,
        }
