from __future__ import annotations

import math
from datetime import datetime, timezone

try:
    from ..schemas import MarketSnapshot
except ImportError:  # pragma: no cover
    from schemas import MarketSnapshot


class MarketAdapter:
    def __init__(self) -> None:
        try:
            import yfinance as yf

            self._yf = yf
        except Exception:
            self._yf = None

    def get_snapshot(self, tickers: list[str], lookback_days: int = 60) -> MarketSnapshot:
        if not tickers:
            return MarketSnapshot(
                tickers=[],
                as_of=self._now_iso(),
                prices={},
                returns_20d={},
                vol_20d={},
                notes=["No tickers provided."],
            )

        prices: dict[str, float] = {}
        returns_20d: dict[str, float] = {}
        vol_20d: dict[str, float] = {}
        notes: list[str] = []

        for ticker in tickers:
            ticker_upper = ticker.upper()
            if not self._yf:
                fallback_price = self._fallback_price(ticker_upper)
                prices[ticker_upper] = fallback_price
                returns_20d[ticker_upper] = 0.0
                vol_20d[ticker_upper] = 0.2
                notes.append("yfinance unavailable; using deterministic placeholders.")
                continue

            try:
                history = self._yf.Ticker(ticker_upper).history(
                    period=f"{max(lookback_days, 25)}d",
                    auto_adjust=True,
                )
                close = history["Close"].dropna()
                if close.empty:
                    raise ValueError("No close prices.")

                prices[ticker_upper] = float(close.iloc[-1])

                if len(close) > 20:
                    returns_20d[ticker_upper] = float(close.iloc[-1] / close.iloc[-21] - 1.0)
                else:
                    returns_20d[ticker_upper] = 0.0

                daily = close.pct_change().dropna().tail(20)
                if daily.empty:
                    vol_20d[ticker_upper] = 0.0
                else:
                    vol_20d[ticker_upper] = float(daily.std() * math.sqrt(252))
            except Exception as exc:
                fallback_price = self._fallback_price(ticker_upper)
                prices[ticker_upper] = fallback_price
                returns_20d[ticker_upper] = 0.0
                vol_20d[ticker_upper] = 0.2
                notes.append(f"{ticker_upper}: market fetch failed ({exc}); placeholder used.")

        return MarketSnapshot(
            tickers=[ticker.upper() for ticker in tickers],
            as_of=self._now_iso(),
            prices=prices,
            returns_20d=returns_20d,
            vol_20d=vol_20d,
            notes=notes,
        )

    def _fallback_price(self, ticker: str) -> float:
        seed = sum(ord(char) for char in ticker)
        return round(80.0 + (seed % 70), 2)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
