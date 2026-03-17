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
        self.last_fetch_report = self._new_fetch_report()

    def get_snapshot(self, tickers: list[str], lookback_days: int = 60) -> MarketSnapshot:
        report = self._new_fetch_report()
        if not tickers:
            report["mode"] = "market-empty"
            self.last_fetch_report = report
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

        normalized_tickers = [ticker.upper() for ticker in tickers]
        for normalized in normalized_tickers:
            if not self._yf:
                prices[normalized] = self._fallback_price(normalized)
                returns_20d[normalized] = 0.0
                vol_20d[normalized] = 0.2
                report["fallback_used"] = True
                self._append_ticker(report["placeholder_tickers"], normalized)
                self._append_note(notes, "yfinance unavailable; using deterministic placeholders.")
                continue

            try:
                history = self._yf.Ticker(normalized).history(
                    period=f"{max(lookback_days, 25)}d",
                    auto_adjust=True,
                )
                close = history["Close"].dropna()
                if close.empty:
                    raise ValueError("No close prices.")

                prices[normalized] = float(close.iloc[-1])
                returns_20d[normalized] = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) > 20 else 0.0

                daily = close.pct_change().dropna().tail(20)
                vol_20d[normalized] = float(daily.std() * math.sqrt(252)) if not daily.empty else 0.0
            except Exception as exc:
                prices[normalized] = self._fallback_price(normalized)
                returns_20d[normalized] = 0.0
                vol_20d[normalized] = 0.2
                report["fallback_used"] = True
                self._append_ticker(report["failed_tickers"], normalized)
                self._append_ticker(report["placeholder_tickers"], normalized)
                notes.append(f"{normalized}: market fetch failed ({exc}); placeholder used.")

        placeholder_count = len(report["placeholder_tickers"])
        if placeholder_count == len(normalized_tickers):
            report["mode"] = "market-placeholder"
        elif placeholder_count:
            report["mode"] = "market-partial-fallback"
        self.last_fetch_report = report

        return MarketSnapshot(
            tickers=normalized_tickers,
            as_of=self._now_iso(),
            prices=prices,
            returns_20d=returns_20d,
            vol_20d=vol_20d,
            notes=notes,
        )

    def _fallback_price(self, ticker: str) -> float:
        seed = sum(ord(char) for char in ticker)
        return round(80.0 + (seed % 70), 2)

    def _append_note(self, notes: list[str], note: str) -> None:
        if note not in notes:
            notes.append(note)

    def _append_ticker(self, tickers: list[str], ticker: str) -> None:
        if ticker not in tickers:
            tickers.append(ticker)

    def _new_fetch_report(self) -> dict[str, bool | str | list[str]]:
        return {
            "fallback_used": False,
            "mode": "market-live",
            "failed_tickers": [],
            "placeholder_tickers": [],
        }

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
