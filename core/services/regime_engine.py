from __future__ import annotations

import math
from datetime import datetime
from numbers import Real
from types import MappingProxyType
from typing import Dict

from core.domain.contracts import Regime, RegimeDecision


class DeterministicRegimeEngine:
    """Deterministic v0 regime classifier using fixed threshold rules.

    The implementation is intentionally deterministic:
    - no model calls
    - no side effects
    - no implicit timestamps
    """

    _DEFAULT_THRESHOLDS = MappingProxyType(
        {
            "risk_off_return": -0.03,
            "risk_off_dy_change": 0.20,
            "risk_on_return": 0.02,
            "risk_on_dy_change": 0.15,
        }
    )

    def __init__(self, thresholds: Dict[str, float] | None = None) -> None:
        """Initialize rule thresholds.

        Args:
            thresholds: Optional partial override of default thresholds.

        Raises:
            ValueError: If thresholds include unknown keys or invalid values.
        """
        self._thresholds = dict(self._DEFAULT_THRESHOLDS)

        if thresholds is None:
            return
        if not isinstance(thresholds, dict):
            raise ValueError("thresholds must be a dictionary when provided.")

        unknown = set(thresholds) - set(self._DEFAULT_THRESHOLDS)
        if unknown:
            unknown_list = ",".join(sorted(unknown))
            raise ValueError(f"Unknown threshold keys: {unknown_list}.")

        for key, raw_value in thresholds.items():
            self._thresholds[key] = self._validate_numeric(raw_value, key)

    def decide(self, features: Dict[str, float], decided_at: datetime) -> RegimeDecision:
        """Return a deterministic regime decision from feature values.

        Args:
            features: Feature map containing:
                ``spx_20d_return``, ``spx_above_200dma``,
                ``dgs10_level``, ``dgs10_20d_change``.
            decided_at: Explicit timestamp for this decision.

        Returns:
            RegimeDecision with regime, confidence, rationale, and decided_at.

        Raises:
            ValueError: If required inputs are missing, non-numeric, or NaN.
        """
        if not isinstance(features, dict):
            raise ValueError("features must be a dictionary.")
        if decided_at is None or not isinstance(decided_at, datetime):
            raise ValueError("decided_at must be an explicit datetime.")

        required_keys = {
            "spx_20d_return",
            "spx_above_200dma",
            "dgs10_level",
            "dgs10_20d_change",
        }
        missing = required_keys - set(features)
        if missing:
            missing_list = ",".join(sorted(missing))
            raise ValueError(f"Missing required feature keys: {missing_list}.")

        r = self._validate_numeric(features["spx_20d_return"], "spx_20d_return")
        ma = self._validate_numeric(features["spx_above_200dma"], "spx_above_200dma")
        y = self._validate_numeric(features["dgs10_level"], "dgs10_level")
        dy = self._validate_numeric(features["dgs10_20d_change"], "dgs10_20d_change")

        if ma not in (0.0, 1.0):
            raise ValueError("spx_above_200dma must be exactly 0.0 or 1.0.")

        off_r = self._thresholds["risk_off_return"]
        off_dy = self._thresholds["risk_off_dy_change"]
        on_r = self._thresholds["risk_on_return"]
        on_dy = self._thresholds["risk_on_dy_change"]

        off_ma = ma == 0.0
        off_ret = r < off_r
        off_dy_cmp = dy > off_dy
        is_risk_off = off_ma and off_ret and off_dy_cmp

        on_ma = ma == 1.0
        on_ret = r > on_r
        on_dy_cmp = dy <= on_dy
        is_risk_on = on_ma and on_ret and on_dy_cmp

        if is_risk_off:
            regime = Regime.RISK_OFF
            rule_id = "RISK_OFF_V0"
            confidence = self._risk_confidence(
                return_margin=off_r - r,
                dy_margin=dy - off_dy,
            )
        elif is_risk_on:
            regime = Regime.RISK_ON
            rule_id = "RISK_ON_V0"
            confidence = self._risk_confidence(
                return_margin=r - on_r,
                dy_margin=on_dy - dy,
            )
        else:
            regime = Regime.NEUTRAL
            rule_id = "NEUTRAL_V0"
            off_score = float(sum((off_ma, off_ret, off_dy_cmp))) / 3.0
            on_score = float(sum((on_ma, on_ret, on_dy_cmp))) / 3.0
            confidence = self._clamp01(0.50 + (1.0 - max(off_score, on_score)) * 0.50)

        rationale = (
            f"rule_id={rule_id};"
            f"r={r:.6f};ma={ma:.1f};y={y:.6f};dy={dy:.6f};"
            f"off(ma==0.0)={str(off_ma).lower()},"
            f"(r<{off_r:.6f})={str(off_ret).lower()},"
            f"(dy>{off_dy:.6f})={str(off_dy_cmp).lower()};"
            f"on(ma==1.0)={str(on_ma).lower()},"
            f"(r>{on_r:.6f})={str(on_ret).lower()},"
            f"(dy<={on_dy:.6f})={str(on_dy_cmp).lower()}"
        )

        return RegimeDecision(
            regime=regime,
            confidence=confidence,
            rationale=rationale,
            decided_at=decided_at,
        )

    @staticmethod
    def _validate_numeric(value: float, name: str) -> float:
        """Validate feature/threshold values as finite real numbers."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a float.")
        value_float = float(value)
        if not math.isfinite(value_float):
            raise ValueError(f"{name} must be finite and non-NaN.")
        return value_float

    @staticmethod
    def _risk_confidence(return_margin: float, dy_margin: float) -> float:
        """Map satisfied rule margins to a deterministic confidence in [0, 1]."""
        return_component = min(max(return_margin, 0.0) / 0.05, 1.0)
        dy_component = min(max(dy_margin, 0.0) / 0.30, 1.0)
        return DeterministicRegimeEngine._clamp01(
            0.70 + (0.15 * return_component) + (0.15 * dy_component)
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        """Clamp float to the closed interval [0.0, 1.0]."""
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
