from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -------------------------
# Shared primitives
# -------------------------

class AssetClass(str, Enum):
    FX = "FX"
    EQUITY_INDEX = "EQUITY_INDEX"
    COMMODITY = "COMMODITY"
    RATES = "RATES"
    CRYPTO = "CRYPTO"
    OTHER = "OTHER"


class Horizon(str, Enum):
    INTRADAY = "INTRADAY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class Regime(str, Enum):
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"


class Stance(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


# -------------------------
# Core domain contracts
# -------------------------

class DataProvenance(BaseModel):
    source: str
    timestamp: datetime
    notes: Optional[str] = None


class MarketSnapshot(BaseModel):
    instrument: str
    asset_class: AssetClass
    price: float
    timestamp: datetime
    provenance: DataProvenance


class RegimeDecision(BaseModel):
    regime: Regime
    confidence: float
    rationale: str
    decided_at: datetime


class AllocationPlan(BaseModel):
    instrument: str
    stance: Stance
    weight: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class ResearchPacket(BaseModel):
    packet_id: str
    as_of: datetime
    created_at: datetime
    engine_version: str
    ruleset_version: str
    input_provenance: List[DataProvenance] = Field(default_factory=list)
    feature_values: Dict[str, float] = Field(default_factory=dict)
    rule_trace: List[str] = Field(default_factory=list)
    snapshot: Optional[MarketSnapshot] = None
    regime: Optional[RegimeDecision] = None
    allocation: Optional[AllocationPlan] = None
    debug: Dict[str, Any] = Field(default_factory=dict)
