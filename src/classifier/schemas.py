from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

VALID_AGENTS = ["portfolio_health","market_research","investment_strategy","financial_calculator","risk_assessment","recommendations","predictive_analysis","general_support"]

class ExtractedEntities(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    amounts: list[float] = Field(default_factory=list)
    time_periods: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

class ClassificationResult(BaseModel):
    intent: str
    confidence: float
    agent: str
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    safety_verdict: str = "clean"
    safety_note: Optional[str] = None
    resolved_query: str

FALLBACK_CLASSIFICATION = ClassificationResult(
    intent="general_support", confidence=0.1, agent="general_support",
    entities=ExtractedEntities(), safety_verdict="clean",
    safety_note="Classification failed — fallback applied", resolved_query=""
)
