from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional

class ConcentrationRisk(BaseModel):
    top_position_pct: float
    top_3_positions_pct: float
    flag: Literal["low","medium","high"]
    top_holding: Optional[str] = None

class Performance(BaseModel):
    total_cost: float
    total_value: float
    total_return_pct: float
    annualized_return_pct: Optional[float] = None
    total_gain_loss: float

class BenchmarkComparison(BaseModel):
    benchmark: str
    benchmark_ticker: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    outperforming: bool

class Observation(BaseModel):
    severity: Literal["info","warning","critical"]
    text: str

class HealthCheckResult(BaseModel):
    concentration_risk: ConcentrationRisk
    performance: Performance
    benchmark_comparison: Optional[BenchmarkComparison] = None
    observations: list[Observation] = Field(default_factory=list)
    disclaimer: str
    raw_summary: str
    is_empty_portfolio: bool = False
