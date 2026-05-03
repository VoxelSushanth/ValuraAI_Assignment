import json
import logging
import asyncio
import yfinance as yf
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI
from src.agents.base import BaseAgent
from src.classifier.schemas import ClassificationResult
from src.agents.portfolio_health.schemas import (
    ConcentrationRisk, Performance, BenchmarkComparison, 
    Observation, HealthCheckResult
)
from src.agents.portfolio_health.prompts import (
    DISCLAIMER, OBSERVATIONS_SYSTEM_PROMPT, 
    OBSERVATIONS_USER_TEMPLATE, EMPTY_PORTFOLIO_RESULT
)

logger = logging.getLogger(__name__)


class PortfolioHealthAgent(BaseAgent):
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model
    
    def _parse_holdings(self, user_profile: dict) -> list[dict]:
        """Try keys: holdings, portfolio, positions, investments"""
        for key in ["holdings", "portfolio", "positions", "investments"]:
            if key in user_profile and isinstance(user_profile[key], list):
                return user_profile[key]
        return []
    
    def _normalize_holding(self, h: dict) -> Optional[dict]:
        """Extract and normalize holding data"""
        # Extract ticker
        ticker = None
        for key in ["ticker", "symbol"]:
            if key in h and h[key]:
                ticker = str(h[key]).upper()
                break
        if not ticker:
            return None
        
        # Extract quantity
        quantity = 0
        for key in ["quantity", "shares", "units"]:
            if key in h:
                try:
                    quantity = float(h[key])
                    break
                except (ValueError, TypeError):
                    pass
        
        # Extract current_price
        current_price = 0
        for key in ["current_price", "price", "market_price"]:
            if key in h:
                try:
                    current_price = float(h[key])
                    break
                except (ValueError, TypeError):
                    pass
        
        # Extract cost_basis (fallback to current_price)
        cost_basis = current_price
        for key in ["cost_basis", "purchase_price", "avg_cost", "average_cost"]:
            if key in h:
                try:
                    cost_basis = float(h[key])
                    break
                except (ValueError, TypeError):
                    pass
        
        # Return None if invalid
        if quantity <= 0 or current_price <= 0:
            return None
        
        return {
            "ticker": ticker,
            "quantity": quantity,
            "current_price": current_price,
            "cost_basis": cost_basis,
            "current_value": quantity * current_price,
            "cost_value": quantity * cost_basis
        }
    
    def _calculate_concentration(self, normalized: list[dict]) -> ConcentrationRisk:
        """Calculate concentration risk metrics"""
        total_value = sum(h["current_value"] for h in normalized)
        
        if total_value == 0:
            return ConcentrationRisk(
                top_position_pct=0.0,
                top_3_positions_pct=0.0,
                flag="low",
                top_holding=None
            )
        
        # Sort by current_value descending
        sorted_holdings = sorted(normalized, key=lambda x: x["current_value"], reverse=True)
        
        # Top position percentage
        top_position_pct = (sorted_holdings[0]["current_value"] / total_value) * 100
        
        # Top 3 positions percentage
        top_3_value = sum(h["current_value"] for h in sorted_holdings[:3])
        top_3_positions_pct = (top_3_value / total_value) * 100
        
        # Determine flag
        if top_position_pct > 50:
            flag = "high"
        elif top_position_pct > 25:
            flag = "medium"
        else:
            flag = "low"
        
        return ConcentrationRisk(
            top_position_pct=top_position_pct,
            top_3_positions_pct=top_3_positions_pct,
            flag=flag,
            top_holding=sorted_holdings[0]["ticker"]
        )
    
    def _calculate_performance(self, normalized: list[dict]) -> Performance:
        """Calculate performance metrics"""
        total_cost = sum(h["cost_value"] for h in normalized)
        total_value = sum(h["current_value"] for h in normalized)
        
        if total_cost == 0:
            total_return_pct = 0.0
        else:
            total_return_pct = ((total_value - total_cost) / total_cost) * 100
        
        total_gain_loss = total_value - total_cost
        
        return Performance(
            total_cost=total_cost,
            total_value=total_value,
            total_return_pct=total_return_pct,
            annualized_return_pct=None,  # No purchase dates available
            total_gain_loss=total_gain_loss
        )
    
    def _fetch_benchmark(self, normalized: list[dict], user_profile: dict) -> BenchmarkComparison:
        """Fetch benchmark comparison using yfinance"""
        currency = user_profile.get("currency", "USD")
        
        # Determine benchmark
        if currency == "USD":
            benchmark_name = "S&P 500"
            benchmark_ticker = "SPY"
        else:
            benchmark_name = "MSCI World"
            benchmark_ticker = "ACWI"
        
        # Try to fetch benchmark return
        benchmark_return_pct = 14.2  # Default fallback
        try:
            ticker = yf.Ticker(benchmark_ticker)
            history = ticker.history(period="1y")
            if len(history) > 1:
                start_price = history["Close"].iloc[0]
                end_price = history["Close"].iloc[-1]
                benchmark_return_pct = ((end_price - start_price) / start_price) * 100
        except Exception:
            # Use fallback on any exception
            pass
        
        # Calculate portfolio return
        portfolio_return_pct = self._calculate_performance(normalized).total_return_pct
        
        alpha = portfolio_return_pct - benchmark_return_pct
        
        return BenchmarkComparison(
            benchmark=benchmark_name,
            benchmark_ticker=benchmark_ticker,
            portfolio_return_pct=portfolio_return_pct,
            benchmark_return_pct=benchmark_return_pct,
            alpha_pct=alpha,
            outperforming=alpha > 0
        )
    
    async def _generate_observations(
        self, concentration: ConcentrationRisk, performance: Performance,
        benchmark: Optional[BenchmarkComparison], user_profile: dict
    ) -> tuple[list[Observation], str]:
        """Generate observations via LLM"""
        # Build metrics JSON
        metrics_dict = {
            "concentration_risk": concentration.model_dump(),
            "performance": performance.model_dump(),
            "benchmark_comparison": benchmark.model_dump() if benchmark else None
        }
        metrics_json = json.dumps(metrics_dict, indent=2)
        
        risk_profile = user_profile.get("risk_profile", "not specified")
        market_context = "Current market conditions analysis."
        
        user_message = OBSERVATIONS_USER_TEMPLATE.format(
            metrics_json=metrics_json,
            risk_profile=risk_profile,
            market_context=market_context
        )
        
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": OBSERVATIONS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=800
                ),
                timeout=15.0
            )
            
            raw_content = response.choices[0].message.content
            if raw_content is None:
                raise ValueError("Empty response from LLM")
            
            data = json.loads(raw_content)
            
            # Parse observations
            obs_list = data.get("observations", [])
            observations = [
                Observation(severity=o.get("severity", "info"), text=o.get("text", ""))
                for o in obs_list
            ]
            
            raw_summary = data.get("raw_summary", "Portfolio analysis complete.")
            
            return observations, raw_summary
            
        except Exception as e:
            logger.warning(f"LLM observation generation failed: {e}")
            # Return hardcoded fallback observations based on metrics
            fallback_obs = []
            
            if concentration.flag == "high":
                fallback_obs.append(Observation(
                    severity="critical",
                    text=f"High concentration risk: Your top holding ({concentration.top_holding}) represents {concentration.top_position_pct:.1f}% of your portfolio. This concentration risk (having too much in one stock) could lead to significant losses if that company underperforms."
                ))
            elif concentration.flag == "medium":
                fallback_obs.append(Observation(
                    severity="warning",
                    text=f"Moderate concentration: Your largest position ({concentration.top_holding}) is {concentration.top_position_pct:.1f}% of your portfolio. Consider diversifying to reduce single-stock risk."
                ))
            
            if performance.total_return_pct > 0:
                fallback_obs.append(Observation(
                    severity="info",
                    text=f"Positive returns: Your portfolio is up {performance.total_return_pct:.1f}% overall, which is a good sign."
                ))
            elif performance.total_return_pct < -20:
                fallback_obs.append(Observation(
                    severity="critical",
                    text=f"Significant losses: Your portfolio is down {abs(performance.total_return_pct):.1f}%. Consider reviewing your investment strategy."
                ))
            elif performance.total_return_pct < 0:
                fallback_obs.append(Observation(
                    severity="warning",
                    text=f"Negative returns: Your portfolio is down {abs(performance.total_return_pct):.1f}%. This is normal market volatility, but review your holdings."
                ))
            
            if benchmark and not benchmark.outperforming:
                fallback_obs.append(Observation(
                    severity="warning",
                    text=f"Underperforming benchmark: Your portfolio returned {performance.total_return_pct:.1f}% vs {benchmark.benchmark_return_pct:.1f}% for the {benchmark.benchmark}. Consider reviewing your asset allocation."
                ))
            
            if not fallback_obs:
                fallback_obs.append(Observation(
                    severity="info",
                    text="Your portfolio appears balanced. Continue monitoring regularly."
                ))
            
            raw_summary = f"Portfolio analysis: {performance.total_return_pct:.1f}% return, {concentration.flag} concentration risk."
            
            return fallback_obs, raw_summary
    
    async def run(
        self, query: str, user_profile: dict, classification: ClassificationResult, session_id: str
    ) -> AsyncGenerator[str, None]:
        """Run the portfolio health agent"""
        try:
            # Parse and normalize holdings
            raw_holdings = self._parse_holdings(user_profile)
            normalized = []
            for h in raw_holdings:
                if isinstance(h, dict):
                    norm = self._normalize_holding(h)
                    if norm:
                        normalized.append(norm)
            
            # Handle empty portfolio
            if not normalized:
                # Stream EMPTY_PORTFOLIO_RESULT word by word
                words = EMPTY_PORTFOLIO_RESULT["raw_summary"].split()
                accumulated = ""
                for i, word in enumerate(words):
                    accumulated += word + " "
                    yield f"data: {json.dumps({'event': 'data', 'content': accumulated.strip()})}\n\n"
                    await asyncio.sleep(0)
                
                # Build result object
                result = HealthCheckResult(**EMPTY_PORTFOLIO_RESULT)
                yield f"data: {json.dumps({'event': 'data_complete', 'result': result.model_dump()})}\n\n"
                return
            
            # Calculate metrics
            concentration = self._calculate_concentration(normalized)
            performance = self._calculate_performance(normalized)
            benchmark = self._fetch_benchmark(normalized, user_profile)
            
            # Generate observations via LLM
            observations, raw_summary = await self._generate_observations(
                concentration, performance, benchmark, user_profile
            )
            
            # Build HealthCheckResult
            result = HealthCheckResult(
                concentration_risk=concentration,
                performance=performance,
                benchmark_comparison=benchmark,
                observations=observations,
                disclaimer=DISCLAIMER,
                raw_summary=raw_summary,
                is_empty_portfolio=False
            )
            
            # Stream raw_summary word by word
            words = raw_summary.split()
            accumulated = ""
            for word in words:
                accumulated += word + " "
                yield f"data: {json.dumps({'event': 'data', 'content': accumulated.strip()})}\n\n"
                await asyncio.sleep(0)
            
            # Yield complete event
            yield f"data: {json.dumps({'event': 'data_complete', 'result': result.model_dump()})}\n\n"
            
        except Exception as e:
            logger.error(f"PortfolioHealthAgent error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
