import json
import asyncio
from typing import Optional, Any
from openai import AsyncOpenAI
from src.classifier.schemas import (
    ClassificationResult, ExtractedEntities, VALID_AGENTS, FALLBACK_CLASSIFICATION
)


CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for Valura wealth management. Your task is to classify user queries into one of 8 agent categories and extract relevant entities.

AVAILABLE AGENTS:
1. portfolio_health - Portfolio analysis, health checks, diversification, risk assessment
   Examples: "how is my portfolio?", "am I diversified?", "give me a health check", "what is my biggest risk?"

2. market_research - Company/sector analysis, stock information, earnings
   Examples: "tell me about Apple", "how is tech sector?", "NVIDIA earnings?", "analyze Tesla"

3. investment_strategy - Asset allocation, retirement planning, strategy advice
   Examples: "what asset allocation?", "should I buy bonds?", "retirement strategy?"

4. financial_calculator - Compound interest, investment projections, calculations
   Examples: "invest 1000/month for 10 years at 8%", "calculate compound interest", "what will 50000 grow to?"

5. risk_assessment - Volatility, Sharpe ratio, risk metrics
   Examples: "how risky is my portfolio?", "what is my volatility?", "Sharpe ratio?"

6. recommendations - Buy/sell recommendations, stock picks
   Examples: "what should I buy?", "should I sell AAPL?", "recommend dividend stocks"

7. predictive_analysis - Price forecasts, future value predictions
   Examples: "where will TSLA be in 6 months?", "forecast S&P 500", "predict my portfolio value"

8. general_support - Educational questions, definitions, greetings
   Examples: "what is a P/E ratio?", "how does compound interest work?", "hello"

ENTITY EXTRACTION RULES:
- tickers: Convert to UPPERCASE, strip exchange suffix (e.g., "ASML.AS" → "ASML")
- amounts: Convert to float numbers
- time_periods: Normalize to standard format (e.g., "6mo" → "6 months")
- sectors: Extract sector names mentioned
- topics: Extract key topics/themes

FOLLOW-UP RESOLUTION:
- If the query is a follow-up (references previous conversation), resolve pronouns/references
- Include the resolved subject in resolved_query field
- Example: Previous="tell me about Microsoft", Current="what about Apple?" → resolved_query="tell me about Apple"

SAFETY VERDICT:
- Set safety_verdict to "clean" for normal queries
- Set to "flagged" if query seems potentially problematic but not blocked by safety guard
- Add safety_note explaining any concerns

OUTPUT FORMAT:
Return ONLY raw JSON with no markdown formatting. No ```json``` wrappers.
{
    "intent": "<description of user intent>",
    "confidence": <float 0.0-1.0>,
    "agent": "<one of the 8 agent names>",
    "entities": {
        "tickers": ["TICKER1", "TICKER2"],
        "amounts": [1000.0, 500.0],
        "time_periods": ["1 year", "6 months"],
        "sectors": ["technology", "healthcare"],
        "topics": ["diversification", "risk"]
    },
    "safety_verdict": "clean",
    "safety_note": null,
    "resolved_query": "<the full resolved query>"
}
"""


class IntentClassifier:
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model
    
    def _build_history_context(self, history: list[dict]) -> str:
        """Format last N turns as '[Turn N] Role: content'"""
        if not history:
            return ""
        
        lines = []
        for i, turn in enumerate(history, start=1):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"[Turn {i}] {role.capitalize()}: {content}")
        
        return "\n".join(lines)
    
    def _build_user_message(self, query: str, history: list[dict]) -> str:
        """Combine history context + current query"""
        history_context = self._build_history_context(history)
        
        if history_context:
            return f"Conversation History:\n{history_context}\n\nCurrent Query: {query}"
        return query
    
    def _parse_response(self, raw: str, original_query: str) -> ClassificationResult:
        """Parse JSON response, validate agent name, normalize entities"""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return FALLBACK_CLASSIFICATION.model_copy(update={"resolved_query": original_query})
        
        # Validate agent name against VALID_AGENTS
        agent_name = data.get("agent", "general_support")
        if agent_name not in VALID_AGENTS:
            agent_name = "general_support"
        
        # Normalize tickers: uppercase and strip exchange suffix
        entities_data = data.get("entities", {})
        tickers = entities_data.get("tickers", [])
        normalized_tickers = []
        for ticker in tickers:
            # Strip exchange suffix (everything after .)
            normalized = ticker.upper().split(".")[0]
            if normalized and normalized not in normalized_tickers:
                normalized_tickers.append(normalized)
        
        # Build entities object
        entities = ExtractedEntities(
            tickers=normalized_tickers,
            amounts=[float(a) for a in entities_data.get("amounts", []) if isinstance(a, (int, float))],
            time_periods=entities_data.get("time_periods", []),
            sectors=entities_data.get("sectors", []),
            topics=entities_data.get("topics", [])
        )
        
        # Ensure resolved_query is never empty
        resolved_query = data.get("resolved_query", "").strip()
        if not resolved_query:
            resolved_query = original_query
        
        return ClassificationResult(
            intent=data.get("intent", "general"),
            confidence=float(data.get("confidence", 0.5)),
            agent=agent_name,
            entities=entities,
            safety_verdict=data.get("safety_verdict", "clean"),
            safety_note=data.get("safety_note"),
            resolved_query=resolved_query
        )
    
    async def classify(self, query: str, history: list[dict], timeout: float = 8.0) -> ClassificationResult:
        """Classify query using OpenAI LLM with timeout"""
        user_message = self._build_user_message(query, history)
        
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=512
                ),
                timeout=timeout
            )
            
            raw_content = response.choices[0].message.content
            if raw_content is None:
                return FALLBACK_CLASSIFICATION.model_copy(update={"resolved_query": query})
            
            return self._parse_response(raw_content, query)
            
        except Exception:
            # On ANY exception return fallback
            return FALLBACK_CLASSIFICATION.model_copy(update={"resolved_query": query})
