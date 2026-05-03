import pytest
import json
from unittest.mock import MagicMock, AsyncMock

from src.classifier.classifier import IntentClassifier
from src.classifier.schemas import VALID_AGENTS, ClassificationResult, ExtractedEntities


def make_classifier_with_response(response_json: dict) -> IntentClassifier:
    """Helper to create classifier with mocked response"""
    client = MagicMock()
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps(response_json)
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    return IntentClassifier(client=client, model="gpt-4o-mini")


@pytest.mark.asyncio
class TestClassifierRouting:
    """Test that classifier routes to correct agents"""
    
    async def test_portfolio_health_routes_correctly(self):
        mock_data = {
            "intent": "portfolio_analysis",
            "confidence": 0.95,
            "agent": "portfolio_health",
            "entities": {},
            "resolved_query": "how is my portfolio?"
        }
        classifier = make_classifier_with_response(mock_data)
        result = await classifier.classify("how is my portfolio?", [])
        assert result.agent == "portfolio_health"
    
    async def test_market_research_routes_correctly(self):
        mock_data = {
            "intent": "stock_research",
            "confidence": 0.90,
            "agent": "market_research",
            "entities": {"tickers": ["AAPL"]},
            "resolved_query": "tell me about Apple"
        }
        classifier = make_classifier_with_response(mock_data)
        result = await classifier.classify("tell me about Apple", [])
        assert result.agent == "market_research"
        assert "AAPL" in result.entities.tickers
    
    async def test_financial_calculator_routes_correctly(self):
        mock_data = {
            "intent": "calculation",
            "confidence": 0.88,
            "agent": "financial_calculator",
            "entities": {"amounts": [1000.0]},
            "resolved_query": "calculate compound interest"
        }
        classifier = make_classifier_with_response(mock_data)
        result = await classifier.classify("calculate compound interest", [])
        assert result.agent == "financial_calculator"
    
    async def test_investment_strategy_routes_correctly(self):
        mock_data = {
            "intent": "strategy_planning",
            "confidence": 0.85,
            "agent": "investment_strategy",
            "entities": {},
            "resolved_query": "what asset allocation should I have?"
        }
        classifier = make_classifier_with_response(mock_data)
        result = await classifier.classify("what asset allocation should I have?", [])
        assert result.agent == "investment_strategy"


@pytest.mark.asyncio
class TestClassifierFallback:
    """Test classifier fallback behavior"""
    
    async def test_api_exception_returns_fallback(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        classifier = IntentClassifier(client=client, model="gpt-4o-mini")
        result = await classifier.classify("test query", [])
        
        assert result.agent == "general_support"
        assert result.confidence == 0.1
        assert result.resolved_query == "test query"
    
    async def test_invalid_json_returns_fallback(self):
        client = MagicMock()
        
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "not json {"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        classifier = IntentClassifier(client=client, model="gpt-4o-mini")
        result = await classifier.classify("test query", [])
        
        assert result.agent == "general_support"
    
    async def test_invalid_agent_name_corrected(self):
        mock_data = {
            "intent": "test",
            "confidence": 0.9,
            "agent": "made_up_agent",
            "entities": {},
            "resolved_query": "test"
        }
        classifier = make_classifier_with_response(mock_data)
        result = await classifier.classify("test", [])
        assert result.agent == "general_support"


@pytest.mark.asyncio
class TestEntityExtraction:
    """Test entity extraction and normalization"""
    
    async def test_tickers_uppercased(self):
        mock_data = {
            "intent": "research",
            "confidence": 0.9,
            "agent": "market_research",
            "entities": {"tickers": ["aapl", "nvda"]},
            "resolved_query": "tell me about aapl and nvda"
        }
        classifier = make_classifier_with_response(mock_data)
        result = await classifier.classify("tell me about aapl and nvda", [])
        
        assert all(t.isupper() for t in result.entities.tickers)
        assert "AAPL" in result.entities.tickers
        assert "NVDA" in result.entities.tickers
    
    async def test_exchange_suffix_stripped(self):
        mock_data = {
            "intent": "research",
            "confidence": 0.9,
            "agent": "market_research",
            "entities": {"tickers": ["ASML.AS"]},
            "resolved_query": "tell me about ASML"
        }
        classifier = make_classifier_with_response(mock_data)
        result = await classifier.classify("tell me about ASML.AS", [])
        
        assert "ASML" in result.entities.tickers
        assert "ASML.AS" not in result.entities.tickers
    
    async def test_followup_resolved(self):
        mock_data = {
            "intent": "research",
            "confidence": 0.9,
            "agent": "market_research",
            "entities": {"tickers": ["AAPL"]},
            "resolved_query": "tell me about Apple"
        }
        classifier = make_classifier_with_response(mock_data)
        
        history = [{"role": "user", "content": "tell me about Microsoft"}]
        result = await classifier.classify("what about Apple?", history)
        
        assert "Apple" in result.resolved_query


@pytest.mark.asyncio
class TestAllAgentsValid:
    """Test that all 8 agents are accepted"""
    
    async def test_all_eight_agents_accepted(self):
        for agent_name in VALID_AGENTS:
            mock_data = {
                "intent": "test",
                "confidence": 0.9,
                "agent": agent_name,
                "entities": {},
                "resolved_query": "test query"
            }
            classifier = make_classifier_with_response(mock_data)
            result = await classifier.classify("test", [])
            assert result.agent == agent_name
