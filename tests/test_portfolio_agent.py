import pytest
import json
from unittest.mock import MagicMock, AsyncMock

from src.agents.portfolio_health.agent import PortfolioHealthAgent
from src.classifier.schemas import ClassificationResult, ExtractedEntities


@pytest.fixture
def mock_openai_for_agent():
    """Create mocked OpenAI client for agent tests"""
    client = MagicMock()
    
    observations_json = json.dumps({
        "observations": [
            {"severity": "info", "text": "Portfolio looks healthy."},
            {"severity": "warning", "text": "Consider diversifying."}
        ],
        "raw_summary": "Your portfolio shows solid returns. Consider diversifying to reduce risk."
    })
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = observations_json
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client


@pytest.mark.asyncio
class TestEmptyPortfolio:
    """Test empty portfolio handling"""
    
    async def test_empty_does_not_crash(self, mock_openai_for_agent, mock_yfinance_fixture):
        agent = PortfolioHealthAgent(client=mock_openai_for_agent, model="gpt-4o-mini")
        
        classification = ClassificationResult(
            intent="portfolio_health",
            confidence=0.95,
            agent="portfolio_health",
            entities=ExtractedEntities(),
            safety_verdict="clean",
            resolved_query="how is my portfolio?"
        )
        
        chunks = []
        async for chunk in agent.run("test", {"holdings": []}, classification, "test-session"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
    
    async def test_empty_returns_build_message(self, mock_openai_for_agent, mock_yfinance_fixture):
        agent = PortfolioHealthAgent(client=mock_openai_for_agent, model="gpt-4o-mini")
        
        classification = ClassificationResult(
            intent="portfolio_health",
            confidence=0.95,
            agent="portfolio_health",
            entities=ExtractedEntities(),
            safety_verdict="clean",
            resolved_query="how is my portfolio?"
        )
        
        full_output = ""
        async for chunk in agent.run("test", {"holdings": []}, classification, "test-session"):
            full_output += chunk
        
        # Check for build-related keywords
        build_keywords = ["start", "begin", "empty", "foundation", "first", "invest", "journey"]
        found_any = any(kw in full_output.lower() for kw in build_keywords)
        assert found_any
    
    async def test_no_holdings_key_works(self, mock_openai_for_agent, mock_yfinance_fixture):
        agent = PortfolioHealthAgent(client=mock_openai_for_agent, model="gpt-4o-mini")
        
        classification = ClassificationResult(
            intent="portfolio_health",
            confidence=0.95,
            agent="portfolio_health",
            entities=ExtractedEntities(),
            safety_verdict="clean",
            resolved_query="how is my portfolio?"
        )
        
        chunks = []
        # Profile with no "holdings" key at all
        async for chunk in agent.run("test", {}, classification, "test-session"):
            chunks.append(chunk)
        
        assert len(chunks) > 0


@pytest.mark.asyncio
class TestConcentrationRisk:
    """Test concentration risk calculation"""
    
    async def test_concentrated_flagged_high(self, mock_openai_for_agent, mock_yfinance_fixture):
        agent = PortfolioHealthAgent(client=mock_openai_for_agent, model="gpt-4o-mini")
        
        classification = ClassificationResult(
            intent="portfolio_health",
            confidence=0.95,
            agent="portfolio_health",
            entities=ExtractedEntities(),
            safety_verdict="clean",
            resolved_query="how is my portfolio?"
        )
        
        # Concentrated portfolio: AAPL is dominant
        profile = {
            "holdings": [
                {"ticker": "AAPL", "quantity": 500, "current_price": 189.30, "cost_basis": 150.00},
                {"ticker": "MSFT", "quantity": 5, "current_price": 415.20, "cost_basis": 380.00}
            ]
        }
        
        full_output = ""
        async for chunk in agent.run("test", profile, classification, "test-session"):
            full_output += chunk
        
        # Parse data_complete event
        lines = full_output.strip().split("\n\n")
        for line in lines:
            if "data_complete" in line:
                data_str = line.replace("data: ", "")
                data = json.loads(data_str)
                result = data.get("result", {})
                assert result["concentration_risk"]["flag"] == "high"
                break


class TestCalculationsDirect:
    """Test calculation methods directly"""
    
    def test_positive_return(self, mock_openai_for_agent):
        agent = PortfolioHealthAgent(client=mock_openai_for_agent, model="gpt-4o-mini")
        
        normalized = [{
            "ticker": "TEST",
            "quantity": 10,
            "current_price": 120.0,
            "cost_basis": 100.0,
            "current_value": 1200.0,
            "cost_value": 1000.0
        }]
        
        perf = agent._calculate_performance(normalized)
        assert abs(perf.total_return_pct - 20.0) < 0.01
    
    def test_negative_return(self, mock_openai_for_agent):
        agent = PortfolioHealthAgent(client=mock_openai_for_agent, model="gpt-4o-mini")
        
        normalized = [{
            "ticker": "TEST",
            "quantity": 10,
            "current_price": 80.0,
            "cost_basis": 100.0,
            "current_value": 800.0,
            "cost_value": 1000.0
        }]
        
        perf = agent._calculate_performance(normalized)
        assert abs(perf.total_return_pct - (-20.0)) < 0.01
    
    def test_concentration_top_holding(self, mock_openai_for_agent):
        agent = PortfolioHealthAgent(client=mock_openai_for_agent, model="gpt-4o-mini")
        
        normalized = [
            {"ticker": "AAPL", "quantity": 10, "current_price": 100.0, "cost_basis": 90.0, "current_value": 1000.0, "cost_value": 900.0},
            {"ticker": "MSFT", "quantity": 10, "current_price": 50.0, "cost_basis": 45.0, "current_value": 500.0, "cost_value": 450.0}
        ]
        
        conc = agent._calculate_concentration(normalized)
        assert conc.top_holding == "AAPL"


@pytest.mark.asyncio
class TestDisclaimerPresence:
    """Test that disclaimer is present in responses"""
    
    async def test_disclaimer_in_normal_response(self, mock_openai_for_agent, mock_yfinance_fixture):
        agent = PortfolioHealthAgent(client=mock_openai_for_agent, model="gpt-4o-mini")
        
        classification = ClassificationResult(
            intent="portfolio_health",
            confidence=0.95,
            agent="portfolio_health",
            entities=ExtractedEntities(),
            safety_verdict="clean",
            resolved_query="how is my portfolio?"
        )
        
        profile = {
            "holdings": [
                {"ticker": "AAPL", "quantity": 10, "current_price": 150.0, "cost_basis": 140.0}
            ]
        }
        
        full_output = ""
        async for chunk in agent.run("test", profile, classification, "test-session"):
            full_output += chunk
        
        # Check for disclaimer text
        disclaimer_keywords = ["informational purposes only", "not constitute investment advice"]
        found = any(kw in full_output.lower() for kw in disclaimer_keywords)
        assert found
    
    async def test_disclaimer_in_empty_response(self, mock_openai_for_agent, mock_yfinance_fixture):
        agent = PortfolioHealthAgent(client=mock_openai_for_agent, model="gpt-4o-mini")
        
        classification = ClassificationResult(
            intent="portfolio_health",
            confidence=0.95,
            agent="portfolio_health",
            entities=ExtractedEntities(),
            safety_verdict="clean",
            resolved_query="how is my portfolio?"
        )
        
        full_output = ""
        async for chunk in agent.run("test", {"holdings": []}, classification, "test-session"):
            full_output += chunk
        
        # Check for disclaimer text
        disclaimer_keywords = ["informational purposes only", "not constitute investment advice"]
        found = any(kw in full_output.lower() for kw in disclaimer_keywords)
        assert found


@pytest.mark.asyncio
class TestYfinanceFailure:
    """Test behavior when yfinance fails"""
    
    async def test_yfinance_failure_still_returns_result(self, mock_openai_for_agent, monkeypatch):
        import yfinance as yf
        
        # Make yf.Ticker raise exception
        def failing_ticker(*args, **kwargs):
            raise Exception("Network error")
        
        monkeypatch.setattr(yf, "Ticker", failing_ticker)
        
        agent = PortfolioHealthAgent(client=mock_openai_for_agent, model="gpt-4o-mini")
        
        classification = ClassificationResult(
            intent="portfolio_health",
            confidence=0.95,
            agent="portfolio_health",
            entities=ExtractedEntities(),
            safety_verdict="clean",
            resolved_query="how is my portfolio?"
        )
        
        profile = {
            "holdings": [
                {"ticker": "AAPL", "quantity": 10, "current_price": 150.0, "cost_basis": 140.0}
            ]
        }
        
        chunks = []
        async for chunk in agent.run("test", profile, classification, "test-session"):
            chunks.append(chunk)
        
        # Should still return some result despite yfinance failure
        assert len(chunks) > 0
