import pytest
from unittest.mock import MagicMock

from src.router import build_registry, AgentRouter
from src.agents.stub import StubAgent
from src.agents.portfolio_health.agent import PortfolioHealthAgent
from src.classifier.schemas import VALID_AGENTS


@pytest.fixture
def mock_client():
    """Return mocked OpenAI client"""
    return MagicMock()


@pytest.fixture
def registry(mock_client):
    """Build agent registry"""
    return build_registry(mock_client, "gpt-4o-mini")


@pytest.fixture
def router(registry):
    """Create AgentRouter with registry"""
    return AgentRouter(registry)


class TestRouterMapping:
    """Test router correctly maps agents"""
    
    def test_portfolio_health_is_correct_class(self, router):
        agent = router.route("portfolio_health")
        assert isinstance(agent, PortfolioHealthAgent)
    
    def test_all_others_are_stub(self, router):
        for name in VALID_AGENTS:
            if name != "portfolio_health":
                agent = router.route(name)
                assert isinstance(agent, StubAgent)
    
    def test_unknown_agent_returns_stub(self, router):
        agent = router.route("unknown_xyz")
        assert isinstance(agent, StubAgent)
    
    def test_unknown_agent_has_correct_name(self, router):
        agent = router.route("mystery")
        assert agent.agent_name == "mystery"
    
    def test_all_valid_agents_in_registry(self, registry):
        for name in VALID_AGENTS:
            assert name in registry


@pytest.mark.asyncio
class TestStubOutput:
    """Test StubAgent output format"""
    
    async def test_stub_yields_valid_json(self):
        from src.classifier.schemas import ClassificationResult, ExtractedEntities
        
        stub = StubAgent("market_research")
        
        classification = ClassificationResult(
            intent="market_research",
            confidence=0.9,
            agent="market_research",
            entities=ExtractedEntities(),
            safety_verdict="clean",
            resolved_query="test query"
        )
        
        chunks = []
        async for chunk in stub.run("test", {}, classification, "test-session"):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Check SSE format
        assert chunk.startswith("data: ")
        
        # Parse JSON
        import json
        data_str = chunk.replace("data: ", "").strip()
        data = json.loads(data_str)
        
        assert data["status"] == "not_implemented"
        assert data["agent"] == "market_research"
