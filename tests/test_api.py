import pytest
import pytest_asyncio
import json
from httpx import AsyncClient, ASGITransport

from src.main import app, _state
from src.safety.guard import SafetyGuard
from src.classifier.classifier import IntentClassifier
from src.router import build_registry, AgentRouter
from src.memory.session import SessionMemory
from src.agents.stub import StubAgent
from src.classifier.schemas import VALID_AGENTS


@pytest_asyncio.fixture
async def test_client(tmp_path):
    """Create test client with mocked dependencies"""
    # Setup SessionMemory in tmp_path
    db_path = str(tmp_path / "test.db")
    memory = SessionMemory(db_path)
    await memory.initialize()
    
    # Create mock OpenAI client
    from unittest.mock import MagicMock, AsyncMock
    mock_openai_client = MagicMock()
    
    classification_json = json.dumps({
        "intent": "portfolio_health",
        "confidence": 0.95,
        "agent": "portfolio_health",
        "entities": {},
        "resolved_query": "how is my portfolio?"
    })
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = classification_json
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Build registry with ALL StubAgents (faster tests)
    all_stub_registry = {name: StubAgent(name) for name in VALID_AGENTS}
    router = AgentRouter(all_stub_registry)
    
    # Inject into _state
    _state["memory"] = memory
    _state["client"] = mock_openai_client
    _state["safety_guard"] = SafetyGuard()
    _state["classifier"] = IntentClassifier(mock_openai_client, "gpt-4o-mini")
    _state["router"] = router
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    
    await memory.close()


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Test health check endpoint"""
    
    async def test_returns_200(self, test_client):
        response = await test_client.get("/health")
        assert response.status_code == 200
    
    async def test_returns_ok_status(self, test_client):
        response = await test_client.get("/health")
        data = response.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
class TestChatEndpointBlocking:
    """Test chat endpoint with blocked queries"""
    
    async def test_blocked_query_contains_safety_blocked(self, test_client):
        payload = {
            "session_id": "test-session",
            "user_id": "test-user",
            "query": "I have insider info from a CEO, should I trade before the announcement?",
            "user_profile": {},
            "tier": "free"
        }
        
        response = await test_client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        assert "safety_blocked" in response.text
    
    async def test_blocked_query_returns_200(self, test_client):
        """SSE always returns 200 even when blocked"""
        payload = {
            "session_id": "test-session",
            "user_id": "test-user",
            "query": "I want to launder money",
            "user_profile": {},
            "tier": "free"
        }
        
        response = await test_client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200


@pytest.mark.asyncio
class TestChatEndpointLegitimate:
    """Test chat endpoint with legitimate queries"""
    
    async def test_legitimate_query_contains_metadata(self, test_client):
        payload = {
            "session_id": "test-session",
            "user_id": "test-user",
            "query": "how is my portfolio?",
            "user_profile": {"holdings": []},
            "tier": "free"
        }
        
        response = await test_client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        assert "metadata" in response.text
    
    async def test_legitimate_query_contains_done(self, test_client):
        payload = {
            "session_id": "test-session",
            "user_id": "test-user",
            "query": "how is my portfolio?",
            "user_profile": {"holdings": []},
            "tier": "free"
        }
        
        response = await test_client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        assert "done" in response.text


@pytest.mark.asyncio
class TestSessionEndpoint:
    """Test session management endpoint"""
    
    async def test_delete_session_200(self, test_client):
        response = await test_client.delete("/api/v1/sessions/test-session")
        assert response.status_code == 200
    
    async def test_delete_returns_session_id(self, test_client):
        response = await test_client.delete("/api/v1/sessions/test-session")
        data = response.json()
        assert data["session_id"] == "test-session"
