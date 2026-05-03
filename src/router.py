import logging
from src.agents.base import BaseAgent
from src.agents.stub import StubAgent
from src.agents.portfolio_health.agent import PortfolioHealthAgent
from src.classifier.schemas import VALID_AGENTS

logger = logging.getLogger(__name__)


def build_registry(client, model: str) -> dict[str, BaseAgent]:
    """Build agent registry with all agents"""
    registry = {
        "portfolio_health": PortfolioHealthAgent(client=client, model=model)
    }
    
    for name in VALID_AGENTS:
        if name not in registry:
            registry[name] = StubAgent(agent_name=name)
    
    return registry


class AgentRouter:
    def __init__(self, registry: dict[str, BaseAgent]):
        self.registry = registry
    
    def route(self, agent_name: str) -> BaseAgent:
        """Route to appropriate agent by name"""
        return self.registry.get(agent_name, StubAgent(agent_name=agent_name))
