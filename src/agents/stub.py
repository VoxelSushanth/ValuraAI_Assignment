import json
from typing import AsyncGenerator
from src.agents.base import BaseAgent
from src.classifier.schemas import ClassificationResult

class StubAgent(BaseAgent):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
    
    async def run(self, query, user_profile, classification, session_id) -> AsyncGenerator[str, None]:
        payload = {
            "event": "not_implemented",
            "status": "not_implemented",
            "agent": self.agent_name,
            "intent": classification.intent,
            "confidence": classification.confidence,
            "entities": classification.entities.model_dump(),
            "resolved_query": classification.resolved_query,
            "message": f"The '{self.agent_name}' agent is not yet implemented. Intent classification was successful."
        }
        yield f"data: {json.dumps(payload)}\n\n"
