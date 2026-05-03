from abc import ABC, abstractmethod
from typing import AsyncGenerator
from src.classifier.schemas import ClassificationResult

class BaseAgent(ABC):
    @abstractmethod
    async def run(self, query: str, user_profile: dict, classification: ClassificationResult, session_id: str) -> AsyncGenerator[str, None]:
        ...
