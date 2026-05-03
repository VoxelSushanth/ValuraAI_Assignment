from pydantic import BaseModel, Field
from typing import Literal

class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    query: str = Field(min_length=1, max_length=2000)
    user_profile: dict = Field(default_factory=dict)
    tier: Literal["free","premium"] = "free"

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"

class SessionClearResponse(BaseModel):
    session_id: str
    turns_deleted: int
    message: str
