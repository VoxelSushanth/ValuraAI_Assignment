import json
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from openai import AsyncOpenAI

from src.config import settings
from src.safety.guard import SafetyGuard
from src.classifier.classifier import IntentClassifier
from src.classifier.schemas import ClassificationResult
from src.router import build_registry, AgentRouter
from src.memory.session import SessionMemory
from src.schemas.api import ChatRequest, HealthResponse, SessionClearResponse

# Global state for lifespan management
_state: dict = {}

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    memory = SessionMemory(settings.database_path)
    await memory.initialize()
    
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    safety_guard = SafetyGuard()
    
    classifier = IntentClassifier(client=client, model=settings.openai_model_dev)
    
    registry = build_registry(client, settings.openai_model_dev)
    router = AgentRouter(registry)
    
    _state["memory"] = memory
    _state["client"] = client
    _state["safety_guard"] = safety_guard
    _state["classifier"] = classifier
    _state["router"] = router
    
    yield
    
    # Shutdown
    await client.close()
    await memory.close()


app = FastAPI(title="Valura AI Microservice", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse()


@app.delete("/api/v1/sessions/{session_id}", response_model=SessionClearResponse)
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    memory: SessionMemory = _state["memory"]
    turns_deleted = await memory.clear_session(session_id)
    
    return SessionClearResponse(
        session_id=session_id,
        turns_deleted=turns_deleted,
        message=f"Deleted {turns_deleted} turns from session {session_id}"
    )


@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with SSE streaming"""
    return EventSourceResponse(event_stream(request))


async def event_stream(request: ChatRequest) -> AsyncGenerator[dict, None]:
    """Generate SSE events for chat response"""
    try:
        guard: SafetyGuard = _state["safety_guard"]
        memory: SessionMemory = _state["memory"]
        client: AsyncOpenAI = _state["client"]
        router: AgentRouter = _state["router"]
        
        # Step 1: Safety check
        safety_result = guard.check(request.query)
        if safety_result.blocked:
            yield {
                "event": "error",
                "data": json.dumps({
                    "code": "safety_blocked",
                    "category": safety_result.category,
                    "message": safety_result.response
                })
            }
            yield {"event": "done"}
            return
        
        # Step 2: Get conversation history
        history = await memory.get_last_n_turns(request.session_id, 3)
        
        # Step 3: Classify intent with tier-based model
        model = settings.get_model_for_tier(request.tier)
        classifier = IntentClassifier(client=client, model=model)
        classification: ClassificationResult = await classifier.classify(
            request.query, history
        )
        
        # Step 4: Yield metadata event
        yield {
            "event": "metadata",
            "data": json.dumps({"classification": classification.model_dump()})
        }
        
        # Step 5: Route to appropriate agent
        agent = router.route(classification.agent)
        
        # Step 6: Stream agent response with timeout check
        start_time = asyncio.get_event_loop().time()
        timeout = settings.max_response_timeout
        
        accumulated_content = ""
        async for chunk in agent.run(
            query=request.query,
            user_profile=request.user_profile,
            classification=classification,
            session_id=request.session_id
        ):
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "code": "timeout",
                        "message": f"Response exceeded {timeout}s timeout"
                    })
                }
                break
            yield {"event": "data", "data": chunk}
        
        # Step 7: Save conversation turns
        await memory.add_turn(request.session_id, "user", request.query)
        await memory.add_turn(
            request.session_id, 
            "assistant", 
            classification.resolved_query or request.query
        )
        
        # Step 8: Yield done event
        yield {"event": "done"}
        
    except Exception as e:
        logger.error(f"Event stream error: {e}")
        yield {
            "event": "error",
            "data": json.dumps({
                "code": "internal_error",
                "message": str(e)
            })
        }
