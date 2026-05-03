import json
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
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

_state: dict = {}
logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Valura AI starting up...")

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

    logger.info("Valura AI ready.")
    yield

    logger.info("Valura AI shutting down...")
    await client.close()
    await memory.close()


app = FastAPI(title="Valura AI Microservice", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()


@app.delete("/api/v1/sessions/{session_id}", response_model=SessionClearResponse)
async def clear_session(session_id: str):
    memory: SessionMemory = _state["memory"]
    turns_deleted = await memory.clear_session(session_id)
    return SessionClearResponse(
        session_id=session_id,
        turns_deleted=turns_deleted,
        message=f"Deleted {turns_deleted} turns from session {session_id}"
    )


@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    return EventSourceResponse(event_stream(request))


async def event_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Full pipeline: Safety → Classify → Route → Stream → Save.
    All SSE events are raw strings: "data: {json}\n\n"
    This avoids double-wrapping when agents yield their own SSE strings.
    """
    def make_event(payload: dict) -> str:
        """Helper: format a dict as a raw SSE data string."""
        return f"data: {json.dumps(payload)}\n\n"

    try:
        guard: SafetyGuard = _state["safety_guard"]
        memory: SessionMemory = _state["memory"]
        client: AsyncOpenAI = _state["client"]
        router: AgentRouter = _state["router"]

        # ── Step 1: Safety check ──────────────────────────────────────────
        safety_result = guard.check(request.query)
        if safety_result.blocked:
            logger.info(f"Query blocked. Category: {safety_result.category}")
            yield make_event({
                "event": "error",
                "code": "safety_blocked",
                "category": safety_result.category,
                "message": safety_result.response
            })
            yield make_event({"event": "done"})
            return

        # ── Step 2: Conversation history ──────────────────────────────────
        history = await memory.get_last_n_turns(request.session_id, 3)

        # ── Step 3: Classify ──────────────────────────────────────────────
        model = settings.get_model_for_tier(request.tier)
        classifier = IntentClassifier(client=client, model=model)
        classification: ClassificationResult = await classifier.classify(
            request.query, history
        )
        logger.info(f"Classified as: {classification.agent} (confidence: {classification.confidence:.2f})")

        # ── Step 4: Metadata event ────────────────────────────────────────
        yield make_event({
            "event": "metadata",
            "classification": classification.model_dump()
        })

        # ── Step 5: Route ─────────────────────────────────────────────────
        agent = router.route(classification.agent)

        # ── Step 6: Stream agent response with wall-clock timeout ─────────
        start_time = asyncio.get_event_loop().time()
        timed_out = False

        async for chunk in agent.run(
            query=request.query,
            user_profile=request.user_profile,
            classification=classification,
            session_id=request.session_id
        ):
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > settings.max_response_timeout:
                logger.warning(f"Agent timed out after {elapsed:.1f}s")
                yield make_event({
                    "event": "error",
                    "code": "timeout",
                    "message": f"Response exceeded {settings.max_response_timeout}s timeout"
                })
                timed_out = True
                break

            # chunk is already a raw SSE string from the agent: "data: {...}\n\n"
            # yield it directly — do NOT wrap again
            yield chunk

        # ── Step 7: Save to memory ────────────────────────────────────────
        if not timed_out:
            try:
                await memory.add_turn(request.session_id, "user", request.query)
                await memory.add_turn(
                    request.session_id,
                    "assistant",
                    classification.resolved_query or request.query
                )
            except Exception as e:
                logger.error(f"Memory save failed (non-fatal): {e}")

        # ── Step 8: Done ──────────────────────────────────────────────────
        yield make_event({"event": "done"})

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        yield f"data: {json.dumps({'event': 'error', 'code': 'internal_error', 'message': 'An unexpected error occurred.'})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
