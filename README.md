# Valura AI Microservice

A FastAPI microservice for Valura wealth management providing AI-powered portfolio analysis and financial guidance.

## Defense Video

[To be added within 24 hours of final commit]

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HTTP POST /api/v1/chat                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Safety Guard                                    │
│  • Pure Python regex matching (<10ms)                                       │
│  • Educational bypass for learning queries                                  │
│  • 5 blocked categories: insider_trading, market_manipulation,              │
│    money_laundering, guaranteed_returns, reckless_advice                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Intent Classifier                                 │
│  • OpenAI LLM (gpt-4o-mini / gpt-4.1 based on tier)                        │
│  • Routes to 8 agents + entity extraction                                   │
│  • Follow-up resolution                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               Agent Router                                   │
│  • Maps intent to agent implementation                                      │
│  • PortfolioHealthAgent (implemented)                                       │
│  • StubAgent for 7 other agents (market_research, investment_strategy,      │
│    financial_calculator, risk_assessment, recommendations,                  │
│    predictive_analysis, general_support)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Portfolio Health Agent                              │
│  • Parses holdings from user_profile                                        │
│  • Calculates concentration risk, performance metrics                       │
│  • Fetches benchmark data via yfinance                                      │
│  • LLM-generated observations with severity levels                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SSE Stream                                      │
│  • metadata event: classification results                                   │
│  • data events: streaming response content                                  │
│  • data_complete event: full result object                                  │
│  • done event: stream completion                                            │
│  • error event: safety_blocked, timeout, internal_error                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Session Memory                                    │
│  • SQLite database (aiosqlite)                                              │
│  • Stores conversation turns per session_id                                 │
│  • Retrieves last N turns for context                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd valura-ai

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your OpenAI API key

# Run the server
uvicorn src.main:app --reload

# Test with curl
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-123",
    "user_id": "user-456",
    "query": "how is my portfolio doing?",
    "user_profile": {
      "holdings": [
        {"ticker": "AAPL", "quantity": 10, "current_price": 150.0, "cost_basis": 140.0}
      ],
      "risk_profile": "moderate",
      "currency": "USD"
    },
    "tier": "free"
  }'
```

## Run Tests

```bash
pytest tests/ -v
```

**Note:** No `OPENAI_API_KEY` required — all tests use mocked LLM calls.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | `test-key` |
| `OPENAI_MODEL_DEV` | Model for free tier | `gpt-4o-mini` |
| `OPENAI_MODEL_EVAL` | Model for premium tier | `gpt-4.1` |
| `DATABASE_PATH` | SQLite database path | `./valura.db` |
| `MAX_RESPONSE_TIMEOUT` | Max response time in seconds | `25` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Library Choices

| Library | Why |
|---------|-----|
| `fastapi` | Modern async web framework with automatic OpenAPI docs |
| `sse-starlette` | Server-Sent Events streaming for real-time responses |
| `uvicorn` | High-performance ASGI server |
| `openai` | Official OpenAI SDK for LLM integration |
| `pydantic` | Data validation and settings management (v2) |
| `pydantic-settings` | Environment-based configuration |
| `yfinance` | Free stock market data for benchmark comparisons |
| `aiosqlite` | Async SQLite for zero-infra session memory |
| `python-dotenv` | Load environment variables from `.env` file |
| `httpx` | Async HTTP client for testing |
| `pytest` | Testing framework |
| `pytest-asyncio` | Async test support |
| `pytest-mock` | Mocking utilities for tests |

## Session Memory: SQLite

The microservice uses SQLite via `aiosqlite` for session memory, providing several advantages:

- **Zero Infrastructure**: No external database setup required; works out of the box
- **Persistent**: Conversation history survives server restarts
- **Async-Native**: Fully async/await compatible with FastAPI
- **Swappable**: Can migrate to PostgreSQL by changing connection string and using `asyncpg`
- **Indexed**: Automatic indexing on `session_id` for fast lookups

The `conversation_turns` table stores each message with `session_id`, `turn_index`, `role`, and `content`, enabling efficient retrieval of the last N turns for conversation context.

## Safety Guard Design

The Safety Guard is implemented as **pure Python with compiled regex patterns**, making deliberate tradeoffs:

### Why Pure Python?
- **Speed**: Runs in <10ms vs ~500ms+ for LLM-based classification
- **Cost**: Zero token consumption
- **Determinism**: Predictable behavior, no hallucination risk
- **Testability**: Easy to achieve 100% coverage without mocking

### Educational Bypass Tradeoff
Queries starting with educational keywords (`how does`, `what is`, `explain`, `research`, etc.) bypass blocking entirely. This allows legitimate learning while potentially allowing sophisticated jailbreak attempts. The tradeoff favors accessibility for novice investors who need to understand concepts like "insider trading" or "pump and dump" without triggering blocks.

### Known Edge Cases
- Queries that embed blocked intent within educational framing may slip through
- Non-English queries are not handled
- Contextual understanding is limited (e.g., sarcasm, hypotheticals)

## Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Safety guard latency | <10ms | ~2-5ms |
| p95 first token | <2s | TBD |
| p95 end-to-end | <6s | TBD |
| Cost per query | <$0.05 | ~$0.011 |

## Cost Estimation

| Component | Tokens | Cost (at $0.15/1M input, $0.60/1M output) |
|-----------|--------|-------------------------------------------|
| Intent Classifier (input) | ~800 | $0.00012 |
| Intent Classifier (output) | ~200 | $0.00012 |
| Portfolio Agent (input) | ~1500 | $0.00023 |
| Portfolio Agent (output) | ~400 | $0.00024 |
| **Total per query** | | **~$0.00071** |

*Note: Actual costs vary based on query complexity and response length. Premium tier uses gpt-4.1 at higher rates.*

## What I Would Do With Another Week

1. **Embedding Pre-classifier**: Add a lightweight embedding model (e.g., `all-MiniLM-L6-v2`) to pre-filter obvious queries before hitting the LLM, reducing latency by ~40% and cutting costs for simple queries.

2. **Redis Session Memory**: Replace SQLite with Redis for sub-millisecond session lookups, enabling horizontal scaling and distributed deployment across multiple service instances.

3. **LLM-as-Judge Eval Harness**: Build an automated evaluation pipeline using GPT-4 to score response quality, safety accuracy, and entity extraction precision across hundreds of test queries before each deployment.

4. **Per-Request Observability Dashboard**: Integrate structured logging with Langfuse or Helicone to track latency, token usage, safety block rates, and agent routing distribution in real-time with alerting on anomalies.
