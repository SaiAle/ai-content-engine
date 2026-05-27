"""
Shared pytest fixtures for the AI Content Automation Engine test suite.

All external I/O (LangChain models, Tavily, Firecrawl, Postgres) is mocked.
asyncio_mode = "auto" is set so every async test runs automatically without
an explicit @pytest.mark.asyncio decorator.

Heavy optional dependencies that are not installed in the test environment
(langgraph-checkpoint-postgres, langchain.hub, langchain.agents) are stubbed
into sys.modules BEFORE any project module is imported so collection never
fails due to a missing package.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage

# ---------------------------------------------------------------------------
# Ensure the project root is importable when pytest is run from tests/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Stub heavy / unavailable modules before any project code is imported.
# This must happen at module scope (not inside a fixture) so that collection
# itself never triggers an ImportError.
#
# Use unconditional assignment (sys.modules[name] = mod) NOT setdefault,
# so that even if the real package is installed it cannot load its broken
# C-extension / native-library dependencies during test collection.
# ---------------------------------------------------------------------------

def _stub(name: str, **attrs) -> MagicMock:
    """Create a MagicMock module stub and register it in sys.modules."""
    mod = MagicMock()
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod  # unconditional — overrides real package if installed
    return mod


# psycopg / psycopg2 — required by langgraph-checkpoint-postgres; libpq
# is not available in the test container so we stub the whole tree first.
for _psycopg_mod in [
    "psycopg",
    "psycopg.pq",
    "psycopg.abc",
    "psycopg.adapt",
    "psycopg.connection",
    "psycopg.cursor",
    "psycopg.rows",
    "psycopg.types",
    "psycopg_pool",
    "psycopg2",
]:
    _stub(_psycopg_mod)

# langgraph.checkpoint.postgres (requires psycopg; not installed in CI)
_pg_mod = _stub("langgraph.checkpoint.postgres")
_pg_aio = _stub("langgraph.checkpoint.postgres.aio")
_MockSaver = MagicMock()
_MockSaver.from_conn_string = AsyncMock(return_value=AsyncMock())
_pg_aio.AsyncPostgresSaver = _MockSaver
_pg_mod.aio = _pg_aio

# langchain.hub  (removed in langchain >=1.x)
_stub("langchain.hub")

# langchain.agents (create_react_agent may not be present in all versions)
_stub("langchain.agents")

# langgraph.prebuilt.create_react_agent (ensure importable)
import langgraph.prebuilt as _lgp  # noqa: E402
if not hasattr(_lgp, "create_react_agent"):
    _lgp.create_react_agent = MagicMock()

from state import (  # noqa: E402
    ContentState,
    Draft,
    Outline,
    ReviewResult,
    SEOResult,
    Source,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# pytest-asyncio configuration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")


# ---------------------------------------------------------------------------
# mock_settings
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    """
    Patch config.settings with safe, deterministic test values.
    No real API keys or external services are required.
    """
    with patch("config.settings") as patched:
        patched.openai_api_key = "test-openai-key"
        patched.anthropic_api_key = "test-anthropic-key"
        patched.tavily_api_key = "test-tavily-key"
        patched.firecrawl_api_key = "test-firecrawl-key"
        patched.database_url = (
            "postgresql+asyncpg://postgres:password@localhost:5432/test_db"
        )
        patched.redis_url = "redis://localhost:6379/0"
        patched.jwt_secret_key = "test-secret-key-for-testing-only"
        patched.jwt_algorithm = "HS256"
        patched.jwt_expire_minutes = 60
        patched.primary_model = "anthropic:claude-haiku-test"
        patched.review_model = "openai:gpt-4o-test"
        patched.fast_model = "anthropic:claude-haiku-test"
        patched.max_revision_iterations = 3
        patched.max_recursion_limit = 10
        patched.token_budget_per_run = 80_000
        patched.node_timeout_seconds = 60.0
        patched.global_run_timeout_seconds = 600.0
        patched.wordpress_url = "https://test.example.com"
        patched.wordpress_username = "testuser"
        patched.wordpress_app_password = "testpass"
        patched.langsmith_tracing = False
        patched.langsmith_api_key = ""
        patched.langsmith_project = "test-project"
        yield patched


# ---------------------------------------------------------------------------
# sample_state
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_state() -> ContentState:
    """
    A fully-populated ContentState for nodes that require prior pipeline state.
    """
    outline = Outline(
        title="The Complete Guide to AI Agents in 2025",
        meta_description=(
            "Discover how AI agents are transforming automation. "
            "Learn architecture, use cases, and implementation strategies for 2025."
        ),
        sections=[
            "What Are AI Agents?",
            "Core Architecture and Components",
            "Real-World Use Cases",
            "Implementation Guide",
            "Future Outlook",
        ],
        target_word_count=1500,
        target_audience="Software engineers and technical product managers",
        tone="informative",
        primary_keyword="AI agents",
        secondary_keywords=[
            "autonomous agents",
            "LLM agents",
            "agentic AI",
            "multi-agent systems",
        ],
    )

    draft = Draft(
        title="The Complete Guide to AI Agents in 2025",
        body=(
            "# The Complete Guide to AI Agents in 2025\n\n"
            "AI agents are autonomous software entities that perceive their environment "
            "and take actions to achieve goals. [Source](https://example.com/ai-agents)\n\n"
            "## What Are AI Agents?\n\n"
            "An AI agent combines a large language model with tools, memory, and planning "
            "capabilities to complete complex, multi-step tasks without constant human supervision.\n\n"
            "## Core Architecture and Components\n\n"
            "Modern AI agents consist of four key components: perception, reasoning, action, "
            "and memory. The reasoning layer is typically powered by an LLM.\n\n"
            "## Real-World Use Cases\n\n"
            "AI agents are being deployed in customer support, code generation, research "
            "automation, and content creation pipelines.\n\n"
            "## Implementation Guide\n\n"
            "To implement an AI agent, start with a framework like LangGraph or AutoGen, "
            "define your tools, and establish clear success criteria.\n\n"
            "## Future Outlook\n\n"
            "Multi-agent systems and agentic AI architectures will become standard "
            "infrastructure components by 2026.\n"
        ),
        word_count=152,
        sources_used=[
            "https://example.com/ai-agents",
            "https://example.com/llm-agents",
        ],
    )

    review = ReviewResult(
        verdict="ACCEPT",
        quality_score=8.5,
        issues=[],
        suggestions=["Consider adding a comparison table of popular frameworks"],
        fact_check_passed=True,
    )

    seo_result = SEOResult(
        seo_score=82.0,
        optimized_title="AI Agents Guide 2025: Architecture, Use Cases & Implementation",
        optimized_meta=(
            "Master AI agents in 2025. Explore architecture, real-world use cases, "
            "and step-by-step implementation strategies for autonomous AI systems."
        ),
        optimized_body=draft.body,
        slug="ai-agents-guide-2025",
    )

    sources = [
        Source(
            url="https://example.com/ai-agents",
            title="Introduction to AI Agents",
            snippet="AI agents are autonomous software systems that use LLMs to complete tasks.",
            relevance_score=0.95,
        ),
        Source(
            url="https://example.com/llm-agents",
            title="LLM-Powered Agents",
            snippet="Large language models enable agents to reason about complex problems.",
            relevance_score=0.88,
        ),
    ]

    return ContentState(
        topic="AI Agents",
        content_type="blog_post",
        target_platform="wordpress",
        sources=sources,
        outline=outline,
        draft=draft,
        review=review,
        seo_result=seo_result,
        revision_iteration=1,
        human_approved=None,
        token_usage=TokenUsage(
            total_tokens=5_000,
            prompt_tokens=3_500,
            completion_tokens=1_500,
            estimated_cost_usd=0.034,
        ),
        run_metadata={
            "thread_id": "test-thread-001",
            "topic": "AI Agents",
            "content_type": "blog_post",
            "target_platform": "wordpress",
            "research_summary": (
                "AI agents combine LLMs with tools and memory to autonomously complete tasks."
            ),
        },
    )


# ---------------------------------------------------------------------------
# mock_llm
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """
    A MagicMock that behaves like a LangChain BaseChatModel.

    - .invoke() / .ainvoke() return a fake AIMessage
    - .with_structured_output() returns itself (chain pipe works)
    - .with_retry() / .with_fallbacks() return itself
    - supports the | operator via __or__
    """
    llm = MagicMock()
    fake_message = AIMessage(content="This is a test response from the mock LLM.")
    llm.invoke.return_value = fake_message
    llm.ainvoke = AsyncMock(return_value=fake_message)
    llm.__or__ = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    llm.with_retry = MagicMock(return_value=llm)
    llm.with_fallbacks = MagicMock(return_value=llm)
    return llm


# ---------------------------------------------------------------------------
# async_client  (FastAPI + httpx)
# ---------------------------------------------------------------------------


@pytest.fixture
async def async_client():
    """
    An httpx.AsyncClient pointed at the FastAPI app.

    The Postgres checkpointer startup/shutdown lifecycle is mocked so no
    real database is required.  The global _checkpointer is injected
    directly so get_checkpointer() dependency works without startup event.
    """
    mock_saver = AsyncMock()
    mock_saver.setup = AsyncMock()
    mock_saver.aclose = AsyncMock()
    mock_saver.aget_state = AsyncMock(return_value=None)

    with (
        patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.from_conn_string",
            new=AsyncMock(return_value=mock_saver),
        ),
        patch("api.main.AsyncPostgresSaver") as mock_cls,
    ):
        mock_cls.from_conn_string = AsyncMock(return_value=mock_saver)

        import api.main as main_module

        main_module._checkpointer = mock_saver

        from api.main import app

        async with httpx.AsyncClient(
            app=app,
            base_url="http://testserver",
        ) as client:
            yield client
