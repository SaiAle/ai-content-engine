"""
Shared node utilities:
  • build_model()   — provider-agnostic model with fallbacks + retries
  • track_tokens()  — accumulate token usage into state
  • node_timeout()  — asyncio timeout context manager
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from loguru import logger

from config import settings
from state import TokenUsage


def build_model(
    model_id: str | None = None,
    *,
    fallback_id: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """
    Build a provider-agnostic chat model using `init_chat_model`.
    Wraps with exponential-backoff retries and an optional fallback model.

    model_id format: "provider:model-name"  e.g. "anthropic:claude-sonnet-4-6"
    """
    primary_id = model_id or settings.primary_model
    fallback_model_id = fallback_id or settings.fast_model

    primary = init_chat_model(
        primary_id,
        temperature=temperature,
        max_tokens=max_tokens,
    ).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    if primary_id == fallback_model_id:
        return primary

    fallback = init_chat_model(
        fallback_model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    ).with_retry(stop_after_attempt=2, wait_exponential_jitter=True)

    return primary.with_fallbacks([fallback])


@asynccontextmanager
async def node_timeout(seconds: float | None = None) -> AsyncIterator[None]:
    """Context manager that cancels the enclosed block after `seconds`."""
    limit = seconds or settings.node_timeout_seconds
    try:
        async with asyncio.timeout(limit):
            yield
    except asyncio.TimeoutError:
        raise TimeoutError(f"Node exceeded {limit}s timeout")


def accumulate_tokens(current: TokenUsage, response_metadata: dict) -> TokenUsage:
    """
    Parse token counts from a LangChain response metadata dict and add to
    the running TokenUsage tally.  Handles OpenAI and Anthropic formats.
    """
    usage = response_metadata.get("usage", response_metadata.get("token_usage", {}))
    prompt = int(
        usage.get("prompt_tokens", usage.get("input_tokens", 0))
    )
    completion = int(
        usage.get("completion_tokens", usage.get("output_tokens", 0))
    )
    total = prompt + completion
    # Rough cost estimate: $3/1M input + $15/1M output (Claude Sonnet 4.6 pricing)
    cost = (prompt * 3 + completion * 15) / 1_000_000

    return TokenUsage(
        total_tokens=current.total_tokens + total,
        prompt_tokens=current.prompt_tokens + prompt,
        completion_tokens=current.completion_tokens + completion,
        estimated_cost_usd=current.estimated_cost_usd + cost,
    )


def check_token_budget(usage: TokenUsage) -> None:
    """Raise BudgetExceededError if the per-run token budget is exhausted."""
    if usage.total_tokens >= settings.token_budget_per_run:
        raise RuntimeError(
            f"Token budget exhausted: {usage.total_tokens} >= "
            f"{settings.token_budget_per_run} tokens"
        )
