"""
Planner node — produces a structured Outline from topic + seed sources.

Objective: $\text{Outline} = \arg\max_{o} P(o \mid \text{topic}, \text{sources})$
Structured output enforced via Pydantic + `with_structured_output`.

Input:  state.topic, state.sources, state.content_type
Output: state.outline
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from loguru import logger

from config import settings
from nodes.base import accumulate_tokens, build_model, check_token_budget, node_timeout
from state import ContentState, Outline, TokenUsage

_SYSTEM = """You are a senior content strategist and SEO expert.
Given a topic and seed research, produce a structured content outline.
Return ONLY valid JSON matching the Outline schema — no commentary."""

_HUMAN = """Topic: {topic}
Content type: {content_type}
Seed research:
{seed_context}

Produce a complete Outline."""


async def planner_node(state: ContentState) -> dict:
    check_token_budget(state.token_usage)
    logger.info("[planner] generating outline")

    seed_context = "\n".join(
        f"- [{s.title}]({s.url}): {s.snippet}" for s in state.sources[:5]
    ) or "No seed sources available."

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", _HUMAN)]
    )
    model = build_model(settings.primary_model, temperature=0.2)
    structured_model = model.with_structured_output(Outline)

    chain = prompt | structured_model

    async with node_timeout():
        result: Outline = await chain.ainvoke(
            {
                "topic": state.topic,
                "content_type": state.content_type,
                "seed_context": seed_context,
            }
        )

    # Token tracking — metadata available on the raw model call
    logger.info(
        f"[planner] outline ready: {result.title!r} "
        f"sections={len(result.sections)} kw={result.primary_keyword!r}"
    )

    return {"outline": result}
