"""
Writer node — LCEL chain that generates a full article Draft.

Loss proxy: maximise P(draft | outline, research, prior_review)
  where prior_review injects Reviewer critique on revision iterations.

$\mathcal{L}_{\text{writer}} = -\log P(\hat{d} \mid o, r, c)$

Input:  state.outline, state.sources, state.run_metadata["research_summary"],
        state.review (if revision_iteration > 0)
Output: state.draft, state.revision_iteration (incremented)
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from config import settings
from nodes.base import (
    accumulate_tokens,
    build_model,
    check_token_budget,
    node_timeout,
)
from state import ContentState, Draft, TokenUsage

_SYSTEM = """You are an expert content writer and SEO specialist.
Write a complete, publication-quality article in Markdown.
Requirements:
- Follow the outline exactly (same sections, same order).
- Target word count: {target_word_count} words (±10%).
- Tone: {tone}.
- Naturally incorporate the primary keyword ({primary_keyword}) and secondary keywords ({secondary_keywords}).
- Cite every factual claim with an inline Markdown link [source](url).
- No filler sentences. Every paragraph must add value.
- Output only the article body — no preamble, no commentary."""

_HUMAN_FIRST = """Outline:
{outline}

Research brief:
{research_summary}

Write the full article now."""

_HUMAN_REVISION = """Outline:
{outline}

Research brief:
{research_summary}

Previous draft (score: {prev_score}/10):
{prev_draft}

Reviewer critique:
{critique}

Rewrite the article addressing ALL critique points. Improve quality score above 8.0."""


async def writer_node(state: ContentState) -> dict:
    check_token_budget(state.token_usage)

    if state.outline is None:
        return {"error": "writer_node called before outline was generated"}

    logger.info(f"[writer] iteration={state.revision_iteration}")

    outline_str = (
        f"Title: {state.outline.title}\n"
        f"Meta: {state.outline.meta_description}\n"
        f"Sections:\n"
        + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(state.outline.sections))
    )
    research_summary = state.run_metadata.get("research_summary", "No research available.")
    sources_block = "\n".join(
        f"- [{s.title}]({s.url})" for s in state.sources[:12]
    )

    system_prompt = _SYSTEM.format(
        target_word_count=state.outline.target_word_count,
        tone=state.outline.tone,
        primary_keyword=state.outline.primary_keyword,
        secondary_keywords=", ".join(state.outline.secondary_keywords),
    )

    if state.revision_iteration == 0 or state.review is None:
        human_prompt = _HUMAN_FIRST.format(
            outline=outline_str,
            research_summary=research_summary + "\n\nAvailable sources:\n" + sources_block,
        )
    else:
        critique = "\n".join(
            [f"Issues: {', '.join(state.review.issues)}"]
            + [f"Suggestions: {', '.join(state.review.suggestions)}"]
        )
        human_prompt = _HUMAN_REVISION.format(
            outline=outline_str,
            research_summary=research_summary + "\n\nAvailable sources:\n" + sources_block,
            prev_draft=state.draft.body if state.draft else "",
            prev_score=state.review.quality_score if state.review else "N/A",
            critique=critique,
        )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)]
    )
    model = build_model(settings.primary_model, temperature=0.6, max_tokens=6000)
    structured_model = model.with_structured_output(Draft)
    chain = prompt | structured_model

    async with node_timeout(seconds=settings.node_timeout_seconds * 2):
        draft: Draft = await chain.ainvoke({})

    # Ensure word count is set
    if draft.word_count == 0:
        draft.word_count = len(draft.body.split())

    logger.info(
        f"[writer] draft ready: {draft.word_count} words "
        f"iter={state.revision_iteration}"
    )

    return {
        "draft": draft,
        "revision_iteration": state.revision_iteration + 1,
    }
