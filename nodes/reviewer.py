"""
Reviewer / Fact-checker node — Reflexion-style critique.

$\text{ReviewResult} = \text{Reviewer}(\hat{d}, \text{sources})$

Uses a *different* model from the Writer to maximise diversity of critique.
Equipped with web_search to ground-truth factual claims.

Input:  state.draft, state.outline, state.sources
Output: state.review, state.issues (appended)
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from config import settings
from nodes.base import build_model, check_token_budget, node_timeout
from state import ContentState, ReviewResult

_SYSTEM = """You are a rigorous fact-checker and editorial reviewer.
Evaluate the article draft against the outline and research sources.

Scoring criteria (0–10):
  • Factual accuracy (3 pts): all claims supported, no hallucinations
  • Outline adherence (2 pts): all sections present and covered
  • Writing quality (2 pts): clarity, flow, no filler
  • SEO / keyword usage (1 pt): primary keyword used naturally 3-5×
  • Citation quality (1 pt): sources cited inline, URLs valid
  • Brand voice (1 pt): tone matches target

Verdict:
  • ACCEPT  — quality_score ≥ 7.5 AND fact_check_passed = true
  • REVISE  — otherwise

Return structured JSON. Be specific in issues and suggestions."""

_HUMAN = """Article title: {title}
Target tone: {tone}
Primary keyword: {primary_keyword}

Draft:
{draft_body}

Known sources (for fact-checking):
{sources}

Evaluate and return ReviewResult JSON."""


async def reviewer_node(state: ContentState) -> dict:
    check_token_budget(state.token_usage)

    if state.draft is None or state.outline is None:
        return {"error": "reviewer_node called before draft was generated"}

    logger.info(f"[reviewer] reviewing draft ({state.draft.word_count} words)")

    sources_block = "\n".join(
        f"- [{s.title}]({s.url}): {s.snippet}" for s in state.sources[:10]
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", _HUMAN)]
    )
    # Use a different model from the Writer for diversity (Reflexion pattern)
    model = build_model(
        settings.review_model,
        fallback_id=settings.primary_model,
        temperature=0.1,
    )
    structured_model = model.with_structured_output(ReviewResult)
    chain = prompt | structured_model

    async with node_timeout():
        review: ReviewResult = await chain.ainvoke(
            {
                "title": state.outline.title,
                "tone": state.outline.tone,
                "primary_keyword": state.outline.primary_keyword,
                "draft_body": state.draft.body,
                "sources": sources_block,
            }
        )

    logger.info(
        f"[reviewer] verdict={review.verdict} score={review.quality_score} "
        f"fact_check={review.fact_check_passed} issues={len(review.issues)}"
    )

    return {
        "review": review,
        "issues": review.issues,
    }
