"""
SEO / Brand-voice Optimizer node.

Applies final SEO tweaks: optimised title, meta description, slug,
keyword density enforcement, internal-link placeholders.

Input:  state.draft, state.outline
Output: state.seo_result
"""
from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from config import settings
from nodes.base import build_model, check_token_budget, node_timeout
from state import ContentState, SEOResult

_SYSTEM = """You are an SEO specialist. Optimise the article for search.
Rules:
  1. Title: 50-60 chars, primary keyword near the start.
  2. Meta description: 130-155 chars, primary keyword included, compelling CTA.
  3. Body: ensure primary keyword density 1-2% (count/total_words × 100).
     Insert the keyword naturally if under-represented; remove stuffing if over.
  4. Slug: lowercase, hyphen-separated, ≤6 words, include primary keyword.
  5. Preserve all Markdown formatting, headings, and citation links.
Return SEOResult JSON."""

_HUMAN = """Primary keyword: {primary_keyword}
Secondary keywords: {secondary_keywords}
Current title: {title}
Current meta: {meta_description}

Article body:
{body}

Return optimised SEOResult."""


async def optimizer_node(state: ContentState) -> dict:
    check_token_budget(state.token_usage)

    if state.draft is None or state.outline is None:
        return {"error": "optimizer_node called before draft/outline was available"}

    logger.info("[optimizer] running SEO optimisation")

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", _HUMAN)]
    )
    model = build_model(settings.fast_model, temperature=0.2)
    structured_model = model.with_structured_output(SEOResult)
    chain = prompt | structured_model

    async with node_timeout():
        seo_result: SEOResult = await chain.ainvoke(
            {
                "primary_keyword": state.outline.primary_keyword,
                "secondary_keywords": ", ".join(state.outline.secondary_keywords),
                "title": state.outline.title,
                "meta_description": state.outline.meta_description,
                "body": state.draft.body,
            }
        )

    logger.info(
        f"[optimizer] seo_score={seo_result.seo_score} "
        f"slug={seo_result.slug!r}"
    )
    return {"seo_result": seo_result}
