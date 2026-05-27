"""
Ingestor node — validates the incoming brief, enriches it with a quick
web-search context snapshot, and populates state.run_metadata.

Input:  state.topic, state.content_type
Output: state.sources (initial seed), state.run_metadata
"""
from __future__ import annotations

from loguru import logger

from state import ContentState, Source
from nodes.base import node_timeout
from tools.search import web_search


async def ingestor_node(state: ContentState) -> dict:
    """
    1. Validate that topic is non-empty.
    2. Run a seed Tavily search (3 results) for initial context.
    3. Populate run_metadata with content_type and topic.
    """
    if not state.topic.strip():
        return {"error": "topic is required — cannot proceed with empty brief."}

    logger.info(f"[ingestor] topic={state.topic!r} type={state.content_type}")

    seed_results: list[dict] = []
    async with node_timeout():
        try:
            seed_results = await web_search.ainvoke(
                {"query": state.topic, "max_results": 3}
            )
        except Exception as exc:
            logger.warning(f"[ingestor] seed search failed: {exc} — continuing without seed sources")

    sources = [
        Source(
            url=r["url"],
            title=r["title"],
            snippet=r["content"][:400],
            relevance_score=float(r.get("score", 0.5)),
        )
        for r in seed_results
        if r.get("url")
    ]

    return {
        "sources": sources,
        "run_metadata": {
            "topic": state.topic,
            "content_type": state.content_type,
            "target_platform": state.target_platform,
        },
    }
