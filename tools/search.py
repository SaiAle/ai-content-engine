"""
Search & retrieval tools.
  • web_search     — Tavily research-grade search
  • scrape_url     — Firecrawl clean-markdown scrape
  • vector_search  — pgvector similarity retrieval
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import tool
from loguru import logger

from config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Web search (Tavily)
# ─────────────────────────────────────────────────────────────────────────────

@tool
async def web_search(query: str, max_results: int = 6) -> list[dict[str, str]]:
    """
    Run a research-grade web search via Tavily.
    Returns a list of {url, title, content, score} dicts.
    Raises RuntimeError on API failure.
    """
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY not set — cannot run web search.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
            },
        )
        resp.raise_for_status()

    data = resp.json()
    results: list[dict[str, str]] = [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "score": str(r.get("score", 0.0)),
        }
        for r in data.get("results", [])
    ]
    logger.debug(f"web_search({query!r}) → {len(results)} results")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# URL scraping (Firecrawl)
# ─────────────────────────────────────────────────────────────────────────────

@tool
async def scrape_url(url: str) -> str:
    """
    Scrape a URL via Firecrawl and return clean Markdown content.
    Raises RuntimeError if the API key is missing or the request fails.
    """
    if not settings.firecrawl_api_key:
        raise RuntimeError("FIRECRAWL_API_KEY not set — cannot scrape URL.")

    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={
                "Authorization": f"Bearer {settings.firecrawl_api_key}",
                "Content-Type": "application/json",
            },
            json={"url": url, "formats": ["markdown"]},
        )
        resp.raise_for_status()

    data = resp.json()
    markdown: str = data.get("data", {}).get("markdown", "")
    logger.debug(f"scrape_url({url!r}) → {len(markdown)} chars")
    return markdown
