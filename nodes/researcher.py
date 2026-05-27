"""
Researcher node — ReAct agent that gathers cited sources for every section
in the Outline, then returns enriched sources and a research summary.

Uses LangChain's `create_react_agent` (LangGraph 1.0 compatible) with
Tavily search + Firecrawl scraping tools, capped at 12 tool steps.

$\text{Research}_i = \text{WebSearch}(s_i) \cup \text{Scrape}(\text{top\_url}_i)$
for each section $s_i$ in Outline.sections.

Input:  state.outline, state.sources
Output: state.sources (appended), state.run_metadata["research_summary"]
"""
from __future__ import annotations

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent as lg_create_react_agent
from loguru import logger

from config import settings
from nodes.base import build_model, check_token_budget, node_timeout
from state import ContentState, Source
from tools.search import scrape_url, web_search

_RESEARCHER_SYSTEM = """You are a meticulous research agent for a content automation engine.
Your job: gather factual, cited information for an article outline.
For each section in the outline, search the web and scrape the top result.
Synthesise findings into a concise research brief.
Always cite sources by URL.
Stop after you have covered all sections or after 12 tool calls — whichever comes first."""


async def researcher_node(state: ContentState) -> dict:
    check_token_budget(state.token_usage)

    if state.outline is None:
        return {"error": "researcher_node called before outline was generated"}

    sections_str = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(state.outline.sections)
    )
    query = (
        f"Article: '{state.outline.title}'\n"
        f"Primary keyword: {state.outline.primary_keyword}\n"
        f"Sections to research:\n{sections_str}\n\n"
        "Search each section, scrape the most relevant URL per section, "
        "and compile a research brief with citations."
    )

    model = build_model(settings.primary_model, temperature=0.1)
    tools = [web_search, scrape_url]

    # LangGraph prebuilt react agent (LangGraph 1.0 pattern)
    agent = lg_create_react_agent(
        model,
        tools=tools,
        prompt=_RESEARCHER_SYSTEM,
    )

    async with node_timeout(seconds=settings.node_timeout_seconds * 3):
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": 14},
        )

    # Extract final AI message as research summary
    final_msg = result["messages"][-1]
    research_summary: str = (
        final_msg.content if hasattr(final_msg, "content") else str(final_msg)
    )

    # Parse any URLs from tool call results and add as sources
    new_sources: list[Source] = []
    for msg in result["messages"]:
        if hasattr(msg, "artifact") and isinstance(msg.artifact, list):
            for item in msg.artifact:
                if isinstance(item, dict) and item.get("url"):
                    new_sources.append(
                        Source(
                            url=item["url"],
                            title=item.get("title", item["url"]),
                            snippet=item.get("content", "")[:400],
                            relevance_score=float(item.get("score", 0.7)),
                        )
                    )

    logger.info(
        f"[researcher] summary={len(research_summary)} chars "
        f"new_sources={len(new_sources)}"
    )

    return {
        "sources": new_sources,
        "run_metadata": {
            **state.run_metadata,
            "research_summary": research_summary,
        },
    }
