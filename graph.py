"""
LangGraph StateGraph — AI Content Automation Engine

Topology:
  ingestor → planner → researcher → writer → reviewer
               ↑                              │
               └──── revise (< MAX_ITER) ─────┤
                                              │ accept
                                           optimizer
                                              │
                                       [HITL interrupt]
                                              │ approved
                                           publisher → END

Edges:
  • review_router : REVISE → writer | ACCEPT → optimizer
  • hitl_router   : None   → interrupt | True → publisher | False → publisher (rejected)
"""
from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from loguru import logger

from config import settings
from nodes.ingestor import ingestor_node
from nodes.optimizer import optimizer_node
from nodes.planner import planner_node
from nodes.publisher import publisher_node
from nodes.researcher import researcher_node
from nodes.reviewer import reviewer_node
from nodes.writer import writer_node
from state import ContentState


# ─────────────────────────────────────────────────────────────────────────────
# HITL wrapper — injects interrupt() before publisher
# ─────────────────────────────────────────────────────────────────────────────

async def hitl_node(state: ContentState) -> dict:
    """
    Pauses the graph and surfaces the SEO-optimised draft for human review.
    Resumes when the caller provides Command(resume={"approved": True/False, "feedback": "..."}).
    """
    if state.seo_result is None:
        return {"error": "hitl_node called before SEO optimisation"}

    approval_payload = {
        "title": state.seo_result.optimized_title,
        "meta": state.seo_result.optimized_meta,
        "slug": state.seo_result.slug,
        "seo_score": state.seo_result.seo_score,
        "body_preview": state.seo_result.optimized_body[:500],
        "options": ["approve", "reject"],
    }

    logger.info("[hitl] suspending graph — waiting for human approval")
    human_input: dict = interrupt(approval_payload)

    approved: bool = str(human_input.get("approved", "false")).lower() == "true"
    feedback: str = human_input.get("feedback", "")

    logger.info(f"[hitl] resumed: approved={approved} feedback={feedback!r}")
    return {
        "human_approved": approved,
        "human_feedback": feedback,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge routers
# ─────────────────────────────────────────────────────────────────────────────

def review_router(state: ContentState) -> Literal["writer", "optimizer", "error"]:
    """
    Route after reviewer_node:
      REVISE  + iterations_remaining → writer
      ACCEPT  or max_iterations      → optimizer
      error in state                 → error
    """
    if state.error:
        return "error"

    if state.review is None:
        logger.warning("[router] review is None — falling through to optimizer")
        return "optimizer"

    max_iter = settings.max_revision_iterations
    if state.review.verdict == "REVISE" and state.revision_iteration < max_iter:
        logger.info(
            f"[router] REVISE iter={state.revision_iteration}/{max_iter}"
        )
        return "writer"

    logger.info(
        f"[router] ACCEPT or max_iter reached "
        f"(score={state.review.quality_score} iter={state.revision_iteration})"
    )
    return "optimizer"


def error_router(state: ContentState) -> Literal["__end__"]:
    """Terminal: surface error and halt."""
    logger.error(f"[router] terminal error: {state.error}")
    return END


# ─────────────────────────────────────────────────────────────────────────────
# Graph factory
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None) -> StateGraph:
    """
    Construct and compile the ContentState graph.
    Pass an AsyncPostgresSaver checkpointer for production.
    Pass None (default) for local / testing use.
    """
    builder = StateGraph(ContentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("ingestor", ingestor_node)
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("optimizer", optimizer_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("publisher", publisher_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("ingestor")

    # ── Unconditional edges ───────────────────────────────────────────────────
    builder.add_edge("ingestor", "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_edge("optimizer", "hitl")
    builder.add_edge("hitl", "publisher")
    builder.add_edge("publisher", END)

    # ── Conditional: reviewer → writer | optimizer | error ────────────────────
    builder.add_conditional_edges(
        "reviewer",
        review_router,
        {"writer": "writer", "optimizer": "optimizer", "error": END},
    )

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["publisher"],  # Safety net: always stop before publishing
    )

    logger.info("[graph] compiled successfully")
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: async context manager for production graph with Postgres
# ─────────────────────────────────────────────────────────────────────────────

async def get_production_graph():
    """
    Returns a compiled graph backed by AsyncPostgresSaver.
    Usage:
        async with get_production_graph() as graph:
            await graph.ainvoke(...)
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _ctx():
        async with await AsyncPostgresSaver.from_conn_string(
            settings.database_url.replace("+asyncpg", "")
        ) as checkpointer:
            await checkpointer.setup()
            yield build_graph(checkpointer)

    return _ctx()
