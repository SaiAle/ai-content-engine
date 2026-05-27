"""
LangGraph state schema.

Nested namespace pattern — each agent owns a sub-model.
Top-level ContentState is the single source of truth for the entire graph.

Reducers:
  • messages     → add_messages  (id-based de-dup, append)
  • sources      → operator.add  (append-only list)
  • issues       → operator.add
  • everything else → last-write-wins (default LangGraph behaviour)
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Sub-state models
# ─────────────────────────────────────────────────────────────────────────────

class Source(BaseModel):
    url: str
    title: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class Outline(BaseModel):
    title: str
    meta_description: str = Field(max_length=160)
    sections: list[str] = Field(min_length=3)
    target_word_count: int = Field(ge=300, le=5000)
    target_audience: str
    tone: Literal["informative", "persuasive", "conversational", "technical"]
    primary_keyword: str
    secondary_keywords: list[str]


class Draft(BaseModel):
    title: str
    body: str = Field(description="Full article body in Markdown")
    word_count: int
    sources_used: list[str] = Field(description="List of source URLs cited")


class ReviewResult(BaseModel):
    verdict: Literal["ACCEPT", "REVISE"]
    quality_score: float = Field(ge=0.0, le=10.0)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    fact_check_passed: bool


class SEOResult(BaseModel):
    seo_score: float = Field(ge=0.0, le=100.0)
    optimized_title: str
    optimized_meta: str = Field(max_length=160)
    optimized_body: str
    slug: str


class PublishResult(BaseModel):
    platform: str
    post_id: str
    url: str
    idempotency_key: str


class TokenUsage(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Top-level graph state
# ─────────────────────────────────────────────────────────────────────────────

class ContentState(BaseModel):
    """
    Single shared state object for the entire content-automation graph.
    LangGraph merges partial dict returns from each node into this model.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    topic: str = Field(default="", description="Content topic / brief")
    content_type: Literal[
        "blog_post", "newsletter", "social_thread", "product_description"
    ] = Field(default="blog_post")
    target_platform: str = Field(default="wordpress")

    # ── Conversation history (de-duped append) ────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list
    )

    # ── Research ──────────────────────────────────────────────────────────────
    sources: Annotated[list[Source], operator.add] = Field(default_factory=list)
    outline: Outline | None = None

    # ── Generation ────────────────────────────────────────────────────────────
    draft: Draft | None = None
    revision_iteration: int = 0

    # ── Review ────────────────────────────────────────────────────────────────
    review: ReviewResult | None = None
    issues: Annotated[list[str], operator.add] = Field(default_factory=list)

    # ── SEO / optimisation ────────────────────────────────────────────────────
    seo_result: SEOResult | None = None

    # ── Human-in-the-loop ─────────────────────────────────────────────────────
    human_approved: bool | None = None
    human_feedback: str = ""

    # ── Publishing ────────────────────────────────────────────────────────────
    publish_results: list[PublishResult] = Field(default_factory=list)

    # ── Budget & observability ────────────────────────────────────────────────
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    run_metadata: dict[str, Any] = Field(default_factory=dict)

    # ── Terminal flag ─────────────────────────────────────────────────────────
    error: str = ""
    completed: bool = False
