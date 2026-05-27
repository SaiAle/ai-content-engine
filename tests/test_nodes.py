"""
tests/test_nodes.py

Integration-style unit tests for each graph node.
All LangChain models, Tavily, Firecrawl, and external HTTP calls are mocked.
Each test exercises node I/O contracts in isolation.

Mock strategy for LCEL chain nodes (planner, writer, reviewer, optimizer):
  - build_model() returns mock_model (MagicMock)
  - mock_model.with_structured_output() returns structured_model (AsyncMock)
  - ChatPromptTemplate.from_messages() returns mock_prompt (MagicMock)
  - mock_prompt.__or__(structured_model) = structured_model  (via __or__ override)
  - chain = prompt | structured_model  ->  chain IS structured_model
  - structured_model.ainvoke.return_value = <expected Pydantic object>

This avoids the MagicMock.__or__ / pytest-asyncio event-loop interaction that
causes AsyncMock return values to be ignored when mock_chain is a plain MagicMock.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from state import (
    ContentState,
    Draft,
    Outline,
    ReviewResult,
    SEOResult,
    Source,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> ContentState:
    defaults = dict(topic="AI Agents", content_type="blog_post")
    defaults.update(overrides)
    return ContentState(**defaults)


def _make_outline() -> Outline:
    return Outline(
        title="AI Agents: A Complete Guide",
        meta_description="Everything you need to know about AI agents in one comprehensive guide.",
        sections=[
            "What Are AI Agents?",
            "How They Work",
            "Use Cases",
            "Implementation",
            "Future",
        ],
        target_word_count=1200,
        target_audience="Developers",
        tone="informative",
        primary_keyword="AI agents",
        secondary_keywords=["LLM agents", "autonomous agents"],
    )


def _make_draft() -> Draft:
    body = (
        "# AI Agents: A Complete Guide\n\n"
        "AI agents are autonomous systems powered by large language models. "
        "They combine reasoning, memory, and tool use to complete complex tasks.\n\n"
        "## How They Work\n\n"
        "Agents perceive input, reason with an LLM, and act via tools.\n"
    )
    return Draft(
        title="AI Agents: A Complete Guide",
        body=body,
        word_count=len(body.split()),
        sources_used=["https://example.com/agents"],
    )


def _chain_mock(return_value):
    """
    Return an AsyncMock that acts as the compiled LCEL chain.

    Wired so that:
        chain = prompt | structured_model  ->  this AsyncMock
        await chain.ainvoke(...)           ->  return_value
    """
    chain = AsyncMock()
    chain.ainvoke = AsyncMock(return_value=return_value)
    return chain


def _mock_model_and_prompt(chain):
    """
    Return (mock_model, mock_prompt_cls) so that inside a node:

        prompt = ChatPromptTemplate.from_messages(...)   -> mock_prompt
        model  = build_model(...)                        -> mock_model
        sm     = model.with_structured_output(X)         -> chain
        chain  = prompt | sm                             -> chain (via __or__)
    """
    mock_model = MagicMock()
    mock_model.with_structured_output = MagicMock(return_value=chain)

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=chain)

    mock_prompt_cls = MagicMock()
    mock_prompt_cls.from_messages = MagicMock(return_value=mock_prompt)

    return mock_model, mock_prompt_cls


def _timeout_mock():
    """Return a ready-to-use async context-manager mock for node_timeout()."""
    mt = MagicMock()
    mt.return_value.__aenter__ = AsyncMock(return_value=None)
    mt.return_value.__aexit__ = AsyncMock(return_value=False)
    return mt


# ---------------------------------------------------------------------------
# ingestor_node
# ---------------------------------------------------------------------------


class TestIngestorNode:
    async def test_ingestor_empty_topic_returns_error(self):
        """Empty topic must set state.error immediately without calling search."""
        from nodes.ingestor import ingestor_node

        state = _base_state(topic="   ")

        with patch("nodes.ingestor.web_search") as mock_search:
            result = await ingestor_node(state)

        assert "error" in result
        assert result["error"]
        mock_search.ainvoke.assert_not_called()

    async def test_ingestor_with_topic_calls_web_search_and_returns_sources(self):
        """Valid topic triggers web_search and maps results to Source objects."""
        from nodes.ingestor import ingestor_node

        state = _base_state(topic="AI Agents")
        fake_results = [
            {
                "url": "https://example.com/agents",
                "title": "Intro to AI Agents",
                "content": "AI agents use LLMs to complete tasks autonomously.",
                "score": "0.92",
            },
            {
                "url": "https://example.com/llm-tools",
                "title": "LLM Tool Use",
                "content": "Tool-augmented language models expand agent capabilities.",
                "score": "0.85",
            },
        ]

        with patch("nodes.ingestor.node_timeout") as mock_timeout, \
             patch("nodes.ingestor.web_search") as mock_search:
            mock_timeout.return_value.__aenter__ = AsyncMock(return_value=None)
            mock_timeout.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_search.ainvoke = AsyncMock(return_value=fake_results)
            result = await ingestor_node(state)

        assert "sources" in result
        assert len(result["sources"]) == 2
        assert result["sources"][0].url == "https://example.com/agents"
        assert result["sources"][0].relevance_score == 0.92
        assert "run_metadata" in result
        assert result["run_metadata"]["topic"] == "AI Agents"

    async def test_ingestor_search_failure_returns_empty_sources(self):
        """If web_search raises, ingestor gracefully returns empty sources."""
        from nodes.ingestor import ingestor_node

        state = _base_state(topic="AI Agents")

        with patch("nodes.ingestor.node_timeout") as mock_timeout, \
             patch("nodes.ingestor.web_search") as mock_search:
            mock_timeout.return_value.__aenter__ = AsyncMock(return_value=None)
            mock_timeout.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_search.ainvoke = AsyncMock(
                side_effect=RuntimeError("Tavily unavailable")
            )
            result = await ingestor_node(state)

        assert "error" not in result or not result.get("error")
        assert result["sources"] == []


# ---------------------------------------------------------------------------
# planner_node
# ---------------------------------------------------------------------------


class TestPlannerNode:
    async def test_planner_creates_outline(self):
        """planner_node calls the structured-output chain and sets state.outline."""
        from nodes.planner import planner_node

        expected_outline = _make_outline()
        state = ContentState(
            topic="AI Agents",
            content_type="blog_post",
            sources=[
                Source(
                    url="https://example.com",
                    title="Example",
                    snippet="Snippet",
                    relevance_score=0.8,
                )
            ],
        )
        chain = _chain_mock(expected_outline)
        mock_model, mock_prompt_cls = _mock_model_and_prompt(chain)

        with patch("nodes.planner.build_model", return_value=mock_model), \
             patch("nodes.planner.ChatPromptTemplate", mock_prompt_cls), \
             patch("nodes.planner.node_timeout", _timeout_mock()), \
             patch("nodes.planner.check_token_budget"):
            result = await planner_node(state)

        assert "outline" in result
        assert result["outline"].title == expected_outline.title
        assert result["outline"].primary_keyword == "AI agents"
        assert len(result["outline"].sections) >= 3


# ---------------------------------------------------------------------------
# writer_node
# ---------------------------------------------------------------------------


class TestWriterNode:
    async def test_writer_first_iteration_returns_draft(self, sample_state):
        """On iteration 0, writer produces a draft and sets revision_iteration=1."""
        from nodes.writer import writer_node

        state = sample_state.model_copy(
            update={"revision_iteration": 0, "review": None, "draft": None}
        )
        expected_draft = _make_draft()
        chain = _chain_mock(expected_draft)
        mock_model, mock_prompt_cls = _mock_model_and_prompt(chain)

        with patch("nodes.writer.build_model", return_value=mock_model), \
             patch("nodes.writer.ChatPromptTemplate", mock_prompt_cls), \
             patch("nodes.writer.node_timeout", _timeout_mock()), \
             patch("nodes.writer.check_token_budget"):
            result = await writer_node(state)

        assert "draft" in result
        assert result["draft"].title == expected_draft.title
        assert result["revision_iteration"] == 1

    async def test_writer_revision(self, sample_state):
        """On iteration > 0 with prior review, revision_iteration increments to 2."""
        from nodes.writer import writer_node

        review = ReviewResult(
            verdict="REVISE",
            quality_score=5.5,
            issues=["Missing citations in section 2", "Weak conclusion"],
            suggestions=["Add more concrete examples"],
            fact_check_passed=False,
        )
        state = sample_state.model_copy(
            update={
                "revision_iteration": 1,
                "review": review,
                "draft": _make_draft(),
            }
        )
        revised_draft = _make_draft()
        chain = _chain_mock(revised_draft)
        mock_model, mock_prompt_cls = _mock_model_and_prompt(chain)

        with patch("nodes.writer.build_model", return_value=mock_model), \
             patch("nodes.writer.ChatPromptTemplate", mock_prompt_cls), \
             patch("nodes.writer.node_timeout", _timeout_mock()), \
             patch("nodes.writer.check_token_budget"):
            result = await writer_node(state)

        assert "draft" in result
        assert result["revision_iteration"] == 2

    async def test_writer_missing_outline_returns_error(self):
        """writer_node with no outline in state must return an error."""
        from nodes.writer import writer_node

        state = ContentState(topic="AI Agents", outline=None)

        with patch("nodes.writer.check_token_budget"):
            result = await writer_node(state)

        assert "error" in result
        assert result["error"]


# ---------------------------------------------------------------------------
# reviewer_node
# ---------------------------------------------------------------------------


class TestReviewerNode:
    async def test_reviewer_accept(self, sample_state):
        """reviewer_node maps a high-quality draft to ACCEPT ReviewResult."""
        from nodes.reviewer import reviewer_node

        expected_review = ReviewResult(
            verdict="ACCEPT",
            quality_score=8.5,
            issues=[],
            suggestions=["Consider adding a comparison table"],
            fact_check_passed=True,
        )
        chain = _chain_mock(expected_review)
        mock_model, mock_prompt_cls = _mock_model_and_prompt(chain)

        with patch("nodes.reviewer.build_model", return_value=mock_model), \
             patch("nodes.reviewer.ChatPromptTemplate", mock_prompt_cls), \
             patch("nodes.reviewer.node_timeout", _timeout_mock()), \
             patch("nodes.reviewer.check_token_budget"):
            result = await reviewer_node(sample_state)

        assert "review" in result
        assert result["review"].verdict == "ACCEPT"
        assert result["review"].quality_score == 8.5
        assert result["review"].fact_check_passed is True

    async def test_reviewer_revise(self, sample_state):
        """reviewer_node maps a low-quality draft to REVISE with issues."""
        from nodes.reviewer import reviewer_node

        expected_review = ReviewResult(
            verdict="REVISE",
            quality_score=5.0,
            issues=["Missing citations", "Factual inaccuracy in section 3"],
            suggestions=["Add more sources", "Verify claim about training data"],
            fact_check_passed=False,
        )
        chain = _chain_mock(expected_review)
        mock_model, mock_prompt_cls = _mock_model_and_prompt(chain)

        with patch("nodes.reviewer.build_model", return_value=mock_model), \
             patch("nodes.reviewer.ChatPromptTemplate", mock_prompt_cls), \
             patch("nodes.reviewer.node_timeout", _timeout_mock()), \
             patch("nodes.reviewer.check_token_budget"):
            result = await reviewer_node(sample_state)

        assert result["review"].verdict == "REVISE"
        assert result["review"].quality_score == 5.0
        assert "issues" in result
        assert len(result["issues"]) == 2

    async def test_reviewer_missing_draft_returns_error(self):
        """reviewer_node with no draft must return an error key."""
        from nodes.reviewer import reviewer_node

        state = ContentState(topic="AI Agents", draft=None, outline=_make_outline())

        with patch("nodes.reviewer.check_token_budget"):
            result = await reviewer_node(state)

        assert "error" in result
        assert result["error"]


# ---------------------------------------------------------------------------
# optimizer_node
# ---------------------------------------------------------------------------


class TestOptimizerNode:
    async def test_optimizer_runs(self, sample_state):
        """optimizer_node calls the SEO chain and stores SEOResult in state."""
        from nodes.optimizer import optimizer_node

        expected_seo = SEOResult(
            seo_score=88.0,
            optimized_title="AI Agents 2025: Complete Architecture & Implementation Guide",
            optimized_meta=(
                "Learn AI agents from scratch. Covers architecture, tools, memory, "
                "and production implementation with code examples."
            ),
            optimized_body=sample_state.draft.body,
            slug="ai-agents-2025-guide",
        )
        chain = _chain_mock(expected_seo)
        mock_model, mock_prompt_cls = _mock_model_and_prompt(chain)

        with patch("nodes.optimizer.build_model", return_value=mock_model), \
             patch("nodes.optimizer.ChatPromptTemplate", mock_prompt_cls), \
             patch("nodes.optimizer.node_timeout", _timeout_mock()), \
             patch("nodes.optimizer.check_token_budget"):
            result = await optimizer_node(sample_state)

        assert "seo_result" in result
        assert result["seo_result"].seo_score == 88.0
        assert result["seo_result"].slug == "ai-agents-2025-guide"

    async def test_optimizer_missing_draft_returns_error(self):
        """optimizer_node without a draft must return an error."""
        from nodes.optimizer import optimizer_node

        state = ContentState(topic="AI Agents", draft=None, outline=_make_outline())

        with patch("nodes.optimizer.check_token_budget"):
            result = await optimizer_node(state)

        assert "error" in result
        assert result["error"]


# ---------------------------------------------------------------------------
# publisher_node
# ---------------------------------------------------------------------------


class TestPublisherNode:
    async def test_publisher_approved(self, sample_state):
        """When human_approved=True, publisher invokes publish_to_wordpress."""
        from nodes.publisher import publisher_node

        state = sample_state.model_copy(update={"human_approved": True})
        fake_result = {
            "platform": "wordpress",
            "post_id": "42",
            "url": "https://test.example.com/ai-agents-guide-2025/",
            "idempotency_key": "abc123def456789012345678",
        }

        with patch("nodes.publisher.publish_to_wordpress") as mock_wp:
            mock_wp.ainvoke = AsyncMock(return_value=fake_result)
            result = await publisher_node(state)

        mock_wp.ainvoke.assert_called_once()
        assert len(result["publish_results"]) == 1
        assert result["publish_results"][0].platform == "wordpress"
        assert result["publish_results"][0].post_id == "42"
        assert result["completed"] is True

    async def test_publisher_rejected(self, sample_state):
        """When human_approved=False, publisher terminates without publishing."""
        from nodes.publisher import publisher_node

        state = sample_state.model_copy(update={"human_approved": False})

        with patch("nodes.publisher.publish_to_wordpress") as mock_wp, \
             patch("nodes.publisher.publish_to_social") as mock_social:
            result = await publisher_node(state)

        mock_wp.ainvoke.assert_not_called()
        mock_social.ainvoke.assert_not_called()
        assert result["completed"] is True
        assert result["run_metadata"]["publish_status"] == "rejected_by_human"

    async def test_publisher_approved_social_platform(self, sample_state):
        """Publisher routes to publish_to_social for non-wordpress platforms."""
        from nodes.publisher import publisher_node

        state = sample_state.model_copy(
            update={"human_approved": True, "target_platform": "linkedin"}
        )
        fake_social = {
            "platform": "linkedin",
            "post_id": "stub-abc123def456",
            "url": "https://linkedin.example.com/posts/abc123def456",
            "idempotency_key": "abc123def456789012345678",
        }

        with patch("nodes.publisher.publish_to_social") as mock_social, \
             patch("nodes.publisher.publish_to_wordpress") as mock_wp:
            mock_social.ainvoke = AsyncMock(return_value=fake_social)
            result = await publisher_node(state)

        mock_wp.ainvoke.assert_not_called()
        mock_social.ainvoke.assert_called_once()
        assert result["publish_results"][0].platform == "linkedin"
        assert result["completed"] is True
