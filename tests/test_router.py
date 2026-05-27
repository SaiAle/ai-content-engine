"""
tests/test_router.py

Unit tests for review_router — the conditional edge that routes from
reviewer_node to either "writer" (revise), "optimizer" (accept), or
"error" / "__end__" (terminal error).

No external I/O. All tests are synchronous because review_router is a
plain function, not a coroutine.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from state import ContentState, ReviewResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_review(verdict: str, score: float = 7.0) -> ReviewResult:
    return ReviewResult(
        verdict=verdict,  # type: ignore[arg-type]
        quality_score=score,
        issues=["Issue A"] if verdict == "REVISE" else [],
        suggestions=[],
        fact_check_passed=(verdict == "ACCEPT"),
    )


def _state_with_review(
    verdict: str,
    iteration: int,
    max_iter: int = 3,
    error: str = "",
    score: float = 7.0,
) -> ContentState:
    review = _make_review(verdict, score) if verdict else None
    return ContentState(
        topic="AI Agents",
        review=review,
        revision_iteration=iteration,
        error=error,
    )


# ---------------------------------------------------------------------------
# review_router tests
# ---------------------------------------------------------------------------


class TestReviewRouter:
    def test_router_revise_under_limit_returns_writer(self):
        """
        REVISE verdict with iteration < max_revision_iterations must route to writer.
        Default max_revision_iterations = 3; iteration = 1 → should loop back.
        """
        from graph import review_router

        state = _state_with_review("REVISE", iteration=1)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert result == "writer"

    def test_router_revise_iteration_zero_returns_writer(self):
        """REVISE at iteration 0 (first review) must route to writer."""
        from graph import review_router

        state = _state_with_review("REVISE", iteration=0)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert result == "writer"

    def test_router_revise_iteration_2_under_max_3_returns_writer(self):
        """REVISE at iteration 2 with max=3 still has room for one more revision."""
        from graph import review_router

        state = _state_with_review("REVISE", iteration=2)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert result == "writer"

    def test_router_accept_returns_optimizer(self):
        """ACCEPT verdict regardless of iteration count must route to optimizer."""
        from graph import review_router

        state = _state_with_review("ACCEPT", iteration=0, score=9.0)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert result == "optimizer"

    def test_router_accept_high_iteration_returns_optimizer(self):
        """ACCEPT after many revisions still routes to optimizer."""
        from graph import review_router

        state = _state_with_review("ACCEPT", iteration=3, score=8.0)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert result == "optimizer"

    def test_router_max_iterations_reached_returns_optimizer(self):
        """
        REVISE with iteration == max_revision_iterations must route to optimizer
        (force-accept to prevent infinite loop).
        """
        from graph import review_router

        state = _state_with_review("REVISE", iteration=3)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert result == "optimizer"

    def test_router_revise_iteration_exceeds_max_returns_optimizer(self):
        """iteration > max (edge case from external state injection) → optimizer."""
        from graph import review_router

        state = _state_with_review("REVISE", iteration=5)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert result == "optimizer"

    def test_router_error_state_returns_error(self):
        """Any non-empty state.error must short-circuit to error regardless of review."""
        from graph import review_router

        state = _state_with_review(
            "REVISE", iteration=1, error="Token budget exhausted"
        )

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert result == "error"

    def test_router_error_with_accept_verdict_still_returns_error(self):
        """Error flag takes priority even when verdict is ACCEPT."""
        from graph import review_router

        state = _state_with_review("ACCEPT", iteration=0, error="LLM timeout")

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert result == "error"

    def test_router_none_review_falls_through_to_optimizer(self):
        """None review (node skipped) must default to optimizer, not crash."""
        from graph import review_router

        state = ContentState(topic="AI Agents", review=None, revision_iteration=0)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert result == "optimizer"

    def test_router_custom_max_iterations_respected(self):
        """When max_revision_iterations=1, iteration=1 with REVISE → optimizer."""
        from graph import review_router

        state = _state_with_review("REVISE", iteration=1)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 1
            result = review_router(state)

        assert result == "optimizer"

    def test_router_custom_max_iterations_allows_first_revision(self):
        """When max_revision_iterations=2, iteration=1 with REVISE → writer."""
        from graph import review_router

        state = _state_with_review("REVISE", iteration=1)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 2
            result = review_router(state)

        assert result == "writer"

    def test_router_return_type_is_string(self):
        """review_router always returns a plain str (not None, not enum)."""
        from graph import review_router

        state = _state_with_review("ACCEPT", iteration=0)

        with patch("graph.settings") as mock_settings:
            mock_settings.max_revision_iterations = 3
            result = review_router(state)

        assert isinstance(result, str)
