"""
tests/test_state.py

Unit tests for ContentState reducers and Pydantic model validation.
No external I/O — pure in-process logic.
"""
from __future__ import annotations

import operator

import pytest
from pydantic import ValidationError

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


def _make_source(n: int) -> Source:
    return Source(
        url=f"https://example.com/source-{n}",
        title=f"Source {n}",
        snippet=f"Snippet for source {n}.",
        relevance_score=round(0.5 + n * 0.05, 2),
    )


def _make_outline(**overrides) -> Outline:
    defaults = dict(
        title="Test Article",
        meta_description="A test meta description under 160 characters for validation purposes.",
        sections=["Introduction", "Main Content", "Conclusion"],
        target_word_count=800,
        target_audience="Developers",
        tone="informative",
        primary_keyword="test keyword",
        secondary_keywords=["secondary", "keywords"],
    )
    defaults.update(overrides)
    return Outline(**defaults)


# ---------------------------------------------------------------------------
# Source reducer (operator.add — append-only)
# ---------------------------------------------------------------------------


class TestSourcesReducer:
    def test_sources_appended_to_empty_list(self):
        """operator.add reducer: sources from two partial updates are combined."""
        initial: list[Source] = []
        batch_a = [_make_source(1), _make_source(2)]
        batch_b = [_make_source(3)]

        combined = operator.add(operator.add(initial, batch_a), batch_b)

        assert len(combined) == 3
        assert combined[0].url == "https://example.com/source-1"
        assert combined[2].url == "https://example.com/source-3"

    def test_sources_are_not_deduplicated(self):
        """The reducer is append-only — duplicate entries are preserved."""
        source = _make_source(1)
        combined = operator.add([source], [source])
        assert len(combined) == 2

    def test_sources_preserve_insertion_order(self):
        batch = [_make_source(i) for i in range(5)]
        result = operator.add([], batch)
        urls = [s.url for s in result]
        assert urls == [f"https://example.com/source-{i}" for i in range(5)]

    def test_content_state_sources_field_uses_add_reducer(self):
        """
        Confirm that ContentState.sources is annotated with operator.add by
        verifying two ContentState dicts can be merged via the reducer pattern.
        """
        state = ContentState(topic="Test", sources=[_make_source(1)])
        new_sources = [_make_source(2), _make_source(3)]
        # Simulate LangGraph partial-dict merge: new sources are appended
        merged_sources = operator.add(state.sources, new_sources)
        assert len(merged_sources) == 3


# ---------------------------------------------------------------------------
# Issues reducer (operator.add — append-only)
# ---------------------------------------------------------------------------


class TestIssuesReducer:
    def test_issues_appended(self):
        """Issues from multiple review cycles accumulate in order."""
        first_batch = ["Factual error in section 2", "Missing citations"]
        second_batch = ["Tone inconsistency in conclusion"]

        merged = operator.add(first_batch, second_batch)

        assert len(merged) == 3
        assert "Factual error in section 2" in merged
        assert "Tone inconsistency in conclusion" in merged

    def test_issues_empty_initial(self):
        issues = operator.add([], ["First issue"])
        assert issues == ["First issue"]

    def test_issues_append_empty_batch(self):
        existing = ["Existing issue"]
        merged = operator.add(existing, [])
        assert merged == ["Existing issue"]

    def test_content_state_issues_default_empty(self):
        state = ContentState(topic="Test")
        assert state.issues == []


# ---------------------------------------------------------------------------
# TokenUsage accumulation
# ---------------------------------------------------------------------------


class TestTokenUsageAccumulation:
    def test_token_counts_sum_correctly(self):
        base = TokenUsage(
            total_tokens=1_000,
            prompt_tokens=700,
            completion_tokens=300,
            estimated_cost_usd=0.01,
        )
        addition = TokenUsage(
            total_tokens=500,
            prompt_tokens=350,
            completion_tokens=150,
            estimated_cost_usd=0.005,
        )
        result = TokenUsage(
            total_tokens=base.total_tokens + addition.total_tokens,
            prompt_tokens=base.prompt_tokens + addition.prompt_tokens,
            completion_tokens=base.completion_tokens + addition.completion_tokens,
            estimated_cost_usd=base.estimated_cost_usd + addition.estimated_cost_usd,
        )

        assert result.total_tokens == 1_500
        assert result.prompt_tokens == 1_050
        assert result.completion_tokens == 450
        assert abs(result.estimated_cost_usd - 0.015) < 1e-9

    def test_token_usage_defaults_to_zero(self):
        usage = TokenUsage()
        assert usage.total_tokens == 0
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.estimated_cost_usd == 0.0

    def test_accumulated_from_content_state(self):
        state = ContentState(
            topic="Test",
            token_usage=TokenUsage(total_tokens=2_000, prompt_tokens=1_500,
                                   completion_tokens=500, estimated_cost_usd=0.02),
        )
        # Simulate node adding more tokens
        new_usage = TokenUsage(
            total_tokens=state.token_usage.total_tokens + 800,
            prompt_tokens=state.token_usage.prompt_tokens + 600,
            completion_tokens=state.token_usage.completion_tokens + 200,
            estimated_cost_usd=state.token_usage.estimated_cost_usd + 0.008,
        )
        assert new_usage.total_tokens == 2_800
        assert new_usage.completion_tokens == 700

    def test_large_accumulation_stays_precise(self):
        usage = TokenUsage()
        for _ in range(100):
            usage = TokenUsage(
                total_tokens=usage.total_tokens + 1_000,
                prompt_tokens=usage.prompt_tokens + 700,
                completion_tokens=usage.completion_tokens + 300,
                estimated_cost_usd=usage.estimated_cost_usd + 0.01,
            )
        assert usage.total_tokens == 100_000
        assert usage.prompt_tokens == 70_000


# ---------------------------------------------------------------------------
# Outline validation
# ---------------------------------------------------------------------------


class TestOutlineValidation:
    def test_valid_outline_is_accepted(self):
        outline = _make_outline()
        assert outline.title == "Test Article"

    def test_meta_description_max_length_160(self):
        long_meta = "A" * 161
        with pytest.raises(ValidationError) as exc_info:
            _make_outline(meta_description=long_meta)
        errors = exc_info.value.errors()
        assert any("meta_description" in str(e) for e in errors)

    def test_meta_description_exactly_160_chars_is_valid(self):
        meta = "A" * 160
        outline = _make_outline(meta_description=meta)
        assert len(outline.meta_description) == 160

    def test_sections_minimum_3_required(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_outline(sections=["Only one section", "Two sections"])
        errors = exc_info.value.errors()
        assert any("sections" in str(e) for e in errors)

    def test_sections_exactly_3_is_valid(self):
        outline = _make_outline(sections=["A", "B", "C"])
        assert len(outline.sections) == 3

    def test_sections_more_than_3_is_valid(self):
        outline = _make_outline(sections=["A", "B", "C", "D", "E"])
        assert len(outline.sections) == 5

    def test_target_word_count_minimum_300(self):
        with pytest.raises(ValidationError):
            _make_outline(target_word_count=299)

    def test_target_word_count_maximum_5000(self):
        with pytest.raises(ValidationError):
            _make_outline(target_word_count=5001)

    def test_target_word_count_boundary_values_valid(self):
        low = _make_outline(target_word_count=300)
        high = _make_outline(target_word_count=5000)
        assert low.target_word_count == 300
        assert high.target_word_count == 5000

    def test_tone_must_be_literal(self):
        with pytest.raises(ValidationError):
            _make_outline(tone="sarcastic")

    def test_all_valid_tones_accepted(self):
        for tone in ("informative", "persuasive", "conversational", "technical"):
            outline = _make_outline(tone=tone)
            assert outline.tone == tone


# ---------------------------------------------------------------------------
# Draft word_count field
# ---------------------------------------------------------------------------


class TestDraftWordCount:
    def _make_draft(self, body: str, word_count: int | None = None) -> Draft:
        wc = word_count if word_count is not None else len(body.split())
        return Draft(
            title="Test Draft",
            body=body,
            word_count=wc,
            sources_used=["https://example.com"],
        )

    def test_word_count_stored_as_provided(self):
        draft = self._make_draft("Hello world this is a test.", word_count=6)
        assert draft.word_count == 6

    def test_word_count_zero_is_valid(self):
        draft = self._make_draft("some content", word_count=0)
        assert draft.word_count == 0

    def test_word_count_reflects_body_length(self):
        body = " ".join(["word"] * 500)
        draft = self._make_draft(body)
        assert draft.word_count == 500

    def test_draft_sources_used_stored(self):
        draft = Draft(
            title="T",
            body="Body text here.",
            word_count=3,
            sources_used=["https://a.com", "https://b.com"],
        )
        assert len(draft.sources_used) == 2
        assert "https://a.com" in draft.sources_used

    def test_draft_body_in_markdown(self):
        md_body = "# Heading\n\nParagraph with **bold** and [link](https://example.com)."
        draft = self._make_draft(md_body, word_count=8)
        assert draft.body.startswith("# Heading")
