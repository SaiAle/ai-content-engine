"""
tests/test_api.py

End-to-end tests for the FastAPI service using an httpx.AsyncClient.

All Postgres checkpointer calls and LangGraph graph execution are mocked
so the tests run without any external services.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from jose import jwt

from api.auth import create_access_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_token(username: str = "testuser") -> str:
    """Issue a real JWT using the test secret configured in conftest mock_settings."""
    with patch("api.auth.settings") as s:
        s.jwt_secret_key = "test-secret-key-for-testing-only"
        s.jwt_algorithm = "HS256"
        s.jwt_expire_minutes = 60
        return create_access_token(username)


def _auth_headers(username: str = "testuser") -> dict[str, str]:
    return {"Authorization": f"Bearer {_get_token(username)}"}


def _make_mock_checkpointer():
    mock = AsyncMock()
    mock.setup = AsyncMock()
    mock.aclose = AsyncMock()
    mock.aget_state = AsyncMock(return_value=None)
    return mock


# ---------------------------------------------------------------------------
# Fixture: isolated app client with all external deps mocked
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """
    Fresh httpx.AsyncClient for each test.
    Patches:
      - AsyncPostgresSaver (no real DB)
      - build_graph (no real LangGraph compilation)
      - asyncio.create_task (no background graph execution)
    """
    mock_saver = _make_mock_checkpointer()

    with patch("api.main.AsyncPostgresSaver") as mock_cls:
        mock_cls.from_conn_string = AsyncMock(return_value=mock_saver)

        import api.main as main_module

        main_module._checkpointer = mock_saver

        from api.main import app

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    async def test_health_returns_200_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"

    async def test_health_no_auth_required(self, client):
        """Health endpoint must be publicly accessible."""
        response = await client.get("/health")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /token
# ---------------------------------------------------------------------------


class TestTokenEndpoint:
    async def test_token_issue_returns_access_token(self, client):
        with patch("api.auth.settings") as mock_settings:
            mock_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_settings.jwt_algorithm = "HS256"
            mock_settings.jwt_expire_minutes = 60

            response = await client.post("/token", json={"username": "alice"})

        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    async def test_token_is_valid_jwt(self, client):
        with patch("api.auth.settings") as mock_settings:
            mock_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_settings.jwt_algorithm = "HS256"
            mock_settings.jwt_expire_minutes = 60

            response = await client.post("/token", json={"username": "alice"})
            token = response.json()["access_token"]

        payload = jwt.decode(
            token,
            "test-secret-key-for-testing-only",
            algorithms=["HS256"],
        )
        assert payload["sub"] == "alice"

    async def test_token_missing_username_returns_422(self, client):
        response = await client.post("/token", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /runs  (start run)
# ---------------------------------------------------------------------------


class TestStartRunEndpoint:
    async def test_start_run_unauthorized_returns_401(self, client):
        """Request without Authorization header must be rejected with 401."""
        response = await client.post(
            "/runs",
            json={"topic": "AI Agents in Production", "content_type": "blog_post"},
        )
        assert response.status_code == 401

    async def test_start_run_invalid_token_returns_401(self, client):
        response = await client.post(
            "/runs",
            headers={"Authorization": "Bearer not-a-real-token"},
            json={"topic": "AI Agents", "content_type": "blog_post"},
        )
        assert response.status_code == 401

    async def test_start_run_authorized_returns_202_with_thread_id(self, client):
        """Valid JWT + valid body must return 202 and a thread_id."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={})

        with (
            patch("api.main.build_graph", return_value=mock_graph),
            patch("api.main.asyncio.create_task"),
            patch("api.auth.settings") as mock_auth_settings,
        ):
            mock_auth_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_auth_settings.jwt_algorithm = "HS256"

            token = _get_token("bob")
            response = await client.post(
                "/runs",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "topic": "AI Agents in Production",
                    "content_type": "blog_post",
                    "target_platform": "wordpress",
                },
            )

        assert response.status_code == 202
        body = response.json()
        assert "thread_id" in body
        assert len(body["thread_id"]) == 36  # UUID4 length
        assert body["status"] == "started"

    async def test_start_run_topic_too_short_returns_422(self, client):
        """topic with fewer than 3 characters must fail validation."""
        with (
            patch("api.main.build_graph"),
            patch("api.main.asyncio.create_task"),
            patch("api.auth.settings") as mock_auth_settings,
        ):
            mock_auth_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_auth_settings.jwt_algorithm = "HS256"

            token = _get_token("bob")
            response = await client.post(
                "/runs",
                headers={"Authorization": f"Bearer {token}"},
                json={"topic": "AI"},  # 2 chars — below min_length=3
            )

        assert response.status_code == 422

    async def test_start_run_missing_topic_returns_422(self, client):
        with (
            patch("api.main.build_graph"),
            patch("api.main.asyncio.create_task"),
            patch("api.auth.settings") as mock_auth_settings,
        ):
            mock_auth_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_auth_settings.jwt_algorithm = "HS256"

            token = _get_token("bob")
            response = await client.post(
                "/runs",
                headers={"Authorization": f"Bearer {token}"},
                json={},
            )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /runs/{thread_id}
# ---------------------------------------------------------------------------


class TestGetRunEndpoint:
    async def test_get_run_not_found_returns_404(self, client):
        """Unknown thread_id must yield 404."""
        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=None)

        with (
            patch("api.main.build_graph", return_value=mock_graph),
            patch("api.auth.settings") as mock_auth_settings,
        ):
            mock_auth_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_auth_settings.jwt_algorithm = "HS256"

            token = _get_token("alice")
            response = await client.get(
                "/runs/nonexistent-thread-id-00000",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 404

    async def test_get_run_existing_returns_state(self, client):
        """Existing thread_id returns thread state with expected keys."""
        fake_snapshot = MagicMock()
        fake_snapshot.values = {"topic": "AI Agents", "completed": False}
        fake_snapshot.next = ["writer"]
        fake_snapshot.metadata = {"step": 3}

        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=fake_snapshot)

        with (
            patch("api.main.build_graph", return_value=mock_graph),
            patch("api.auth.settings") as mock_auth_settings,
        ):
            mock_auth_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_auth_settings.jwt_algorithm = "HS256"

            token = _get_token("alice")
            response = await client.get(
                "/runs/known-thread-abc123",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["thread_id"] == "known-thread-abc123"
        assert "values" in body
        assert "next" in body

    async def test_get_run_unauthorized_returns_401(self, client):
        response = await client.get("/runs/some-thread-id")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# POST /runs/{thread_id}/resume
# ---------------------------------------------------------------------------


class TestResumeRunEndpoint:
    async def test_resume_run_approved_returns_200(self, client):
        """POST /runs/{id}/resume with approved=true must return 200."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={})

        with (
            patch("api.main.build_graph", return_value=mock_graph),
            patch("api.main.asyncio.create_task"),
            patch("api.auth.settings") as mock_auth_settings,
        ):
            mock_auth_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_auth_settings.jwt_algorithm = "HS256"

            token = _get_token("editor")
            response = await client.post(
                "/runs/test-thread-resume-001/resume",
                headers={"Authorization": f"Bearer {token}"},
                json={"approved": True, "feedback": "Looks great, publish it."},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["thread_id"] == "test-thread-resume-001"
        assert "approved" in body["status"] or "publishing" in body["status"]

    async def test_resume_run_rejected_returns_200(self, client):
        """POST /runs/{id}/resume with approved=false must return 200."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={})

        with (
            patch("api.main.build_graph", return_value=mock_graph),
            patch("api.main.asyncio.create_task"),
            patch("api.auth.settings") as mock_auth_settings,
        ):
            mock_auth_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_auth_settings.jwt_algorithm = "HS256"

            token = _get_token("editor")
            response = await client.post(
                "/runs/test-thread-resume-002/resume",
                headers={"Authorization": f"Bearer {token}"},
                json={"approved": False, "feedback": "Not ready — needs major revision."},
            )

        assert response.status_code == 200
        body = response.json()
        assert "rejected" in body["status"] or "terminating" in body["status"]

    async def test_resume_run_unauthorized_returns_401(self, client):
        response = await client.post(
            "/runs/some-thread-id/resume",
            json={"approved": True},
        )
        assert response.status_code == 401

    async def test_resume_run_missing_approved_field_returns_422(self, client):
        """approved is required — omitting it must fail validation."""
        with (
            patch("api.main.build_graph"),
            patch("api.main.asyncio.create_task"),
            patch("api.auth.settings") as mock_auth_settings,
        ):
            mock_auth_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_auth_settings.jwt_algorithm = "HS256"

            token = _get_token("editor")
            response = await client.post(
                "/runs/some-thread-id/resume",
                headers={"Authorization": f"Bearer {token}"},
                json={"feedback": "no approval field"},
            )

        assert response.status_code == 422

    async def test_resume_creates_background_task(self, client):
        """resume_run must schedule a background asyncio task (not block)."""
        mock_graph = MagicMock()

        with (
            patch("api.main.build_graph", return_value=mock_graph),
            patch("api.main.asyncio.create_task") as mock_create_task,
            patch("api.auth.settings") as mock_auth_settings,
        ):
            mock_auth_settings.jwt_secret_key = "test-secret-key-for-testing-only"
            mock_auth_settings.jwt_algorithm = "HS256"

            token = _get_token("editor")
            await client.post(
                "/runs/thread-bg-task/resume",
                headers={"Authorization": f"Bearer {token}"},
                json={"approved": True},
            )

        mock_create_task.assert_called_once()
