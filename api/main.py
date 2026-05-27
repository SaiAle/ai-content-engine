"""
FastAPI service for the AI Content Automation Engine.

Endpoints:
  POST /token                       — issue JWT (dev convenience)
  POST /runs                        — start a new content run
  GET  /runs/{thread_id}            — get run state
  GET  /runs/{thread_id}/stream     — SSE stream of graph events
  POST /runs/{thread_id}/resume     — submit HITL approval/rejection
  GET  /health                      — liveness probe
  GET  /metrics                     — Prometheus metrics

All endpoints except /health and /token require Bearer JWT.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from api.auth import create_access_token, get_current_user
from config import settings
from graph import build_graph
from state import ContentState

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(
    title="AI Content Automation Engine",
    version="1.0.0",
    description="LangGraph-powered, production-grade content generation service.",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ─────────────────────────────────────────────────────────────────────────────
# Checkpointer lifecycle
# ─────────────────────────────────────────────────────────────────────────────

_checkpointer: AsyncPostgresSaver | None = None


async def get_checkpointer() -> AsyncPostgresSaver:
    global _checkpointer
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialised — app startup incomplete.")
    return _checkpointer


@app.on_event("startup")
async def startup() -> None:
    global _checkpointer
    conn_str = settings.database_url.replace("+asyncpg", "")
    _checkpointer = await AsyncPostgresSaver.from_conn_string(conn_str)
    await _checkpointer.setup()
    logger.info("AsyncPostgresSaver initialised")


@app.on_event("shutdown")
async def shutdown() -> None:
    global _checkpointer
    if _checkpointer:
        await _checkpointer.aclose()
        logger.info("AsyncPostgresSaver closed")


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    content_type: str = Field(default="blog_post")
    target_platform: str = Field(default="wordpress")


class ResumeRequest(BaseModel):
    approved: bool
    feedback: str = ""


class TokenRequest(BaseModel):
    username: str


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/token", tags=["auth"])
async def issue_token(req: TokenRequest) -> dict[str, str]:
    """Dev-only: issue a JWT for a username. Add proper auth in production."""
    token = create_access_token(req.username)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/runs", tags=["runs"], status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("10/minute")
async def start_run(
    request: Request,
    body: RunRequest,
    user: str = Depends(get_current_user),
    checkpointer: AsyncPostgresSaver = Depends(get_checkpointer),
) -> dict[str, str]:
    """
    Start a new content generation run.
    Returns thread_id — use it to stream events or poll state.
    """
    thread_id = str(uuid.uuid4())
    graph = build_graph(checkpointer)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_recursion_limit,
    }

    initial_state = ContentState(
        topic=body.topic,
        content_type=body.content_type,  # type: ignore[arg-type]
        target_platform=body.target_platform,
        run_metadata={"thread_id": thread_id, "started_by": user},
    )

    # Kick off in background — the graph will checkpoint at each super-step
    asyncio.create_task(
        _run_graph(graph, initial_state, config)
    )

    logger.info(f"[api] started run thread_id={thread_id} user={user}")
    return {"thread_id": thread_id, "status": "started"}


async def _run_graph(graph, initial_state: ContentState, config: dict) -> None:
    try:
        async with asyncio.timeout(settings.global_run_timeout_seconds):
            await graph.ainvoke(initial_state.model_dump(), config)
    except asyncio.TimeoutError:
        logger.error(f"[api] run timed out config={config}")
    except Exception as exc:
        logger.exception(f"[api] run failed: {exc}")


@app.get("/runs/{thread_id}", tags=["runs"])
async def get_run_state(
    thread_id: str,
    user: str = Depends(get_current_user),
    checkpointer: AsyncPostgresSaver = Depends(get_checkpointer),
) -> dict[str, Any]:
    """Return the latest checkpointed state for a run."""
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_graph(checkpointer)
    snapshot = await graph.aget_state(config)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No run found: {thread_id}")
    return {
        "thread_id": thread_id,
        "values": snapshot.values,
        "next": snapshot.next,
        "metadata": snapshot.metadata,
    }


@app.get("/runs/{thread_id}/stream", tags=["runs"])
async def stream_run(
    thread_id: str,
    request: Request,
    user: str = Depends(get_current_user),
    checkpointer: AsyncPostgresSaver = Depends(get_checkpointer),
) -> EventSourceResponse:
    """
    SSE stream of LangGraph events for a given run.
    Events: node_start, node_end, on_chain_stream (token-level), error.
    """
    graph = build_graph(checkpointer)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_recursion_limit,
    }

    async def event_generator() -> AsyncIterator[dict]:
        try:
            async for event in graph.astream_events(
                None,  # None = resume from last checkpoint
                config=config,
                version="v2",
            ):
                if await request.is_disconnected():
                    break
                yield {
                    "event": event["event"],
                    "data": json.dumps(
                        {
                            "name": event.get("name"),
                            "data": _serialise(event.get("data")),
                        }
                    ),
                }
        except Exception as exc:
            logger.exception(f"[sse] stream error: {exc}")
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())


@app.post("/runs/{thread_id}/resume", tags=["runs"])
async def resume_run(
    thread_id: str,
    body: ResumeRequest,
    user: str = Depends(get_current_user),
    checkpointer: AsyncPostgresSaver = Depends(get_checkpointer),
) -> dict[str, str]:
    """
    Resume a paused run after HITL review.
    Provide approved=true to publish, approved=false to reject.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_recursion_limit,
    }
    graph = build_graph(checkpointer)

    asyncio.create_task(
        _resume_graph(
            graph,
            config,
            Command(resume={"approved": str(body.approved).lower(), "feedback": body.feedback}),
        )
    )

    action = "approved — publishing" if body.approved else "rejected — terminating"
    logger.info(f"[api] resume thread_id={thread_id} {action} user={user}")
    return {"thread_id": thread_id, "status": action}


async def _resume_graph(graph, config: dict, command: Command) -> None:
    try:
        async with asyncio.timeout(settings.global_run_timeout_seconds):
            await graph.ainvoke(command, config)
    except asyncio.TimeoutError:
        logger.error(f"[api] resume timed out config={config}")
    except Exception as exc:
        logger.exception(f"[api] resume failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _serialise(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable objects to strings."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    return str(obj)
