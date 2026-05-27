"""
Observability bootstrap.
Call `setup_observability()` once at process startup before importing graph.

  • LangSmith tracing  — one env-var toggle
  • Loguru             — structured JSON logs with correlation IDs
  • Prometheus counters — per-node latency and token cost histograms
"""
from __future__ import annotations

import os
import sys
import time
from functools import wraps
from typing import Any, Callable

from loguru import logger
from prometheus_client import Counter, Histogram

from config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────

NODE_DURATION = Histogram(
    "content_engine_node_duration_seconds",
    "Latency per graph node",
    ["node_name"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)

TOKEN_COST = Histogram(
    "content_engine_token_cost_usd",
    "Estimated LLM cost per run in USD",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

RUNS_TOTAL = Counter(
    "content_engine_runs_total",
    "Total content runs",
    ["status"],  # started | completed | failed | rejected
)

REVISION_ITERATIONS = Histogram(
    "content_engine_revision_iterations",
    "Number of writer→reviewer iterations per run",
    buckets=[1, 2, 3, 4, 5],
)


def track_node(node_name: str) -> Callable:
    """Decorator: record node execution time in Prometheus."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                NODE_DURATION.labels(node_name=node_name).observe(
                    time.perf_counter() - start
                )
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Loguru structured logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_observability() -> None:
    """
    Idempotent setup — safe to call multiple times.
    """
    # Remove default Loguru handler; replace with JSON stdout
    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | "
            "{name}:{function}:{line} | {message}"
        ),
        level="DEBUG" if os.getenv("DEBUG") else "INFO",
        serialize=False,
        enqueue=True,
    )

    # LangSmith
    if settings.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        logger.info(
            f"LangSmith tracing ON — project={settings.langsmith_project}"
        )
    else:
        logger.info("LangSmith tracing OFF")

    logger.info("Observability initialised")
