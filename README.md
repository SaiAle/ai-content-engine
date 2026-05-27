# AI Content Automation Engine

> **Production-grade AI pipeline** — LangGraph 1.0 · FastAPI · PostgreSQL · Human-in-the-Loop

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-green)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

A fully automated, multi-agent content pipeline that researches, writes, edits, and publishes long-form content with **human approval gates** before anything goes live. Built with LangGraph 1.0 stateful graphs, durable PostgreSQL checkpointing, and a React HITL UI for editorial control.

```
Topic In → Research → Draft → Review → Optimize → HITL Approval → Publish
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI  (:8000)                         │
│   POST /runs   GET /runs/{id}/stream   POST /runs/{id}/resume   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  LangGraph  │  StateGraph (interrupt)
                    │   Pipeline  │
                    └──────┬──────┘
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌──────▼──────┐  ┌─────▼──────┐
    │ Ingestor  │   │ Researcher  │  │  Writer    │
    │ (parse)   │   │ (Tavily +   │  │ (GPT-4o /  │
    └─────┬─────┘   │  Firecrawl) │  │  Claude)   │
          │         └──────┬──────┘  └─────┬──────┘
    ┌─────▼─────┐          │         ┌─────▼──────┐
    │  Planner  │          │         │  Reviewer  │
    │ (outline) │          │         │ (critique) │
    └───────────┘          │         └─────┬──────┘
                           │               │ (loop ≤3x)
                    ┌──────▼───────────────▼──────┐
                    │         Optimizer            │
                    │      (SEO + metadata)        │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │      HITL interrupt()         │  ← React UI
                    │  Human approves / rejects     │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │         Publisher             │
                    │  WordPress · LinkedIn · X     │
                    └───────────────────────────────┘

  Storage: PostgreSQL (AsyncPostgresSaver) + Redis cache
  Observability: LangSmith · Prometheus · Grafana
```

---

## Features

| Feature | Detail |
|---|---|
| **Multi-agent graph** | 7-node LangGraph StateGraph with conditional edges and reviewer loop |
| **HITL approval** | `interrupt()` halts before publish; React UI for approve/reject/feedback |
| **Durable state** | PostgreSQL AsyncPostgresSaver — survives crashes, supports multi-replica |
| **SSE streaming** | `/runs/{id}/stream` streams token-by-token via Server-Sent Events |
| **Provider-agnostic LLM** | `init_chat_model("openai:gpt-4o")` — swap to Claude/Gemini in `.env` |
| **SEO optimizer** | Structured output: slug, meta description, keyword density score |
| **Idempotent publish** | SHA-256 idempotency key prevents duplicate posts on retry |
| **JWT auth** | `POST /token` → Bearer token for all API endpoints |
| **Rate limiting** | `slowapi` per-IP limits on `/runs` |
| **Observability** | Prometheus histograms per node, LangSmith traces, Loguru JSON logs |
| **Kubernetes-ready** | HPA (CPU 70% / mem 80%), pod anti-affinity, SSE-aware Ingress |
| **74 tests** | pytest async suite covering state, nodes, router, and API |

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key (or any LangChain-supported provider)
- Tavily API key (web search)

### 1. Clone & configure

```bash
git clone https://github.com/SaiAle/ai-content-engine.git
cd ai-content-engine
cp .env.example .env
# Edit .env — add OPENAI_API_KEY, TAVILY_API_KEY, etc.
```

### 2. Start services

```bash
docker compose up -d
# PostgreSQL :5432, Redis :6379, API :8000
```

### 3. Get a token & run

```bash
# Authenticate
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -d "username=admin&password=secret" | jq -r .access_token)

# Start a content run
curl -X POST http://localhost:8000/runs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"topic":"The Future of AI Agents","content_type":"blog_post","target_platform":"wordpress"}'

# Stream output live
curl -N http://localhost:8000/runs/{thread_id}/stream \
  -H "Authorization: Bearer $TOKEN"

# Approve in HITL gate
curl -X POST http://localhost:8000/runs/{thread_id}/resume \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "feedback": "Looks great!"}'
```

### 4. Local CLI (no Docker)

```bash
pip install -r requirements.txt
python run_local.py --topic "Future of AI Agents" --type blog_post
```

---

## Environment Variables

```env
# LLM
PRIMARY_MODEL=openai:gpt-4o
REVIEW_MODEL=anthropic:claude-3-5-sonnet-20241022

# APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
FIRECRAWL_API_KEY=fc-...

# Infrastructure
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/content_engine
REDIS_URL=redis://localhost:6379/0

# Publishing
WORDPRESS_URL=https://yourblog.com
WORDPRESS_USERNAME=admin
WORDPRESS_APP_PASSWORD=xxxx xxxx xxxx

# Observability
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=ai-content-engine
```

---

## Project Structure

```
ai_content_engine/
├── api/
│   ├── main.py          # FastAPI app, SSE, JWT, rate limiting
│   └── auth.py          # Bearer token logic
├── nodes/
│   ├── base.py          # init_chat_model, retry, fallback, timeout
│   ├── ingestor.py      # Topic parsing & context extraction
│   ├── planner.py       # Outline generation
│   ├── researcher.py    # Tavily search + Firecrawl scrape
│   ├── writer.py        # Draft generation (revision-aware)
│   ├── reviewer.py      # Critique & score
│   ├── optimizer.py     # SEO title/slug/meta
│   └── publisher.py     # WordPress + social dispatch
├── tools/
│   ├── search.py        # @tool web_search, scrape_url
│   └── publish.py       # @tool publish_to_wordpress, publish_to_social
├── tests/               # 74 pytest tests
├── frontend/
│   └── ContentEngineUI.jsx  # React HITL approval UI
├── monitoring/
│   ├── grafana_dashboard.json
│   ├── prometheus_alerts.yml
│   └── docker-compose.monitoring.yml
├── k8s/                 # Kubernetes manifests
├── graph.py             # StateGraph definition
├── state.py             # ContentState Pydantic model
├── config.py            # Pydantic v2 Settings
├── observability.py     # Loguru + Prometheus + LangSmith
├── run_local.py         # CLI runner
├── Dockerfile
└── docker-compose.yml
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/token` | Get JWT access token |
| `POST` | `/runs` | Start a new content run (202 Accepted) |
| `GET` | `/runs/{id}` | Get run state snapshot |
| `GET` | `/runs/{id}/stream` | SSE stream of graph events + tokens |
| `POST` | `/runs/{id}/resume` | Resume after HITL gate (approve/reject) |
| `GET` | `/metrics` | Prometheus metrics endpoint |

---

## Monitoring

```bash
# Start Prometheus + Grafana
docker compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d

# Grafana at http://localhost:3000 (admin/admin)
```

**Dashboard panels:** Node latency (p50/p95/p99) · Token cost · Run success rate · Revision iterations · API latency · Active runs

**Alert rules:** High node latency · Run failure rate >10% · Token budget burn · API latency · No runs started (dead-man)

---

## Kubernetes Deployment

```bash
kubectl apply -k k8s/
# Deploys: Namespace, Secrets, ConfigMap, PostgreSQL StatefulSet,
#          Redis, API Deployment (podAntiAffinity), HPA, Ingress
```

HPA scales API pods from 2 → 10 replicas at CPU 70% / Memory 80%.

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
# 74 tests: state, nodes, router, API
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 1.0 (StateGraph + interrupt) |
| LLM | LangChain `init_chat_model` (provider-agnostic) |
| API | FastAPI + uvicorn (uvloop + httptools) |
| State persistence | PostgreSQL via `langgraph-checkpoint-postgres` |
| Cache | Redis |
| Auth | python-jose JWT + passlib |
| Rate limiting | slowapi |
| Observability | Loguru · Prometheus · Grafana · LangSmith |
| Containerisation | Docker Compose · Kubernetes (HPA) |
| Testing | pytest-asyncio · httpx (ASGITransport) |
| Frontend | React (JSX) + Lucide icons + SSE EventSource |

---

## License

MIT © 2026 SaiAle
