"""
Local CLI runner — Phase 1 smoke test.
Usage:
    python run_local.py --topic "The Future of AI Agents" --type blog_post
"""
from __future__ import annotations

import argparse
import asyncio

from loguru import logger
from langgraph.checkpoint.memory import MemorySaver

from graph import build_graph
from observability import setup_observability
from state import ContentState


async def main(topic: str, content_type: str, platform: str) -> None:
    setup_observability()

    # Use MemorySaver ONLY for local dev — never in production
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer)

    thread_id = f"local-{topic[:20].replace(' ', '-').lower()}"
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 30,
    }

    initial = ContentState(
        topic=topic,
        content_type=content_type,  # type: ignore[arg-type]
        target_platform=platform,
        run_metadata={"thread_id": thread_id, "started_by": "cli"},
    )

    logger.info(f"Starting local run: {topic!r}")

    async for event in graph.astream_events(
        initial.model_dump(), config=config, version="v2"
    ):
        kind = event["event"]
        name = event.get("name", "")
        if kind == "on_chain_start" and name not in ("LangGraph",):
            logger.info(f"  ▶ {name}")
        elif kind == "on_chain_end" and name not in ("LangGraph",):
            logger.info(f"  ✓ {name}")
        elif kind == "on_chat_model_stream":
            # Token streaming — print to stdout for live writing effect
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)

    # Get final state
    snapshot = await graph.aget_state(config)
    state_vals = snapshot.values

    print("\n\n" + "=" * 60)

    if state_vals.get("seo_result"):
        seo = state_vals["seo_result"]
        print(f"Title:      {seo['optimized_title']}")
        print(f"Slug:       {seo['slug']}")
        print(f"SEO Score:  {seo['seo_score']}")
        print(f"Meta:       {seo['optimized_meta']}")
        print("\nArticle body (preview):")
        print(seo["optimized_body"][:1000])
    elif state_vals.get("error"):
        print(f"ERROR: {state_vals['error']}")
    else:
        print("Run interrupted for HITL — use /runs/{thread_id}/resume to approve.")

    if state_vals.get("token_usage"):
        tu = state_vals["token_usage"]
        print(f"\nTokens used: {tu['total_tokens']:,}  "
              f"Est. cost: ${tu['estimated_cost_usd']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Content Engine CLI")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--type", default="blog_post",
                        choices=["blog_post", "newsletter", "social_thread", "product_description"])
    parser.add_argument("--platform", default="wordpress")
    args = parser.parse_args()

    asyncio.run(main(args.topic, args.type, args.platform))
