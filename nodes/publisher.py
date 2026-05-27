"""
Publisher node — dispatches the SEO-optimised article to target platforms.

HITL: this node is entered *after* `interrupt()` returns human_approved=True.
If approval was rejected, the run is terminated with an explanation.

Input:  state.seo_result, state.human_approved, state.target_platform,
        state.run_metadata["thread_id"]
Output: state.publish_results, state.completed
"""
from __future__ import annotations

from loguru import logger

from state import ContentState, PublishResult
from tools.publish import publish_to_social, publish_to_wordpress


async def publisher_node(state: ContentState) -> dict:
    if not state.human_approved:
        logger.info("[publisher] human rejected — terminating run")
        return {
            "completed": True,
            "run_metadata": {**state.run_metadata, "publish_status": "rejected_by_human"},
        }

    if state.seo_result is None:
        return {"error": "publisher_node called before SEO optimisation"}

    thread_id: str = state.run_metadata.get("thread_id", "unknown")
    results: list[PublishResult] = []

    logger.info(
        f"[publisher] publishing to platform={state.target_platform} "
        f"thread_id={thread_id}"
    )

    if state.target_platform == "wordpress":
        raw = await publish_to_wordpress.ainvoke(
            {
                "title": state.seo_result.optimized_title,
                "body": state.seo_result.optimized_body,
                "slug": state.seo_result.slug,
                "meta_description": state.seo_result.optimized_meta,
                "thread_id": thread_id,
                "status": "publish",
            }
        )
        results.append(PublishResult(**raw))

    elif state.target_platform in ("linkedin", "twitter", "bluesky"):
        # Generate a social-media variant (first 280 chars of optimised body)
        social_content = (
            f"{state.seo_result.optimized_title}\n\n"
            + state.seo_result.optimized_body[:800]
            + f"\n\nRead more: {results[0].url if results else ''}"
        )
        raw = await publish_to_social.ainvoke(
            {
                "platform": state.target_platform,
                "content": social_content,
                "thread_id": thread_id,
            }
        )
        results.append(PublishResult(**raw))

    else:
        logger.warning(f"[publisher] unknown platform {state.target_platform!r} — skipping publish")

    logger.info(f"[publisher] {len(results)} results")
    return {
        "publish_results": results,
        "completed": True,
    }
