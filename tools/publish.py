"""
Publishing tools — idempotent wrappers for downstream CMS & social targets.
Each tool accepts an idempotency_key to prevent duplicate posts on retry.
"""
from __future__ import annotations

import base64
import hashlib
from typing import Any, Literal

import httpx
from langchain_core.tools import tool
from loguru import logger

from config import settings


def _make_idempotency_key(*parts: str) -> str:
    raw = ":".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ─────────────────────────────────────────────────────────────────────────────
# WordPress REST API
# ─────────────────────────────────────────────────────────────────────────────

@tool
async def publish_to_wordpress(
    title: str,
    body: str,
    slug: str,
    meta_description: str,
    thread_id: str,
    status: str = "draft",
) -> dict[str, str]:
    """
    Publish (or update) a post to WordPress via the REST API.
    Uses idempotency_key to prevent duplicate posts.
    Returns {post_id, url, idempotency_key}.
    """
    if not all([settings.wordpress_url, settings.wordpress_username, settings.wordpress_app_password]):
        raise RuntimeError("WordPress credentials not configured.")

    idem_key = _make_idempotency_key(thread_id, slug)
    credentials = base64.b64encode(
        f"{settings.wordpress_username}:{settings.wordpress_app_password}".encode()
    ).decode()

    payload: dict[str, Any] = {
        "title": title,
        "content": body,
        "slug": slug,
        "status": status,
        "excerpt": meta_description,
        "meta": {"idempotency_key": idem_key},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check if a post with this slug already exists (idempotency check)
        existing = await client.get(
            f"{settings.wordpress_url}/wp-json/wp/v2/posts",
            headers={"Authorization": f"Basic {credentials}"},
            params={"slug": slug, "status": "any"},
        )
        existing.raise_for_status()
        existing_posts = existing.json()

        if existing_posts:
            post_id = str(existing_posts[0]["id"])
            post_url = existing_posts[0]["link"]
            logger.info(f"WordPress: post already exists (id={post_id}), skipping duplicate.")
        else:
            resp = await client.post(
                f"{settings.wordpress_url}/wp-json/wp/v2/posts",
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            post_id = str(data["id"])
            post_url = data["link"]
            logger.info(f"WordPress: published post id={post_id} url={post_url}")

    return {
        "platform": "wordpress",
        "post_id": post_id,
        "url": post_url,
        "idempotency_key": idem_key,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stub: social media (extend with real API keys as needed)
# ─────────────────────────────────────────────────────────────────────────────

@tool
async def publish_to_social(
    platform: Literal["linkedin", "twitter", "bluesky"],
    content: str,
    thread_id: str,
) -> dict[str, str]:
    """
    Stub publisher for social media platforms.
    Replace the body with real API calls (X API v2, LinkedIn API, Bluesky AT Protocol).
    Returns a mock result for now — wire up API keys in config.py when ready.
    """
    idem_key = _make_idempotency_key(thread_id, platform, content[:32])
    logger.info(f"[STUB] publish_to_social platform={platform} idem_key={idem_key}")
    return {
        "platform": platform,
        "post_id": f"stub-{idem_key}",
        "url": f"https://{platform}.example.com/posts/{idem_key}",
        "idempotency_key": idem_key,
    }
