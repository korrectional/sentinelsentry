"""
Asynchronous RSS feed client and parser with deduplication.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional, Set, Any
from collections import OrderedDict
import feedparser
import httpx
from dateutil import parser as date_parser

from sentinelsentry.schema import RawFeedItem
from sentinelsentry.config import FeedConfig

logger = logging.getLogger("sentinelsentry.feeds.client")


class DeduplicationCache:
    """Bounded LRU-style cache for seen feed item IDs."""

    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._seen: OrderedDict[str, float] = OrderedDict()

    def is_seen(self, item_id: str) -> bool:
        return item_id in self._seen

    def mark_seen(self, item_id: str) -> None:
        if item_id in self._seen:
            self._seen.move_to_end(item_id)
            return
        self._seen[item_id] = datetime.now(timezone.utc).timestamp()
        if len(self._seen) > self.max_size:
            self._seen.popitem(last=False)

    def clear(self) -> None:
        self._seen.clear()


class RSSFeedClient:
    """
    Asynchronous poller for RSS / dispatch feeds with feedparser parsing and deduplication.
    """

    def __init__(
        self,
        feed_config: FeedConfig,
        timeout_seconds: float = 8.0,
        cache_size: int = 2000,
    ):
        self.config = feed_config
        self.timeout = timeout_seconds
        self.dedup_cache = DeduplicationCache(max_size=cache_size)
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def _parse_pubdate(self, entry: Any) -> datetime:
        """Robustly parse pubDate from feed entry or default to current UTC."""
        date_str = entry.get("published") or entry.get("updated") or entry.get("pubDate")
        if date_str:
            try:
                dt = date_parser.parse(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
        return datetime.now(timezone.utc)

    def _generate_item_id(self, entry: Any, feed_id: str) -> str:
        """Derive or extract a reliable unique ID for the entry."""
        if entry.get("id"):
            return str(entry["id"]).strip()
        if entry.get("guid"):
            return str(entry["guid"]).strip()

        # Fallback hash of title + description
        title = entry.get("title", "")
        summary = entry.get("summary", entry.get("description", ""))
        sig = f"{feed_id}:{title}:{summary}"
        return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]

    def parse_feed_content(self, xml_content: str) -> List[RawFeedItem]:
        """
        Parse raw XML string with feedparser and return newly discovered items.
        """
        parsed = feedparser.parse(xml_content)
        new_items: List[RawFeedItem] = []

        for entry in parsed.entries:
            item_id = self._generate_item_id(entry, self.config.id)
            if self.dedup_cache.is_seen(item_id):
                continue

            raw_text = entry.get("summary") or entry.get("description") or entry.get("title", "")
            title = entry.get("title", "")
            link = entry.get("link")
            pub_date = self._parse_pubdate(entry)

            # Metadata extraction (e.g. category)
            meta = {}
            if "tags" in entry and entry.tags:
                meta["categories"] = [t.term for t in entry.tags if hasattr(t, "term")]

            raw_item = RawFeedItem(
                id=item_id,
                feed_id=self.config.id,
                feed_name=self.config.name,
                title=title,
                raw_text=raw_text,
                published_at=pub_date,
                link=link,
                guid=item_id,
                metadata=meta,
            )

            self.dedup_cache.mark_seen(item_id)
            new_items.append(raw_item)

        return new_items

    async def fetch_once(self) -> List[RawFeedItem]:
        """
        Perform a single asynchronous fetch of the configured feed URL.
        """
        client = await self._get_http_client()
        try:
            response = await client.get(self.config.url)
            response.raise_for_status()
            return self.parse_feed_content(response.text)
        except Exception as e:
            logger.warning(f"Error fetching feed '{self.config.id}' from {self.config.url}: {e}")
            return []

    async def stream_items(self, max_iterations: Optional[int] = None) -> AsyncGenerator[RawFeedItem, None]:
        """
        Continuously poll the feed and yield new RawFeedItems.
        """
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            items = await self.fetch_once()
            for item in items:
                yield item
            iterations += 1
            await asyncio.sleep(self.config.poll_interval_seconds)
