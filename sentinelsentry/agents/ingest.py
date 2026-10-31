"""
Ingest agents running per feed to poll streams and push to shared queue.
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, List
from sentinelsentry.schema import RawFeedItem
from sentinelsentry.config import FeedConfig
from sentinelsentry.feeds.client import RSSFeedClient
from sentinelsentry.agents.base import BaseAgent, SwarmEventBus

logger = logging.getLogger("sentinelsentry.agents.ingest")


class IngestAgent(BaseAgent):
    """
    Ingest worker polling a single RSS / dispatch feed stream.
    Pushes discovered items into the SwarmEventBus.
    """

    def __init__(
        self,
        feed_config: FeedConfig,
        bus: SwarmEventBus,
        custom_fetcher: Optional[Callable[[], List[RawFeedItem]]] = None,
    ):
        super().__init__(agent_name=f"IngestAgent[{feed_config.id}]", bus=bus)
        self.config = feed_config
        self.client = RSSFeedClient(feed_config)
        self.custom_fetcher = custom_fetcher
        self.total_ingested = 0

    def step_sync(self, max_items: Optional[int] = None) -> int:
        """Synchronously poll the feed (using custom_fetcher or XML parser) and push new items."""
        if not self.custom_fetcher:
            return 0

        raw_items = self.custom_fetcher()
        count = 0
        limit = max_items if max_items is not None else float("inf")
        for item in raw_items:
            if count >= limit:
                break
            if not self.client.dedup_cache.is_seen(item.id):
                self.client.dedup_cache.mark_seen(item.id)
                self.total_ingested += 1
                count += 1
                # Buffer in recent items for cross-feed verification
                self.bus.record_recent_item_sync(item)
                # Enqueue for triage
                self.bus.ingest_items.append(item)
                logger.info(f"[{self.agent_name}] Ingested item {item.id}: {item.title}")
        return count
