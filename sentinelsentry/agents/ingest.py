"""
Ingest agents running per feed to poll streams and push to shared queue.
Simulates individual Maritime micro-VM isolation for each ingest worker.
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
        super().__init__(
            agent_name=f"IngestAgent[{feed_config.id}]",
            bus=bus,
            micro_vm_id=f"maritime-vm-ingest-{feed_config.id.replace('_', '-')}",
            agent_role=f"Feed Ingestion ({feed_config.name})",
        )
        self.config = feed_config
        self.client = RSSFeedClient(feed_config)
        self.custom_fetcher = custom_fetcher
        self.total_ingested = 0

    def step_sync(self, max_items: Optional[int] = None) -> int:
        """Synchronously poll the feed and push new items."""
        if self.custom_fetcher:
            raw_items = self.custom_fetcher()
        elif self.config.url and self.config.url.startswith("http"):
            try:
                import urllib.request
                req = urllib.request.Request(self.config.url, headers={"User-Agent": "SentinelSentry/1.0"})
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    xml_text = resp.read().decode("utf-8")
                raw_items = self.client.parse_feed_content(xml_text)
            except Exception as e:
                logger.debug(f"[{self.agent_name}] Feed fetch failed ({e})")
                raw_items = []
        else:
            self.record_heartbeat("idle")
            return 0

        count = 0
        limit = max_items if max_items is not None else float("inf")
        for item in raw_items:
            if count >= limit:
                break
            if not self.client.dedup_cache.is_seen(item.id):
                self.client.dedup_cache.mark_seen(item.id)
                self.total_ingested += 1
                self.processed_count = self.total_ingested
                count += 1
                # Buffer in recent items for cross-feed verification
                self.bus.record_recent_item_sync(item)
                # Enqueue for triage
                self.bus.ingest_items.append(item)
                logger.info(f"[{self.agent_name}] Ingested item {item.id}: {item.title}")

        self.record_heartbeat("working" if count > 0 else "idle")
        return count

    async def run_async_loop(self) -> None:
        """Asynchronously stream items from feed and enqueue to bus."""
        import asyncio
        self._running = True
        try:
            if self.custom_fetcher:
                while self._running:
                    self.step_sync()
                    await asyncio.sleep(0.1)
            else:
                async for item in self.client.stream_items():
                    if not self._running:
                        break
                    self.bus.record_recent_item_sync(item)
                    self.bus.ingest_items.append(item)
                    self.total_ingested += 1
                    self.processed_count = self.total_ingested
                    self.record_heartbeat("working")
        except asyncio.CancelledError:
            self._running = False
            self.record_heartbeat("idle")
