"""
Base classes, event bus, and coordinator for concurrent agent swarm.
Supports both concurrent asyncio tasks and deterministic discrete step execution.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque

from sentinelsentry.schema import RawFeedItem, EnrichedIncident, TriageOutput

logger = logging.getLogger("sentinelsentry.agents")


class SwarmEventBus:
    """
    Central event queues connecting ingest, triage, verification, geocoding, and dashboard agents.
    Provides deque-based synchronous access and asyncio.Queue for async workers.
    """

    def __init__(self, maxsize: int = 1000, corroboration_window_seconds: float = 600.0):
        # Queues for discrete/synchronous processing
        self.ingest_items: deque[RawFeedItem] = deque(maxlen=maxsize)
        self.verification_items: deque[EnrichedIncident] = deque(maxlen=maxsize)
        self.geocoding_items: deque[EnrichedIncident] = deque(maxlen=maxsize)
        self.dashboard_items: deque[EnrichedIncident] = deque(maxlen=maxsize)

        # Sliding window buffer of recent raw items across feeds (for cross-feed corroboration)
        self.recent_items_buffer: List[RawFeedItem] = []
        self.max_buffer_size = 500
        self.corroboration_window_seconds = corroboration_window_seconds
        self._buffer_lock = threading.Lock()

    def _prune_stale_items(self) -> None:
        """Remove items older than the corroboration time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.corroboration_window_seconds)
        self.recent_items_buffer = [
            item for item in self.recent_items_buffer
            if item.published_at >= cutoff
        ]

    def record_recent_item_sync(self, item: RawFeedItem) -> None:
        with self._buffer_lock:
            self.recent_items_buffer.append(item)
            self._prune_stale_items()
            if len(self.recent_items_buffer) > self.max_buffer_size:
                self.recent_items_buffer.pop(0)

    def get_recent_items_snapshot_sync(self) -> List[RawFeedItem]:
        with self._buffer_lock:
            return list(self.recent_items_buffer)


class BaseAgent(ABC):
    """
    Abstract worker agent supporting both discrete sync stepping and async task execution.
    """

    def __init__(self, agent_name: str, bus: SwarmEventBus):
        self.agent_name = agent_name
        self.bus = bus
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @abstractmethod
    def step_sync(self, max_items: Optional[int] = None) -> int:
        """Process one batch or available items synchronously. Returns number of items processed.
        
        Args:
            max_items: Optional cap on items to process in this step. None means process all available.
        """
        pass


class SwarmCoordinator:
    """
    Coordinates multi-agent execution across Ingest, Triage, Verification, Geocoder, and Dashboard.
    Can run in discrete step mode (for reproducible benchmarks) or continuous loop.
    """

    def __init__(self, bus: Optional[SwarmEventBus] = None):
        self.bus = bus or SwarmEventBus()
        self.ingest_agents: List[Any] = []
        self.triage_workers: List[Any] = []
        self.verification_agent: Optional[Any] = None
        self.geocoder_agent: Optional[Any] = None
        self.dashboard_agent: Optional[Any] = None
        # Async Mode B state
        self._async_running: bool = False
        self._async_tasks: List[asyncio.Task] = []

    def register_ingest_agent(self, agent: Any) -> None:
        self.ingest_agents.append(agent)

    def register_triage_worker(self, worker: Any) -> None:
        self.triage_workers.append(worker)

    def set_verification_agent(self, agent: Any) -> None:
        self.verification_agent = agent

    def set_geocoder_agent(self, agent: Any) -> None:
        self.geocoder_agent = agent

    def set_dashboard_agent(self, agent: Any) -> None:
        self.dashboard_agent = agent

    def step(self) -> Dict[str, int]:
        """
        Execute one discrete execution round through all swarm stages.
        Returns counts of items processed per stage.
        """
        counts = {
            "ingested": 0,
            "triaged": 0,
            "verified": 0,
            "geocoded": 0,
            "delivered": 0,
        }

        # 1. Ingest Stage
        for ing in self.ingest_agents:
            counts["ingested"] += ing.step_sync()

        # 2. Triage Stage
        worker_batch = 5 if len(self.triage_workers) > 1 else None
        for trg in self.triage_workers:
            counts["triaged"] += trg.step_sync(max_items=worker_batch) if worker_batch else trg.step_sync()

        # 3. Verification Stage (risk >= 4)
        if self.verification_agent:
            counts["verified"] += self.verification_agent.step_sync()

        # 4. Geocoder Stage
        if self.geocoder_agent:
            counts["geocoded"] += self.geocoder_agent.step_sync()

        # 5. Dashboard Stage
        if self.dashboard_agent:
            counts["delivered"] += self.dashboard_agent.step_sync()

        return counts

    def drain(self, max_rounds: int = 10) -> List[Dict[str, int]]:
        """Run steps until all internal queues are cleared."""
        round_results = []
        for _ in range(max_rounds):
            res = self.step()
            round_results.append(res)
            # If nothing was processed across all queues, drain is complete
            if sum(res.values()) == 0:
                break
        return round_results

    # ------------------------------------------------------------------
    # Mode B: Asynchronous concurrent task execution
    # ------------------------------------------------------------------

    async def start(self, poll_interval: float = 0.1) -> None:
        """Launch all registered agents as concurrent asyncio tasks.
        
        Each agent runs a loop calling step_sync until stopped.
        """
        if self._async_running:
            logger.warning("SwarmCoordinator.start() called but already running.")
            return

        self._async_running = True
        all_agents: List[Any] = list(self.ingest_agents) + list(self.triage_workers)
        for agent in [self.verification_agent, self.geocoder_agent, self.dashboard_agent]:
            if agent is not None:
                all_agents.append(agent)

        async def _agent_loop(agent: Any) -> None:
            agent._running = True
            while self._async_running:
                try:
                    agent.step_sync()
                except Exception as exc:
                    logger.error(f"[{agent.agent_name}] Error in async loop: {exc}")
                await asyncio.sleep(poll_interval)
            agent._running = False

        self._async_tasks = [asyncio.create_task(_agent_loop(a)) for a in all_agents]
        logger.info(f"SwarmCoordinator started {len(self._async_tasks)} async agent tasks.")

    async def stop(self) -> None:
        """Signal all async agent tasks to stop and await their completion."""
        if not self._async_running:
            return
        self._async_running = False
        for task in self._async_tasks:
            task.cancel()
        await asyncio.gather(*self._async_tasks, return_exceptions=True)
        self._async_tasks.clear()
        logger.info("SwarmCoordinator stopped all async agent tasks.")
