"""
Base classes, event bus, and coordinator for concurrent agent swarm.
Supports both concurrent asyncio tasks and deterministic discrete step execution.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from collections import deque

from sentinelsentry.schema import RawFeedItem, EnrichedIncident, TriageOutput

logger = logging.getLogger("sentinelsentry.agents")


class SwarmEventBus:
    """
    Central event queues connecting ingest, triage, verification, geocoding, and dashboard agents.
    Provides deque-based synchronous access and asyncio.Queue for async workers.
    """

    def __init__(self, maxsize: int = 1000):
        # Queues for discrete/synchronous processing
        self.ingest_items: deque[RawFeedItem] = deque(maxlen=maxsize)
        self.verification_items: deque[EnrichedIncident] = deque(maxlen=maxsize)
        self.geocoding_items: deque[EnrichedIncident] = deque(maxlen=maxsize)
        self.dashboard_items: deque[EnrichedIncident] = deque(maxlen=maxsize)

        # Sliding window buffer of recent raw items across feeds (for cross-feed corroboration)
        self.recent_items_buffer: List[RawFeedItem] = []
        self.max_buffer_size = 500

    def record_recent_item_sync(self, item: RawFeedItem) -> None:
        self.recent_items_buffer.append(item)
        if len(self.recent_items_buffer) > self.max_buffer_size:
            self.recent_items_buffer.pop(0)

    def get_recent_items_snapshot_sync(self) -> List[RawFeedItem]:
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
    def step_sync(self) -> int:
        """Process one batch or available items synchronously. Returns number of items processed."""
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
        for trg in self.triage_workers:
            counts["triaged"] += trg.step_sync()

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
