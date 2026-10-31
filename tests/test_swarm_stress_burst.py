"""
Stress and burst test suite for the Multi-Agent Swarm:
- High-volume incident bursts (100+ items)
- Parallel triage worker dispatch & queue draining
- Deduplication cache under repetitive GUID spam
- Telemetry counter integrity across micro-VM agents
"""

import time
from sentinelsentry.config import settings, FeedConfig
from sentinelsentry.schema import RawFeedItem
from sentinelsentry.agents.base import SwarmEventBus, SwarmCoordinator
from sentinelsentry.agents.ingest import IngestAgent
from sentinelsentry.agents.triage import TriageAgent, LocalTriageEngine
from sentinelsentry.agents.verification import VerificationAgent
from sentinelsentry.agents.geocoder import GeocoderAgent
from sentinelsentry.agents.dashboard import AlertDashboardAgent


def test_swarm_burst_load_100_items():
    """Verify swarm processes a burst of 100 items without memory leaks, deadlocks, or dropped items."""
    bus = SwarmEventBus()
    coordinator = SwarmCoordinator(bus=bus)

    # 2 Triage workers
    t1 = TriageAgent(bus=bus, worker_id=1, engine=LocalTriageEngine())
    t2 = TriageAgent(bus=bus, worker_id=2, engine=LocalTriageEngine())
    verifier = VerificationAgent(bus=bus)
    geocoder = GeocoderAgent(bus=bus, enable_network=False)
    dashboard = AlertDashboardAgent(bus=bus)

    coordinator.register_triage_worker(t1)
    coordinator.register_triage_worker(t2)
    coordinator.set_verification_agent(verifier)
    coordinator.set_geocoder_agent(geocoder)
    coordinator.set_dashboard_agent(dashboard)

    # Inject 100 mixed items
    locations = ["1200 S Saunders St", "I-40 EB MM 287", "Crabtree Valley Mall", "Atlantic Ave & Hodges St"]
    for i in range(100):
        loc = locations[i % len(locations)]
        item = RawFeedItem(
            id=f"burst-item-{i:04d}",
            feed_id="cad_police" if i % 2 == 0 else "social_crisis",
            feed_name="CAD Dispatch",
            title=f"Burst Incident {i}",
            raw_text=f"10-70 COMM BLDG {loc}. Flames showing roof, caller reports emergency {i}",
        )
        bus.record_recent_item_sync(item)
        bus.ingest_items.append(item)

    assert len(bus.ingest_items) == 100

    # Drain queues completely
    coordinator.drain(max_rounds=25)

    # All queues must be empty
    assert len(bus.ingest_items) == 0
    assert len(bus.verification_items) == 0
    assert len(bus.geocoding_items) == 0
    assert len(bus.dashboard_items) == 0

    # Total dashboard incidents must match
    assert len(dashboard.incidents) == 100
    assert dashboard.stats["total_processed"] == 100


def test_dedup_cache_repetitive_spam():
    """Verify that repeated injection of the same GUID is strictly rejected by the dedup cache."""
    bus = SwarmEventBus()
    cfg = FeedConfig(
        id="test_feed",
        name="Test CAD",
        description="Test Feed",
    )
    items_store = []
    agent = IngestAgent(
        feed_config=cfg,
        bus=bus,
        custom_fetcher=lambda: items_store,
    )

    # Add 1 item
    test_item = RawFeedItem(
        id="DUPLICATE-GUID-001",
        feed_id="test_feed",
        feed_name="Test CAD",
        title="Duplicate Alarm",
        raw_text="Waterflow alarm test trip",
    )
    items_store.append(test_item)

    # First poll: ingests 1 item
    count1 = agent.step_sync()
    assert count1 == 1
    assert len(bus.ingest_items) == 1

    # Repeat poll with the exact same item 10 times
    for _ in range(10):
        count_repeat = agent.step_sync()
        assert count_repeat == 0

    # Total in queue remains 1
    assert len(bus.ingest_items) == 1
