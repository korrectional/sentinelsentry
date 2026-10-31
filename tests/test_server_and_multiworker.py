"""
Integration test suite for Feed Server Manager, multi-worker triage load balancing,
and alert dashboard state deduplication.
"""

import pytest
from sentinelsentry.feeds.server import feed_manager
from sentinelsentry.schema import TriageOutput, RawFeedItem, EnrichedIncident
from sentinelsentry.agents.base import SwarmEventBus, SwarmCoordinator
from sentinelsentry.agents.triage import TriageAgent, LocalTriageEngine
from sentinelsentry.agents.dashboard import AlertDashboardAgent


def test_feed_server_manager_normalization_and_aliases():
    """Verify FeedServerManager correctly normalizes hyphens, underscores, and .xml extensions."""
    # 1. Standard underscore with .xml
    xml1 = feed_manager.get_feed_xml("cad_police.xml")
    assert "<rss version=\"2.0\"" in xml1
    assert "Police" in xml1 or "CAD" in xml1

    # 2. Hyphenated with .xml (matching settings.FEEDS url)
    xml2 = feed_manager.get_feed_xml("cad-police.xml")
    assert "<rss version=\"2.0\"" in xml2

    # 3. Without .xml extension
    xml3 = feed_manager.get_feed_xml("cad_police")
    assert "<rss version=\"2.0\"" in xml3

    # 4. Non-existent feed should raise KeyError
    with pytest.raises(KeyError):
        feed_manager.get_feed_xml("non_existent_feed")


def test_feed_server_manager_inject_and_trigger():
    """Verify custom item injection and scenario triggering work with hyphenated IDs."""
    # 1. Inject custom item using hyphenated feed id
    guid = feed_manager.inject_item(
        feed_id="cad-police",
        raw_text="10-50 PI INTERSECTION GLENWOOD AVE & PEACE ST 2 PINNED",
        title="API Test Incident",
        scenario_id="TEST-API-001",
    )
    assert len(guid) > 0

    # Verify item appears in feed XML
    xml = feed_manager.get_feed_xml("cad-police.xml")
    assert guid in xml
    assert "GLENWOOD AVE" in xml

    # 2. Trigger multi-agency crisis burst scenario
    guids = feed_manager.trigger_scenario("SCN-001-WAREHOUSE-FIRE")
    assert len(guids) >= 3


def test_multi_worker_triage_load_balancing():
    """Verify multiple triage workers in coordinator share the load fairly."""
    bus = SwarmEventBus()
    coord = SwarmCoordinator(bus=bus)

    worker1 = TriageAgent(bus, worker_id=1, engine=LocalTriageEngine())
    worker2 = TriageAgent(bus, worker_id=2, engine=LocalTriageEngine())
    coord.register_triage_worker(worker1)
    coord.register_triage_worker(worker2)

    # Enqueue 10 raw items for triage
    for i in range(10):
        bus.ingest_items.append(
            RawFeedItem(
                id=f"item-{i}",
                feed_id="cad_police",
                feed_name="CAD",
                raw_text=f"Incident dispatch number {i} at 1200 S Saunders St",
            )
        )

    # Run one step
    counts = coord.step()
    assert counts["triaged"] == 10

    # Both workers should have processed items (load balanced 5 and 5)
    assert worker1.processed_count == 5
    assert worker2.processed_count == 5


def test_dashboard_reprocessed_incident_deduplication():
    """Re-delivering or updating an incident must not artificially inflate total_processed count."""
    bus = SwarmEventBus()
    dash = AlertDashboardAgent(bus)

    inc1 = EnrichedIncident(
        incident_id="INC-REPEAT-001",
        feed_item=RawFeedItem(id="raw-1", feed_id="cad_police", feed_name="CAD", raw_text="Fire"),
        triage=TriageOutput(incident_type="Structure Fire", casualty_risk_level=3, credibility_score=0.7, location_text="Loc"),
        is_escalated=False,
    )

    dash.process_incident(inc1)
    assert dash.stats["total_processed"] == 1
    assert dash.stats["routine_count"] == 1
    assert dash.stats["escalated_count"] == 0

    # Deliver updated version of same incident (now escalated after verification)
    inc1_updated = inc1.model_copy()
    inc1_updated.is_escalated = True
    dash.process_incident(inc1_updated)

    # Total unique count should still be 1, but routine moved to escalated!
    assert dash.stats["total_processed"] == 1
    assert dash.stats["escalated_count"] == 1
    assert dash.stats["routine_count"] == 0


def test_location_sanitization_null_and_empty():
    """Verify location_text sanitizer in schema handles null, None, and blank inputs."""
    t1 = TriageOutput(
        incident_type="Structure Fire",
        casualty_risk_level=4,
        credibility_score=0.9,
        location_text=None,
    )
    assert t1.location_text == "Unknown"

    t2 = TriageOutput(
        incident_type="Structure Fire",
        casualty_risk_level=4,
        credibility_score=0.9,
        location_text="   ",
    )
    assert t2.location_text == "Unknown"
