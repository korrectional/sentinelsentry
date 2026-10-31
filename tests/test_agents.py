"""
Integration test for multi-agent swarm pipeline and escalation using SwarmCoordinator.
"""

from sentinelsentry.config import FeedConfig
from sentinelsentry.agents.base import SwarmEventBus, SwarmCoordinator
from sentinelsentry.agents.ingest import IngestAgent
from sentinelsentry.agents.triage import TriageAgent, LocalTriageEngine
from sentinelsentry.agents.verification import VerificationAgent
from sentinelsentry.agents.geocoder import GeocoderAgent
from sentinelsentry.agents.dashboard import AlertDashboardAgent
from sentinelsentry.feeds.generator import RSSFeedGenerator


def test_swarm_coordinator_pipeline_and_escalation():
    bus = SwarmEventBus()
    coordinator = SwarmCoordinator(bus=bus)

    # 1. Setup simulated feed generators
    gen_fire = RSSFeedGenerator(
        feed_id="fire_rescue",
        feed_name="Wake Fire Dispatch",
        description="Fire operations",
    )
    gen_police = RSSFeedGenerator(
        feed_id="cad_police",
        feed_name="Raleigh 911 CAD",
        description="Police CAD",
    )

    # 2. Add corroborating high-severity warehouse fire items
    gen_fire.add_item(
        raw_text="STRUCTURE FIRE: 2-alarm commercial at 1200 S Saunders St, heavy smoke venting through roof, occupant trapped",
        title="2-Alarm Fire Saunders",
        scenario_id="SCN-001",
    )
    gen_police.add_item(
        raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF EVAC IN PROGRESS 1 SUBJ INSIDE",
        title="CAD: 10-70 Saunders",
        scenario_id="SCN-001",
    )

    # 3. Create Agents
    cfg_fire = FeedConfig(id="fire_rescue", name="Wake Fire", description="Fire", url="http://local/fire")
    cfg_police = FeedConfig(id="cad_police", name="Raleigh CAD", description="CAD", url="http://local/police")

    ingest_fire = IngestAgent(cfg_fire, bus, custom_fetcher=lambda: gen_fire.to_raw_feed_items())
    ingest_police = IngestAgent(cfg_police, bus, custom_fetcher=lambda: gen_police.to_raw_feed_items())
    triage_worker = TriageAgent(bus, worker_id=1, engine=LocalTriageEngine())
    verification_agent = VerificationAgent(bus)
    geocoder_agent = GeocoderAgent(bus, enable_network=False)
    dashboard_agent = AlertDashboardAgent(bus)

    # 4. Register with Coordinator
    coordinator.register_ingest_agent(ingest_fire)
    coordinator.register_ingest_agent(ingest_police)
    coordinator.register_triage_worker(triage_worker)
    coordinator.set_verification_agent(verification_agent)
    coordinator.set_geocoder_agent(geocoder_agent)
    coordinator.set_dashboard_agent(dashboard_agent)

    # 5. Drain the entire pipeline
    rounds = coordinator.drain(max_rounds=5)
    assert len(rounds) > 0

    # 6. Verify assertions
    assert dashboard_agent.stats["total_processed"] == 2
    assert dashboard_agent.stats["escalated_count"] >= 1
    assert dashboard_agent.stats["corroborated_count"] >= 1

    # Check geocoding resolution
    resolved = [inc for inc in dashboard_agent.incidents.values() if inc.geocoded and inc.geocoded.is_resolved]
    assert len(resolved) >= 1
    assert "Saunders St" in resolved[0].geocoded.display_name

    # Check table output
    table = dashboard_agent.render_board_table()
    assert table is not None
    assert len(table.rows) == 2
