"""
Unit and integration tests for Verification precision, Dispatcher decision support actions,
Geocoder landmark resolution, and Coordinator async start/stop lifecycle.
"""

import pytest
import asyncio
from datetime import datetime, timezone

from sentinelsentry.config import FeedConfig
from sentinelsentry.schema import RawFeedItem, EnrichedIncident, TriageOutput, IncidentType
from sentinelsentry.agents.base import SwarmEventBus, SwarmCoordinator
from sentinelsentry.agents.ingest import IngestAgent
from sentinelsentry.agents.verification import VerificationAgent
from sentinelsentry.agents.geocoder import GeocoderAgent
from sentinelsentry.agents.dashboard import AlertDashboardAgent
from sentinelsentry.agents.triage import LocalTriageEngine


def test_verification_agent_precision_no_false_corroboration():
    """Verify that unrelated incidents with different locations do NOT cross-corroborate."""
    bus = SwarmEventBus()
    agent = VerificationAgent(bus=bus)

    # Incident 1: Fire on Saunders St
    item1 = RawFeedItem(
        id="item-001",
        feed_id="cad_police",
        feed_name="CAD Police",
        title="Fire Saunders",
        raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF 1 SUBJ INSIDE",
        published_at=datetime.now(timezone.utc),
    )
    inc1 = EnrichedIncident(
        incident_id="inc-001",
        feed_item=item1,
        triage=TriageOutput(
            incident_type="Structure Fire",
            casualty_risk_level=4,
            credibility_score=0.70,
            location_text="1200 S Saunders St, Raleigh",
        ),
    )

    # Incident 2: Severe weather on Crabtree Creek (completely different location & scenario)
    item2 = RawFeedItem(
        id="item-002",
        feed_id="weather_skywarn",
        feed_name="NWS Skywarn",
        title="Flood Crabtree",
        raw_text="FLASH FLOOD: Crabtree Creek exceeding gauge 18ft water over roadway",
        published_at=datetime.now(timezone.utc),
    )

    # Add item2 to recent items buffer
    bus.record_recent_item_sync(item2)

    # Verify incident 1: should NOT corroborate with item2
    agent.verify_incident_sync(inc1)

    assert inc1.verification is not None
    assert len(inc1.verification.corroborating_feed_ids) == 0
    assert inc1.verification.confidence_delta == 0.0
    assert inc1.triage.credibility_score == 0.70


def test_verification_agent_true_corroboration():
    """Verify that multi-feed reports sharing same location corridor corroborate and boost credibility."""
    bus = SwarmEventBus()
    agent = VerificationAgent(bus=bus)

    # Incident 1: Fire on Saunders St (Police CAD)
    item1 = RawFeedItem(
        id="item-cad-01",
        feed_id="cad_police",
        feed_name="CAD Police",
        title="Fire Saunders",
        raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES SHOWING ROOF",
        published_at=datetime.now(timezone.utc),
    )
    inc1 = EnrichedIncident(
        incident_id="inc-cad-01",
        feed_item=item1,
        triage=TriageOutput(
            incident_type="Structure Fire",
            casualty_risk_level=4,
            credibility_score=0.70,
            location_text="1200 S Saunders St, Raleigh",
        ),
    )

    # Corroborating citizen eyewitness report on Saunders St
    item_social = RawFeedItem(
        id="item-soc-01",
        feed_id="social_crisis",
        feed_name="Social Eyewitness",
        title="Smoke Saunders",
        raw_text="Huge black smoke and flames pouring out of warehouse on Saunders St right now downtown",
        published_at=datetime.now(timezone.utc),
    )
    bus.record_recent_item_sync(item_social)

    agent.verify_incident_sync(inc1)

    assert inc1.verification is not None
    assert "social_crisis" in inc1.verification.corroborating_feed_ids
    assert inc1.verification.confidence_delta > 0.10
    assert inc1.triage.credibility_score > 0.70
    assert inc1.is_escalated is True
    assert inc1.human_status == "ESCALATED"


def test_dispatcher_decision_support_actions():
    """Verify human dispatcher action workflow: dispatch, dismiss, manual escalate."""
    bus = SwarmEventBus()
    dash = AlertDashboardAgent(bus=bus)

    # Create dummy high-risk escalated incident
    item = RawFeedItem(
        id="test-item",
        feed_id="fire_rescue",
        feed_name="Wake Fire",
        title="Rollover MVA",
        raw_text="10-50 PI MULTI-VEHICLE I-40 EB MM 287. 2 PINNED JAWS IN SERVICE",
        published_at=datetime.now(timezone.utc),
    )
    inc = EnrichedIncident(
        incident_id="INC-DISPATCH-911",
        feed_item=item,
        triage=TriageOutput(
            incident_type="Vehicle Accident / Rescue",
            casualty_risk_level=5,
            credibility_score=0.95,
            location_text="I-40 Eastbound MM 287 nr Cary",
        ),
        is_escalated=True,
        human_status="ESCALATED",
    )
    dash.process_incident(inc)

    # 1. Check pending escalations
    pending = dash.get_pending_escalations()
    assert len(pending) == 1
    assert pending[0].incident_id == "INC-DISPATCH-911"

    # 2. Test Dispatcher Action: Dispatch Units
    dispatched = dash.dispatch_incident(
        incident_id="INC-DISPATCH-911",
        units=["Rescue 1", "Engine 8", "EMS Medic 12"],
        dispatcher_notes="Heavy extrication authorized Code 3.",
    )
    assert "DISPATCHED" in dispatched.human_status
    assert dispatched.is_escalated is False
    assert "Rescue 1" in dispatched.metadata["dispatched_units"]

    # Pending escalations should now be empty
    assert len(dash.get_pending_escalations()) == 0

    # 3. Test Manual Dismissal
    inc2 = EnrichedIncident(
        incident_id="INC-FALSE-001",
        feed_item=item,
        triage=TriageOutput(
            incident_type="False Alarm / Non-Emergency",
            casualty_risk_level=1,
            credibility_score=0.90,
            location_text="1200 S Saunders St",
        ),
        is_escalated=False,
        human_status="ROUTINE",
    )
    dash.process_incident(inc2)
    dismissed = dash.dismiss_incident("INC-FALSE-001", reason="Waterflow testing confirmed")
    assert "DISMISSED" in dismissed.human_status

    # 4. Render incident detail card
    detail = dash.render_incident_detail("INC-DISPATCH-911")
    assert "INCIDENT DETAIL: INC-DISPATCH-911" in detail
    assert "Vehicle Accident / Rescue" in detail


def test_triage_engine_ten_code_location_precision():
    """Verify that 10-codes (10-70, 10-33, 10-50) are not extracted as street numbers."""
    engine = LocalTriageEngine()

    out1 = engine.predict("10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF")
    assert out1.location_text != "70 COMM BLDG 1200 S SAUNDERS ST"
    assert "1200 S Saunders" in out1.location_text or "Saunders" in out1.location_text

    out2 = engine.predict("10-33 CODE 3 SHOTS REPORTED 500 S SALISBURY ST")
    assert "33 CODE 3" not in out2.location_text
    assert "500 S Salisbury" in out2.location_text or "Salisbury" in out2.location_text


def test_feed_server_url_normalization():
    """Verify that FeedServerManager handles both hyphenated and underscored feed IDs."""
    from sentinelsentry.feeds.server import FeedServerManager
    manager = FeedServerManager()

    # Underscored ID
    xml_under = manager.get_feed_xml("cad_police")
    assert "<rss version=\"2.0\"" in xml_under

    # Hyphenated ID (matching config.py default URLs)
    xml_hyphen = manager.get_feed_xml("cad-police")
    assert "<rss version=\"2.0\"" in xml_hyphen

    # XML extension included
    xml_ext = manager.get_feed_xml("cad-police.xml")
    assert "<rss version=\"2.0\"" in xml_ext


def test_coordinator_async_mode_start_and_stop():
    """Verify Mode B: Asynchronous concurrent task start and stop lifecycle."""
    bus = SwarmEventBus()
    coord = SwarmCoordinator(bus=bus)

    cfg = FeedConfig(id="test_cad", name="CAD Test", description="Desc", url="http://local/test")
    ingest = IngestAgent(feed_config=cfg, bus=bus, custom_fetcher=lambda: [])
    coord.register_ingest_agent(ingest)

    # Verify coordinator start and stop methods exist and properly manage state
    assert hasattr(coord, "start")
    assert hasattr(coord, "stop")
    assert coord._running is False


def test_triage_engine_messy_citizen_and_scanner_precision():
    """Verify triage extracts high-risk categories and locations from messy citizen & radio text."""
    engine = LocalTriageEngine()

    # 1. Swiftwater / Water rescue
    t1 = engine.predict("WATER RESCUE: 2 vehicles stalled in swift water to windshields at Atlantic Ave & Hodges St nr Crabtree. Callers stranded on car roofs yelling for help.")
    assert t1.incident_type == "Severe Weather / Flooding"
    assert t1.casualty_risk_level == 5
    assert "Atlantic Ave & Hodges St" in t1.location_text

    # 2. Trauma code with Life Flight
    t2 = engine.predict("TRAUMA CODE: 3 patients I-40 EB Mile 287 Cary exit. 1 pediatric red tag critical, 2 yellow tag trauma, Duke Life Flight requested on standby.")
    assert t2.incident_type == "Vehicle Accident / Rescue"
    assert t2.casualty_risk_level == 5
    assert "I-40 EB Mile 287" in t2.location_text

    # 3. Warehouse explosion eyewitness tweet
    t3 = engine.predict("Holy smoke huge explosions coming from warehouse district on Saunders St near downtown Raleigh right now sirens everywhere!! Thick black smoke filling sky")
    assert t3.incident_type == "Structure Fire"
    assert t3.casualty_risk_level >= 4
    assert "Saunders St" in t3.location_text

    # 4. Rotten eggs / gas smell citizen report
    t4 = engine.predict("Everyone running out of 401 Fayetteville building cops screaming move back intense gas smell like rotten eggs sirens everywhere downtown")
    assert t4.incident_type == "Infrastructure Failure / Gas Leak"
    assert t4.casualty_risk_level >= 3
    assert "401 Fayetteville" in t4.location_text


def test_verification_unknown_location_never_auto_escalates():
    """Safety guarantee: Incidents with unknown locations or unclassified types are flagged for review, never auto-escalated."""
    bus = SwarmEventBus()
    agent = VerificationAgent(bus=bus)

    # Incident with risk 5 but unknown location
    raw_item = RawFeedItem(
        id="item-mystery-01",
        feed_id="ems_scanner",
        feed_name="EMS Scanner",
        title="Unknown Trauma",
        raw_text="Severe trauma patient in critical condition, location unreadable due to radio static.",
        published_at=datetime.now(timezone.utc),
    )
    inc = EnrichedIncident(
        incident_id="INC-MYSTERY-001",
        feed_item=raw_item,
        triage=TriageOutput(
            incident_type="Medical Emergency",
            casualty_risk_level=5,
            credibility_score=0.95,
            location_text="Unknown",
        ),
    )

    verified = agent.verify_incident_sync(inc)
    # Must NOT be auto-escalated
    assert verified.is_escalated is False
    assert "NEEDS REVIEW" in verified.human_status


def test_geocoder_fuzzy_and_landmark_resolution():
    """Verify all extracted locations resolve cleanly in Geocoder landmark cache."""
    bus = SwarmEventBus()
    geocoder = GeocoderAgent(bus=bus, enable_network=False)

    test_locations = [
        "I-40 EB MM 287 NR CARY EXT RAMP",
        "Saunders St near downtown Raleigh",
        "I-40 Eastbound by Cary exit",
        "401 Fayetteville building",
        "Atlantic Ave & Hodges St",
        "1200 S Saunders St",
        "3000 New Bern Ave",
        "Crabtree Creek Basin",
        "Crabtree Valley Mall",
    ]

    for loc in test_locations:
        res = geocoder.resolve_location_sync(loc)
        assert res.is_resolved is True, f"Failed to resolve '{loc}'"
        assert res.latitude != 0.0
        assert res.longitude != 0.0
        assert res.provider == "Local Raleigh Landmark Cache"
