"""
Comprehensive edge cases and QA stress testing for SentinelSentry.
Audits boundary conditions, schema validation clamping, deduplication,
cross-feed corroboration rules, and multi-agent queue dynamics.
"""

import json
from datetime import datetime, timezone, timedelta
from sentinelsentry.schema import (
    TriageOutput,
    RawFeedItem,
    GeocodedLocation,
    CorroborationReport,
    EnrichedIncident,
    IncidentType,
)
from sentinelsentry.config import FeedConfig
from sentinelsentry.feeds.generator import RSSFeedGenerator
from sentinelsentry.feeds.client import RSSFeedClient, DeduplicationCache
from sentinelsentry.agents.base import SwarmEventBus, SwarmCoordinator
from sentinelsentry.agents.ingest import IngestAgent
from sentinelsentry.agents.triage import TriageAgent, LocalTriageEngine
from sentinelsentry.agents.verification import VerificationAgent
from sentinelsentry.agents.geocoder import GeocoderAgent
from sentinelsentry.agents.dashboard import AlertDashboardAgent


# ==============================================================================
# 1. Schema Validation & Clamping Edge Cases
# ==============================================================================

def test_triage_output_boundary_clamping():
    """Verify casualty_risk_level and credibility_score are strictly clamped."""
    # Risk over max (e.g. 99) -> clamps to 5
    t1 = TriageOutput(
        incident_type="Structure Fire",
        casualty_risk_level=99,
        credibility_score=1.5,
        location_text="1200 S Saunders St",
    )
    assert t1.casualty_risk_level == 5
    assert t1.credibility_score == 1.0

    # Risk under min (e.g. -5) -> clamps to 1
    t2 = TriageOutput(
        incident_type="False Alarm / Non-Emergency",
        casualty_risk_level=-5,
        credibility_score=-0.8,
        location_text="Unknown",
    )
    assert t2.casualty_risk_level == 1
    assert t2.credibility_score == 0.0

    # Float risk e.g. "4.2" should round to integer 4
    t3 = TriageOutput(
        incident_type="Medical Emergency",
        casualty_risk_level="4.2",
        credibility_score="0.875",
        location_text="401 Fayetteville St",
    )
    assert t3.casualty_risk_level == 4
    assert t3.credibility_score == 0.88


def test_strict_json_serialization():
    """Verify to_strict_json() generates compact valid single-line JSON."""
    t = TriageOutput(
        incident_type="Structure Fire",
        casualty_risk_level=4,
        credibility_score=0.92,
        location_text="1200 S Saunders St",
    )
    strict_json = t.to_strict_json()
    assert "\n" not in strict_json
    loaded = json.loads(strict_json)
    assert loaded["incident_type"] == "Structure Fire"
    assert loaded["casualty_risk_level"] == 4
    assert loaded["credibility_score"] == 0.92
    assert loaded["location_text"] == "1200 S Saunders St"


# ==============================================================================
# 2. RSS Feed Deduplication & Malformed XML Edge Cases
# ==============================================================================

def test_rss_client_handles_empty_and_malformed_xml():
    """Client must handle empty or broken XML without raising exceptions."""
    cfg = FeedConfig(id="test_feed", name="Test", description="Test", url="http://local/test")
    client = RSSFeedClient(cfg)

    # Empty string
    items_empty = client.parse_feed_content("")
    assert items_empty == []

    # Malformed non-XML string
    items_broken = client.parse_feed_content("THIS IS NOT XML <malformed>>")
    assert isinstance(items_broken, list)

    # XML with missing GUID (should fall back to title + description hash)
    xml_no_guid = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <title>Test Channel</title>
        <item>
          <title>Alert 1</title>
          <description>Gas leak reported on 401 Fayetteville</description>
        </item>
      </channel>
    </rss>"""
    items_no_guid = client.parse_feed_content(xml_no_guid)
    assert len(items_no_guid) == 1
    assert len(items_no_guid[0].id) > 0


# ==============================================================================
# 3. Verification Agent & Cross-Feed Corroboration Rules
# ==============================================================================

def test_verification_agent_same_feed_cannot_corroborate_itself():
    """Items from the SAME feed should never count as cross-feed corroboration."""
    bus = SwarmEventBus()
    va = VerificationAgent(bus)

    # Original incident from 'social_crisis'
    now = datetime.now(timezone.utc)
    raw_item1 = RawFeedItem(
        id="item-soc-1",
        feed_id="social_crisis",
        feed_name="Social Stream",
        raw_text="Huge fire at 1200 S Saunders St downtown!!",
        published_at=now,
    )
    inc = EnrichedIncident(
        incident_id="INC-1",
        feed_item=raw_item1,
        triage=TriageOutput(
            incident_type="Structure Fire",
            casualty_risk_level=4,
            credibility_score=0.45,
            location_text="1200 S Saunders St",
        ),
    )

    # Another tweet from the SAME feed within 10 seconds
    raw_item2 = RawFeedItem(
        id="item-soc-2",
        feed_id="social_crisis",
        feed_name="Social Stream",
        raw_text="Seeing flames and smoke on Saunders St!",
        published_at=now + timedelta(seconds=10),
    )
    bus.record_recent_item_sync(raw_item2)

    # Verify incident
    verified = va.verify_incident_sync(inc)

    # Because candidate was from 'social_crisis', it cannot corroborate itself
    assert verified.verification.confidence_delta == 0.0
    assert verified.triage.credibility_score == 0.45
    # Should NOT be escalated because credibility is only 0.45 (< 0.60 threshold)
    assert not verified.is_escalated


def test_verification_agent_cross_feed_boost_and_auto_escalation():
    """An official CAD post corroborating a citizen report triggers auto-escalation."""
    bus = SwarmEventBus()
    va = VerificationAgent(bus)

    now = datetime.now(timezone.utc)
    raw_social = RawFeedItem(
        id="item-soc-1",
        feed_id="social_crisis",
        feed_name="Social Stream",
        raw_text="Huge fire at 1200 S Saunders St downtown!!",
        published_at=now,
    )
    inc = EnrichedIncident(
        incident_id="INC-1",
        feed_item=raw_social,
        triage=TriageOutput(
            incident_type="Structure Fire",
            casualty_risk_level=4,
            credibility_score=0.45,
            location_text="1200 S Saunders St",
        ),
    )

    # An official Fire dispatch report from a DIFFERENT feed within time window
    raw_fire = RawFeedItem(
        id="item-fire-1",
        feed_id="fire_rescue",
        feed_name="Wake Fire",
        raw_text="STRUCTURE FIRE: 2-alarm at 1200 S Saunders St",
        published_at=now + timedelta(seconds=30),
    )
    bus.record_recent_item_sync(raw_fire)

    verified = va.verify_incident_sync(inc)

    # Official cross-feed match should boost credibility from 0.45 to 0.70 (+0.25)
    assert verified.verification.confidence_delta == 0.25
    assert verified.triage.credibility_score >= 0.70
    # Risk 4 and Credibility 0.70 >= 0.60 -> Auto-escalate!
    assert verified.is_escalated
    assert verified.human_status == "ESCALATED"


def test_verification_agent_out_of_window_does_not_corroborate():
    """A report older than 10 minutes (600 seconds) must not corroborate."""
    bus = SwarmEventBus()
    va = VerificationAgent(bus)

    now = datetime.now(timezone.utc)
    raw_incident = RawFeedItem(
        id="item-cad-1",
        feed_id="cad_police",
        feed_name="Police CAD",
        raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST",
        published_at=now,
    )
    inc = EnrichedIncident(
        incident_id="INC-CAD",
        feed_item=raw_incident,
        triage=TriageOutput(
            incident_type="Structure Fire",
            casualty_risk_level=4,
            credibility_score=0.85,
            location_text="1200 S Saunders St",
        ),
    )

    # Stale candidate from 15 minutes ago (900 seconds)
    stale_item = RawFeedItem(
        id="item-stale",
        feed_id="fire_rescue",
        feed_name="Wake Fire",
        raw_text="STRUCTURE FIRE: 1200 S Saunders St",
        published_at=now - timedelta(seconds=900),
    )
    bus.record_recent_item_sync(stale_item)

    verified = va.verify_incident_sync(inc)
    assert len(verified.verification.corroborating_feed_ids) == 0


# ==============================================================================
# 4. Geocoder Agent Landmark Cache & Fallbacks
# ==============================================================================

def test_geocoder_partial_landmark_match():
    """Verify regional landmarks resolve coordinates even with partial strings."""
    bus = SwarmEventBus()
    ga = GeocoderAgent(bus, enable_network=False)

    # Exact key
    g1 = ga.resolve_location_sync("1200 S Saunders St")
    assert g1.is_resolved
    assert abs(g1.latitude - 35.7592) < 0.001
    assert abs(g1.longitude - (-78.6534)) < 0.001

    # Mixed casing and extra words
    g2 = ga.resolve_location_sync("Accident near Crabtree Valley Mall lower lot")
    assert g2.is_resolved
    assert "Crabtree" in g2.display_name

    # Unknown location fallback
    g3 = ga.resolve_location_sync("Unknown")
    assert not g3.is_resolved
    assert g3.latitude is None


# ==============================================================================
# 5. Local Triage Engine Robustness (Markdown Stripping & Edge Cases)
# ==============================================================================

def test_local_triage_engine_markdown_stripping():
    """Engine must parse JSON even when wrapped in markdown code fences."""
    engine = LocalTriageEngine()
    wrapped_json = """```json
    {
      "incident_type": "Vehicle Accident / Rescue",
      "casualty_risk_level": 5,
      "credibility_score": 0.95,
      "location_text": "I-40 EB MM 287"
    }
    ```"""
    result = engine._extract_json(wrapped_json, fallback_raw_text="dummy text")
    assert result.incident_type == "Vehicle Accident / Rescue"
    assert result.casualty_risk_level == 5
    assert result.credibility_score == 0.95
    assert result.location_text == "I-40 EB MM 287"


def test_triage_engine_extreme_keywords_detection():
    """Trapped occupants / CPR in progress must assign high risk level (4-5)."""
    engine = LocalTriageEngine()

    res_trapped = engine.predict("10-50 PI WITH 2 OCCUPANTS TRAPPED HEAVY EXTRICATION")
    assert res_trapped.casualty_risk_level >= 4
    assert res_trapped.incident_type == IncidentType.VEHICLE_ACCIDENT.value

    res_false_alarm = engine.predict("Automatic waterflow alarm accidental maintenance test cancel units")
    assert res_false_alarm.casualty_risk_level == 1
    assert res_false_alarm.incident_type == IncidentType.NON_EMERGENCY.value
