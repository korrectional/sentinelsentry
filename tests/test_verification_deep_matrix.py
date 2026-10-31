"""
Deep test matrix for the Cross-Feed Verification Agent:
- Sliding 10-minute time window cutoff
- Cross-feed vs same-feed isolation
- Confidence delta calibration
- Auto-escalation threshold enforcement
"""

from datetime import datetime, timezone, timedelta
from sentinelsentry.schema import RawFeedItem, EnrichedIncident, TriageOutput
from sentinelsentry.agents.base import SwarmEventBus
from sentinelsentry.agents.verification import VerificationAgent


def test_verification_same_feed_never_corroborates():
    """Verify multiple items from the SAME feed do not trigger corroboration."""
    bus = SwarmEventBus()
    agent = VerificationAgent(bus=bus)

    # Item 1 from CAD Police
    item1 = RawFeedItem(
        id="cad-01",
        feed_id="cad_police",
        feed_name="CAD Police",
        title="Fire Report",
        raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST FLAMES",
        published_at=datetime.now(timezone.utc),
    )
    inc = EnrichedIncident(
        incident_id="INC-cad-01",
        feed_item=item1,
        triage=TriageOutput(
            incident_type="Structure Fire",
            casualty_risk_level=4,
            credibility_score=0.70,
            location_text="1200 S Saunders St",
        ),
    )

    # Item 2 also from CAD Police
    item2 = RawFeedItem(
        id="cad-02",
        feed_id="cad_police",
        feed_name="CAD Police",
        title="Fire Update",
        raw_text="10-70 UPDATE 1200 S SAUNDERS ST UNITS EN ROUTE",
        published_at=datetime.now(timezone.utc),
    )
    bus.record_recent_item_sync(item2)

    agent.verify_incident_sync(inc)

    # Must NOT count as cross-feed corroboration
    assert inc.verification is not None
    assert len(inc.verification.corroborating_feed_ids) == 0
    assert inc.verification.confidence_delta == 0.0


def test_verification_sliding_window_stale_cutoff():
    """Verify items older than the 10-minute window (600s) are ignored."""
    bus = SwarmEventBus()
    agent = VerificationAgent(bus=bus)

    now = datetime.now(timezone.utc)

    # High-risk incident right now
    item_now = RawFeedItem(
        id="now-01",
        feed_id="cad_police",
        feed_name="CAD Police",
        title="Fire Current",
        raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST FLAMES",
        published_at=now,
    )
    inc = EnrichedIncident(
        incident_id="INC-now-01",
        feed_item=item_now,
        triage=TriageOutput(
            incident_type="Structure Fire",
            casualty_risk_level=4,
            credibility_score=0.70,
            location_text="1200 S Saunders St",
        ),
    )

    # Stale citizen post from 15 minutes ago (900 seconds)
    item_stale = RawFeedItem(
        id="stale-01",
        feed_id="social_crisis",
        feed_name="Social Eyewitness",
        title="Old Smoke",
        raw_text="Smoke visible near Saunders St",
        published_at=now - timedelta(minutes=15),
    )
    bus.record_recent_item_sync(item_stale)

    agent.verify_incident_sync(inc)

    # Must NOT corroborate with stale post
    assert inc.verification is not None
    assert len(inc.verification.corroborating_feed_ids) == 0
    assert inc.verification.confidence_delta == 0.0


def test_verification_cross_feed_scoring_and_escalation():
    """Verify distinct feeds (Fire + Social) corroborate, increase credibility, and escalate."""
    bus = SwarmEventBus()
    agent = VerificationAgent(bus=bus)
    now = datetime.now(timezone.utc)

    # Primary incident: Police CAD
    cad_item = RawFeedItem(
        id="cad-01",
        feed_id="cad_police",
        feed_name="CAD Police",
        title="2-Alarm Fire",
        raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST FLAMES ROOF",
        published_at=now,
    )
    inc = EnrichedIncident(
        incident_id="INC-cad-01",
        feed_item=cad_item,
        triage=TriageOutput(
            incident_type="Structure Fire",
            casualty_risk_level=5,
            credibility_score=0.65,
            location_text="1200 S Saunders St",
        ),
    )

    # Cross-feed 1: Wake Fire dispatch 1 minute ago
    fire_item = RawFeedItem(
        id="fire-01",
        feed_id="fire_rescue",
        feed_name="Wake Fire",
        title="Wake Fire Saunders",
        raw_text="STRUCTURE FIRE: 2-alarm commercial at 1200 S Saunders St. Heavy smoke venting 2nd floor",
        published_at=now - timedelta(minutes=1),
    )
    # Cross-feed 2: Social media 30 seconds ago
    social_item = RawFeedItem(
        id="soc-01",
        feed_id="social_crisis",
        feed_name="Social Media",
        title="Explosion",
        raw_text="Holy smoke huge explosions coming from warehouse on Saunders St",
        published_at=now - timedelta(seconds=30),
    )
    bus.record_recent_item_sync(fire_item)
    bus.record_recent_item_sync(social_item)

    agent.verify_incident_sync(inc)

    assert inc.verification is not None
    assert "fire_rescue" in inc.verification.corroborating_feed_ids
    assert "social_crisis" in inc.verification.corroborating_feed_ids
    assert inc.verification.confidence_delta >= 0.35
    assert inc.triage.credibility_score >= 0.95
    assert inc.is_escalated is True
    assert inc.human_status == "ESCALATED"
