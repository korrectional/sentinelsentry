"""
Audit Round 4 test suite verifying:
1. Radio 10-code collision resolution in triage location extraction.
2. Trauma and medical emergency keyword detection and classification.
3. Dashboard table column constraints and no-wrap formatting.
4. Lazy import of BenchmarkRunner in sentinelsentry.eval package.
"""

import pytest
from sentinelsentry.agents.triage import LocalTriageEngine
from sentinelsentry.agents.geocoder import GeocoderAgent
from sentinelsentry.agents.dashboard import AlertDashboardAgent
from sentinelsentry.agents.base import SwarmEventBus
from sentinelsentry.schema import IncidentType, RawFeedItem, EnrichedIncident, TriageOutput


def test_radio_10_codes_address_collision_resolved():
    """Verify radio codes like 10-33 / CODE 3 are stripped so addresses aren't mangled."""
    engine = LocalTriageEngine()

    raw_text = "10-33 CODE 3 SHOTS REPORTED GLENWOOD AVE & PEACE ST. CALLER REPORTS LOUD POPPING VEHICLE SPEEDING OFF."
    triage = engine.predict(raw_text, feed_source="cad_police")

    assert triage.incident_type == IncidentType.CIVIL_UNREST.value
    # Must NOT contain "33 CODE 3" as address
    assert "33 CODE 3" not in triage.location_text
    assert "glenwood ave & peace st" in triage.location_text.lower()

    # Verify geocoder resolves this extracted location
    bus = SwarmEventBus()
    geocoder = GeocoderAgent(bus=bus, enable_network=False)
    resolved = geocoder._lookup_local_cache(triage.location_text)
    assert resolved is not None
    assert resolved.is_resolved is True
    assert pytest.approx(resolved.latitude, 0.001) == 35.7892
    assert pytest.approx(resolved.longitude, 0.001) == -78.6475


def test_highway_trauma_and_medical_classification():
    """Verify trauma reports on highways classify as Vehicle Accident and cardiac calls as Medical."""
    engine = LocalTriageEngine()

    # 1. Highway trauma incident
    text_highway = "TRAUMA CODE: 3 patients I-40 EB Mile 287 Cary exit. 1 pediatric red tag critical, 2 yellow tag trauma, Duke Life Flight requested on standby."
    triage_hwy = engine.predict(text_highway, feed_source="ems_scanner")
    assert triage_hwy.incident_type == IncidentType.VEHICLE_ACCIDENT.value
    assert triage_hwy.casualty_risk_level >= 4
    assert "I-40" in triage_hwy.location_text

    # 2. Cardiac arrest incident
    text_cardiac = "ECHO RESPONSE: 3000 New Bern Ave nr lobby. 58yo male collapsed unresponsive CPR in progress AED advised shock."
    triage_cardiac = engine.predict(text_cardiac, feed_source="ems_scanner")
    assert triage_cardiac.incident_type == IncidentType.MEDICAL_EMERGENCY.value
    assert triage_cardiac.casualty_risk_level >= 4
    assert "3000 New Bern Ave" in triage_cardiac.location_text


def test_eval_package_lazy_import():
    """Verify sentinelsentry.eval exports BenchmarkRunner lazily without raising AttributeError."""
    import sentinelsentry.eval as eval_pkg
    runner_cls = getattr(eval_pkg, "BenchmarkRunner")
    assert runner_cls.__name__ == "BenchmarkRunner"

    with pytest.raises(AttributeError):
        _ = getattr(eval_pkg, "NonExistentAttribute")


def test_dashboard_table_column_constraints():
    """Verify render_board_table sets no_wrap and proper constraints on Risk, Cred, and Time."""
    bus = SwarmEventBus()
    dash = AlertDashboardAgent(bus=bus)

    inc = EnrichedIncident(
        incident_id="INC-COL-TEST",
        feed_item=RawFeedItem(id="rf-1", feed_id="cad_police", feed_name="CAD", raw_text="Fire"),
        triage=TriageOutput(
            incident_type=IncidentType.STRUCTURE_FIRE.value,
            casualty_risk_level=5,
            credibility_score=0.95,
            location_text="1200 S Saunders St",
        ),
        is_escalated=True,
    )
    dash.process_incident(inc)

    table = dash.render_board_table(limit=5)
    cols = {col.header: col for col in table.columns}

    assert "Time" in cols and cols["Time"].no_wrap is True
    assert "Risk" in cols and cols["Risk"].no_wrap is True
    assert "Cred" in cols and cols["Cred"].no_wrap is True
    assert cols["Risk"].min_width == 7
    assert cols["Cred"].min_width == 5
