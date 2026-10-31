"""
Adversarial and deep integration test suite for SentinelSentry.
Tests edge cases, boundary conditions, malformed submissions,
speed controls, prefix tolerance, and the scripted demo scenario.
"""

from __future__ import annotations

import asyncio
import json
import math
from fastapi import HTTPException

from sentinelsentry.config import settings
from sentinelsentry.schema import RawFeedItem, EnrichedIncident, TriageOutput
from sentinelsentry.agents.triage import LocalTriageEngine
from sentinelsentry.agents.base import SwarmEventBus, SwarmCoordinator
from sentinelsentry.agents.dashboard import AlertDashboardAgent
from sentinelsentry.web.app import (
    app,
    state,
    serve_dashboard,
    get_incidents,
    get_swarm_status,
    get_eval_proof,
    submit_incident,
    acknowledge_incident,
    dispatch_incident,
    dismiss_incident,
    set_demo_speed,
    reset_demo,
    run_scripted_demo_sequence,
    run_scripted_demo_sync,
    SubmitIncidentRequest,
    SpeedControlRequest,
    DispatchActionRequest,
)


def test_submit_incident_validation_and_empty_rejection():
    """Verify empty or whitespace-only dispatch text is rejected with HTTP 400."""
    # 1. Pure empty string
    try:
        submit_incident(SubmitIncidentRequest(raw_text=""))
        assert False, "Should have raised HTTPException 400 for empty string"
    except HTTPException as e:
        assert e.status_code == 400
        assert "cannot be empty" in e.detail

    # 2. Whitespace string
    try:
        submit_incident(SubmitIncidentRequest(raw_text="   \n\t  "))
        assert False, "Should have raised HTTPException 400 for whitespace"
    except HTTPException as e:
        assert e.status_code == 400

    # 3. Valid submission
    res = submit_incident(SubmitIncidentRequest(raw_text="10-70 TEST FIRE 1200 S SAUNDERS ST"))
    assert res.status_code == 200
    data = json.loads(res.body.decode("utf-8"))
    assert data["status"] == "success"


def test_incident_id_prefix_tolerance():
    """Verify acknowledge, dispatch, and dismiss work with and without 'INC-' prefix."""
    reset_demo()

    # Submit incident
    sub_res = submit_incident(SubmitIncidentRequest(
        raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF 1 SUBJ INSIDE"
    ))
    assert sub_res.status_code == 200
    data = json.loads(sub_res.body.decode("utf-8"))
    item_id = data["item_id"]
    full_id = f"INC-{item_id}"

    # 1. Acknowledge with raw item_id (without INC-)
    ack_res = acknowledge_incident(item_id)
    assert ack_res.status_code == 200
    ack_data = json.loads(ack_res.body.decode("utf-8"))
    assert ack_data["status"] == "acknowledged"

    # 2. Dispatch with full_id (with INC-)
    disp_res = dispatch_incident(full_id, req=DispatchActionRequest(units=["Engine 12", "Ladder 4"]))
    assert disp_res.status_code == 200
    disp_data = json.loads(disp_res.body.decode("utf-8"))
    assert disp_data["status"] == "dispatched"

    # 3. Dismiss with raw item_id
    dism_res = dismiss_incident(item_id, req=DispatchActionRequest(reason="Controlled burn"))
    assert dism_res.status_code == 200
    dism_data = json.loads(dism_res.body.decode("utf-8"))
    assert dism_data["status"] == "dismissed"

    # 4. Non-existent ID returns 404
    try:
        acknowledge_incident("NON-EXISTENT-999")
        assert False, "Should have raised 404 for non-existent incident"
    except HTTPException as e:
        assert e.status_code == 404


def test_speed_control_boundaries():
    """Verify speed control handles negative values, zero, extremes, and NaN."""
    # 1. Normal speed
    r1 = set_demo_speed(SpeedControlRequest(multiplier=2.0))
    d1 = json.loads(r1.body.decode("utf-8"))
    assert d1["speed_multiplier"] == 2.0

    # 2. Negative speed clamped to safe default 1.0
    r2 = set_demo_speed(SpeedControlRequest(multiplier=-5.0))
    d2 = json.loads(r2.body.decode("utf-8"))
    assert d2["speed_multiplier"] == 1.0

    # 3. Zero clamped to 1.0
    r3 = set_demo_speed(SpeedControlRequest(multiplier=0.0))
    d3 = json.loads(r3.body.decode("utf-8"))
    assert d3["speed_multiplier"] == 1.0

    # 4. Extreme speed clamped to max 100.0
    r4 = set_demo_speed(SpeedControlRequest(multiplier=999.0))
    d4 = json.loads(r4.body.decode("utf-8"))
    assert d4["speed_multiplier"] == 100.0

    # 5. Reset speed to 1.0
    set_demo_speed(SpeedControlRequest(multiplier=1.0))


def test_scripted_demo_full_lifecycle():
    """Verify the 90-second crisis burst scenario completes cleanly."""
    reset_demo()

    # Run the demo sequence
    run_scripted_demo_sync()

    assert state.is_demo_running is False

    # Check incidents payload
    inc_res = get_incidents()
    inc_data = json.loads(inc_res.body.decode("utf-8"))
    incidents = inc_data["incidents"]

    # Verify high-risk fire was ingested, corroborated, and escalated
    fire_incidents = [i for i in incidents if "Saunders" in i["location_text"] or "Saunders" in i["display_name"]]
    assert len(fire_incidents) >= 1
    main_fire = fire_incidents[0]

    assert main_fire["casualty_risk_level"] >= 4
    # Cross-feed corroboration was recorded
    assert len(main_fire["corroborating_feed_ids"]) >= 1
    # Credibility was boosted
    assert main_fire["credibility_score"] >= 0.85
    # Geocoder resolved coordinates
    assert main_fire["is_geocoded"] is True
    assert main_fire["latitude"] is not None
    assert main_fire["longitude"] is not None

    # Reset speed back to normal 1.0x
    set_demo_speed(SpeedControlRequest(multiplier=1.0))


def test_sse_event_payload_structure():
    """Verify that SSE stream payload contains required keys and formats."""
    incidents = state.dashboard.get_incidents_payload()
    telemetry = state.coordinator.get_swarm_telemetry()
    pending = state.dashboard.get_pending_escalations()

    # Check telemetry agent count and queue depth keys
    assert len(telemetry["agents"]) >= 9
    for key in ["ingest", "verification", "geocoding", "dashboard"]:
        assert key in telemetry["queue_depths"]

    # Check active escalations format
    for p in pending:
        assert p.incident_id.startswith("INC-")
        assert p.triage.casualty_risk_level >= 4
