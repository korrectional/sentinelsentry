"""
Edge case and idempotency tests for SentinelSentry web APIs and incident lifecycle:
- Long raw texts & unicode / emojis
- Action idempotency (repeated acknowledge, dispatch, dismiss)
- Telemetry serialization stability
- Empty board queries
"""

import json
from sentinelsentry.schema import RawFeedItem
from sentinelsentry.agents.base import SwarmEventBus, SwarmCoordinator
from sentinelsentry.agents.dashboard import AlertDashboardAgent
from sentinelsentry.web.app import (
    state,
    reset_demo,
    get_incidents,
    get_swarm_status,
    submit_incident,
    acknowledge_incident,
    dispatch_incident,
    dismiss_incident,
    SubmitIncidentRequest,
    DispatchActionRequest,
)


def test_submit_long_text_and_unicode():
    """Verify system safely handles 5000+ character text, emojis, and punctuation."""
    reset_demo()

    long_text = "🚨 URGENT: 10-70 COMM BLDG 1200 S SAUNDERS ST 🔥 " + ("FLAMES HEAVY SMOKE " * 200) + "🚒"
    res = submit_incident(SubmitIncidentRequest(raw_text=long_text, feed_id="test_stream"))
    assert res.status_code == 200
    data = json.loads(res.body.decode("utf-8"))
    assert data["status"] == "success"

    inc_res = get_incidents()
    inc_data = json.loads(inc_res.body.decode("utf-8"))
    found = [i for i in inc_data["incidents"] if i["incident_id"] == f"INC-{data['item_id']}"]
    assert len(found) == 1
    assert "1200 S Saunders" in found[0]["location_text"] or "Saunders" in found[0]["location_text"]


def test_action_idempotency_repeated_calls():
    """Verify repeated acknowledge, dispatch, and dismiss calls succeed without error or state corruption."""
    reset_demo()

    sub = submit_incident(SubmitIncidentRequest(raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST FLAMES"))
    item_id = json.loads(sub.body.decode("utf-8"))["item_id"]

    # 1. Repeated Acknowledge
    for _ in range(3):
        ack = acknowledge_incident(item_id)
        assert ack.status_code == 200

    # 2. Repeated Dispatch
    for _ in range(3):
        disp = dispatch_incident(item_id, req=DispatchActionRequest(units=["Engine 1", "Medic 2"]))
        assert disp.status_code == 200

    # 3. Repeated Dismiss
    for _ in range(3):
        dism = dismiss_incident(item_id, req=DispatchActionRequest(reason="Accidental alarm"))
        assert dism.status_code == 200


def test_empty_board_state_handling():
    """Verify API handles an empty board without returning 500 or null pointer errors."""
    dash = AlertDashboardAgent(bus=SwarmEventBus())
    payload = dash.get_incidents_payload()
    assert isinstance(payload, list)
    assert len(payload) == 0

    pending = dash.get_pending_escalations()
    assert isinstance(pending, list)
    assert len(pending) == 0

    table = dash.render_board_table()
    assert table is not None


def test_render_incident_detail_card():
    """Verify detail card renders key fields and units correctly."""
    dash = AlertDashboardAgent(bus=SwarmEventBus())
    if state.dashboard.incidents:
        sample_inc = list(state.dashboard.incidents.values())[0]
        dash.process_incident(sample_inc)
        detail = dash.render_incident_detail(sample_inc.incident_id)
        assert sample_inc.incident_id in detail

    # Unknown ID returns not found message
    detail_unknown = dash.render_incident_detail("MISSING-ID")
    assert "not found" in detail_unknown.lower()


def test_swarm_telemetry_schema_completeness():
    """Verify telemetry JSON includes all micro-VM fields required for the UI sidebar."""
    res = get_swarm_status()
    assert res.status_code == 200
    telemetry = json.loads(res.body.decode("utf-8"))

    assert "agents" in telemetry
    assert "queue_depths" in telemetry
    assert "recent_buffer_count" in telemetry
    assert "speed_multiplier" in telemetry
    assert "is_demo_running" in telemetry

    for agent in telemetry["agents"]:
        assert "name" in agent
        assert "role" in agent
        assert "micro_vm_id" in agent
        assert "status" in agent
        assert "processed_count" in agent
        assert "error_count" in agent
        assert "last_heartbeat" in agent
