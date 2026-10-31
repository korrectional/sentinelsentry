"""
Unit tests for SentinelSentry web operations hub, Maritime micro-VM telemetry,
and Gemma 4 multimodal crisis triage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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
    SubmitIncidentRequest,
    SpeedControlRequest,
    DispatchActionRequest,
)


def test_gemma4_multimodal_triage():
    """Verify Gemma 4 triage handles text + image input and text-only input backwards compatibility."""
    engine = LocalTriageEngine()

    # 1. Text-only incident (Standard CAD dispatch)
    text_only = engine.predict("10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF")
    assert text_only.incident_type == "Structure Fire"
    assert text_only.casualty_risk_level >= 4
    assert 0.0 <= text_only.credibility_score <= 1.0
    assert "Saunders" in text_only.location_text

    # 2. Multimodal incident with fire visual evidence
    multimodal_fire = engine.predict(
        raw_text="Explosion and fire at warehouse near 1200 S Saunders St downtown",
        image_data="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD_flames_smoke_damage",
        feed_source="citizen_photo_stream",
    )
    assert multimodal_fire.incident_type == "Structure Fire"
    assert multimodal_fire.casualty_risk_level >= 4
    # Image evidence boosts credibility score
    assert multimodal_fire.credibility_score >= 0.70

    # 3. Multimodal flooding/submerged vehicle
    flood_item = engine.predict(
        raw_text="Water rising rapidly Crabtree Valley Mall lower level road impassable",
        image_data="data:image/png;base64,submerged_car_floodwater_rescue",
        feed_source="skywarn_weather",
    )
    assert flood_item.incident_type == "Severe Weather / Flooding"
    assert flood_item.casualty_risk_level >= 3


def test_web_dashboard_html_rendering():
    """Verify dashboard HTML template renders with required judged-demo components."""
    response = serve_dashboard()
    assert response.status_code == 200
    html = response.body.decode("utf-8")
    assert "SentinelSentry" in html
    assert "Gemma 4" in html
    assert "Decision Support" in html
    assert "leaflet" in html.lower()
    assert "maritime" in html.lower()
    assert "Escalation" in html


def test_swarm_telemetry_maritime_vms():
    """Verify swarm telemetry reports separate Maritime micro-VM tags and queue depths."""
    response = get_swarm_status()
    assert response.status_code == 200
    data = json.loads(response.body.decode("utf-8"))

    assert "agents" in data
    assert "queue_depths" in data
    assert len(data["agents"]) >= 9

    # Verify Maritime micro-VM isolation identifiers
    vm_ids = [a["micro_vm_id"] for a in data["agents"]]
    assert any("maritime-vm-ingest" in vm for vm in vm_ids)
    assert any("maritime-vm-triage" in vm for vm in vm_ids)
    assert any("maritime-vm-verifier" in vm for vm in vm_ids)
    assert any("maritime-vm-geocoder" in vm for vm in vm_ids)
    assert any("maritime-vm-alerter" in vm or "maritime-vm-dashboard" in vm for vm in vm_ids)


def test_eval_proof_endpoint():
    """Verify /api/eval endpoint provides before/after benchmark proof for judges."""
    response = get_eval_proof()
    assert response.status_code == 200
    metrics = json.loads(response.body.decode("utf-8"))

    assert "base_gemma" in metrics
    assert "lora_adapter" in metrics
    assert metrics["lora_adapter"]["valid_json_rate"] == 100.0
    assert metrics["lora_adapter"]["incident_accuracy"] > metrics["base_gemma"]["incident_accuracy"]
    assert metrics["lora_adapter"]["risk_mae"] < metrics["base_gemma"]["risk_mae"]


def test_manual_multimodal_submission_and_lifecycle():
    """Test manual dispatcher submission with image, followed by acknowledge/dispatch actions."""
    # 1. Reset baseline state
    reset_demo()

    # 2. Submit multimodal incident
    req = SubmitIncidentRequest(
        raw_text="CRITICAL COLLAPSE: 2-alarm commercial at 1200 S Saunders St. Heavy smoke, 2 occupants trapped",
        feed_id="dispatcher_priority",
        title="Emergency 911 Call",
        image_data="data:image/jpeg;base64,building_collapse_heavy_smoke",
    )
    submit_res = submit_incident(req)
    assert submit_res.status_code == 200
    sub_data = json.loads(submit_res.body.decode("utf-8"))
    assert sub_data["status"] == "success"
    item_id = sub_data["item_id"]
    incident_id = f"INC-{item_id}"

    # 3. Check incident list
    inc_res = get_incidents()
    assert inc_res.status_code == 200
    inc_data = json.loads(inc_res.body.decode("utf-8"))
    found = [i for i in inc_data["incidents"] if i["incident_id"] == incident_id]
    assert len(found) == 1
    assert found[0]["casualty_risk_level"] >= 4
    assert found[0]["image_data"] is not None

    # 4. Dispatch Response Units
    dispatch_res = dispatch_incident(
        incident_id=incident_id,
        req=DispatchActionRequest(
            units=["Engine 1", "Ladder 3", "Battalion 2", "EMS Medic 9"],
            notes="Authorized structural entry Code 3.",
        ),
    )
    assert dispatch_res.status_code == 200
    disp_data = json.loads(dispatch_res.body.decode("utf-8"))
    assert disp_data["status"] == "dispatched"
    assert "DISPATCHED" in disp_data["human_status"]


def test_speed_control_api():
    """Verify demo feed speed adjustments (0.5x to 100x)."""
    res1 = set_demo_speed(SpeedControlRequest(multiplier=2.5))
    assert res1.status_code == 200
    data1 = json.loads(res1.body.decode("utf-8"))
    assert data1["speed_multiplier"] == 2.5

    # Reset speed back to 1.0x
    res2 = set_demo_speed(SpeedControlRequest(multiplier=1.0))
    assert res2.status_code == 200
