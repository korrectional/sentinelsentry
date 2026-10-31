"""
FastAPI application serving the SentinelSentry interactive emergency operations dashboard.
Provides real-time SSE streaming, WebSocket live updates, Leaflet mapping,
Maritime micro-VM swarm telemetry, and dispatcher decision-support controls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from sentinelsentry.config import settings, FeedConfig
from sentinelsentry.schema import RawFeedItem, EnrichedIncident
from sentinelsentry.agents.base import SwarmEventBus, SwarmCoordinator
from sentinelsentry.agents.ingest import IngestAgent
from sentinelsentry.agents.triage import TriageAgent, LocalTriageEngine
from sentinelsentry.agents.verification import VerificationAgent
from sentinelsentry.agents.geocoder import GeocoderAgent
from sentinelsentry.agents.dashboard import AlertDashboardAgent
from sentinelsentry.feeds.generator import RSSFeedGenerator
from sentinelsentry.eval.benchmark import get_cached_benchmark_metrics

logger = logging.getLogger("sentinelsentry.web")

app = FastAPI(
    title="SentinelSentry: Emergency Crisis Triage Decision Support",
    description="Gemma 4 LoRA Swarm inside Maritime Micro-VMs",
    version="0.2.0",
)

# Global Swarm State
class SwarmState:
    def __init__(self):
        self.bus = SwarmEventBus()
        self.coordinator = SwarmCoordinator(bus=self.bus)
        self.dashboard = AlertDashboardAgent(bus=self.bus)
        self.geocoder = GeocoderAgent(bus=self.bus, enable_network=False)
        self.verification = VerificationAgent(bus=self.bus)
        self.feed_generators: Dict[str, RSSFeedGenerator] = {}
        self.speed_multiplier: float = 1.0
        self.demo_task: Optional[asyncio.Task] = None
        self.is_demo_running: bool = False
        self._init_swarm()

    def _init_swarm(self):
        # 1. Setup simulated feeds
        for cfg in settings.FEEDS:
            gen = RSSFeedGenerator(
                feed_id=cfg.id,
                feed_name=cfg.name,
                description=cfg.description,
            )
            self.feed_generators[cfg.id] = gen
            ingest_worker = IngestAgent(
                feed_config=cfg,
                bus=self.bus,
                custom_fetcher=lambda g=gen: g.to_raw_feed_items(),
            )
            self.coordinator.register_ingest_agent(ingest_worker)

        # 2. Setup dual triage workers
        triage_engine = LocalTriageEngine()
        t1 = TriageAgent(bus=self.bus, worker_id=1, engine=triage_engine)
        t2 = TriageAgent(bus=self.bus, worker_id=2, engine=triage_engine)
        self.coordinator.register_triage_worker(t1)
        self.coordinator.register_triage_worker(t2)

        # 3. Setup verification, geocoder, and dashboard
        self.coordinator.set_verification_agent(self.verification)
        self.coordinator.set_geocoder_agent(self.geocoder)
        self.coordinator.set_dashboard_agent(self.dashboard)

    def populate_baseline(self):
        """Pre-populate baseline emergency and routine calls."""
        for gen in self.feed_generators.values():
            gen.populate_from_scenarios()
        self.coordinator.drain(max_rounds=6)


state = SwarmState()


# Request Models
class SubmitIncidentRequest(BaseModel):
    raw_text: str
    feed_id: Optional[str] = "web_upload"
    title: Optional[str] = "Manual Dispatch Submission"
    image_data: Optional[str] = None
    image_mime_type: Optional[str] = "image/jpeg"


class SpeedControlRequest(BaseModel):
    multiplier: float


class DispatchActionRequest(BaseModel):
    units: Optional[List[str]] = None
    notes: Optional[str] = None
    reason: Optional[str] = None


# ------------------------------------------------------------------
# Background Swarm Runner & Scripted Demo Sequence
# ------------------------------------------------------------------

async def run_scripted_demo_sequence():
    """
    Executes the 90-second scripted scenario:
    1. Routine call (fender bender at Crabtree Valley Mall)
    2. Messy high-risk call (10-70 COMM BLDG 1200 S SAUNDERS ST)
    3. Corroborating call from second feed (Citizen on Saunders St + Wake Fire 2-alarm)
    4. Escalation fires!
    Time delay adjusts with state.speed_multiplier.
    """
    state.is_demo_running = True
    try:
        def delay(seconds: float):
            return max(0.1, seconds / max(0.1, state.speed_multiplier))

        # --- Step 1: Routine Fender Bender (t=0s) ---
        item_id1 = f"demo-routine-{int(time.time())}"
        item1 = RawFeedItem(
            id=item_id1,
            feed_id="cad_police",
            feed_name="Raleigh Metro 911 / Police CAD",
            title="CAD: 10-50 PD Crabtree",
            raw_text="10-50 PD ONLY CRABTREE VALLEY MALL LOT. MINOR FENDER BENDER NO INJURIES DRIVERS EXCHANGING INFO",
        )
        state.bus.record_recent_item_sync(item1)
        state.bus.ingest_items.append(item1)
        state.coordinator.drain(max_rounds=3)

        await asyncio.sleep(delay(25.0))

        # --- Step 2: Messy High-Risk 2-Alarm Fire (t=25s) ---
        item_id2 = f"demo-fire-cad-{int(time.time())}"
        item2 = RawFeedItem(
            id=item_id2,
            feed_id="cad_police",
            feed_name="Raleigh Metro 911 / Police CAD",
            title="CAD: 10-70 Saunders",
            raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF EVAC IN PROGRESS 1 SUBJ REPORTED INSIDE",
            metadata={"scenario_id": "SCN-001-WAREHOUSE-FIRE"},
        )
        state.bus.record_recent_item_sync(item2)
        state.bus.ingest_items.append(item2)
        state.coordinator.drain(max_rounds=3)

        await asyncio.sleep(delay(30.0))

        # --- Step 3: Corroborating Citizen Eyewitness Post (t=55s) ---
        item_id3 = f"demo-fire-social-{int(time.time())}"
        item3 = RawFeedItem(
            id=item_id3,
            feed_id="social_crisis",
            feed_name="Citizen Eyewitness Crisis Stream (X/Reddit)",
            title="Citizen X: Huge Explosion Saunders St",
            raw_text="Holy smoke huge explosions coming from warehouse district on Saunders St downtown Raleigh sirens everywhere!! Thick black smoke filling sky",
            metadata={"scenario_id": "SCN-001-WAREHOUSE-FIRE"},
        )
        state.bus.record_recent_item_sync(item3)
        state.bus.ingest_items.append(item3)

        # Also inject Wake Fire 2-alarm dispatch
        item_id4 = f"demo-fire-wake-{int(time.time())}"
        item4 = RawFeedItem(
            id=item_id4,
            feed_id="fire_rescue",
            feed_name="Wake County Fire & Rescue Dispatch",
            title="WAKE FIRE: 2-ALARM COMM FIRE",
            raw_text="STRUCTURE FIRE: 2-alarm commercial at 1200 S Saunders St. Heavy smoke venting 2nd floor, 1 occupant trapped, collapse zone established",
            metadata={"scenario_id": "SCN-001-WAREHOUSE-FIRE"},
        )
        state.bus.record_recent_item_sync(item4)
        state.bus.ingest_items.append(item4)

        # --- Step 4 & 5: Swarm drain -> Verification cross-corroborates -> Escalation fires! ---
        state.coordinator.drain(max_rounds=5)

    except asyncio.CancelledError:
        pass
    finally:
        state.is_demo_running = False


def run_scripted_demo_sync():
    """Synchronously execute the full crisis scenario without asyncio sleep delays (for testing and discrete runs)."""
    state.is_demo_running = True
    try:
        # Step 1: Routine Fender Bender
        item_id1 = f"demo-routine-{int(time.time())}"
        item1 = RawFeedItem(
            id=item_id1,
            feed_id="cad_police",
            feed_name="Raleigh Metro 911 / Police CAD",
            title="CAD: 10-50 PD Crabtree",
            raw_text="10-50 PD ONLY CRABTREE VALLEY MALL LOT. MINOR FENDER BENDER NO INJURIES DRIVERS EXCHANGING INFO",
        )
        state.bus.record_recent_item_sync(item1)
        state.bus.ingest_items.append(item1)
        state.coordinator.drain(max_rounds=3)

        # Step 2: Messy High-Risk 2-Alarm Fire
        item_id2 = f"demo-fire-cad-{int(time.time())}"
        item2 = RawFeedItem(
            id=item_id2,
            feed_id="cad_police",
            feed_name="Raleigh Metro 911 / Police CAD",
            title="CAD: 10-70 Saunders",
            raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF EVAC IN PROGRESS 1 SUBJ REPORTED INSIDE",
            metadata={"scenario_id": "SCN-001-WAREHOUSE-FIRE"},
        )
        state.bus.record_recent_item_sync(item2)
        state.bus.ingest_items.append(item2)
        state.coordinator.drain(max_rounds=3)

        # Step 3: Corroborating Citizen Eyewitness Post
        item_id3 = f"demo-fire-social-{int(time.time())}"
        item3 = RawFeedItem(
            id=item_id3,
            feed_id="social_crisis",
            feed_name="Citizen Eyewitness Crisis Stream (X/Reddit)",
            title="Citizen X: Huge Explosion Saunders St",
            raw_text="Holy smoke huge explosions coming from warehouse district on Saunders St downtown Raleigh sirens everywhere!! Thick black smoke filling sky",
            metadata={"scenario_id": "SCN-001-WAREHOUSE-FIRE"},
        )
        state.bus.record_recent_item_sync(item3)
        state.bus.ingest_items.append(item3)

        # Also inject Wake Fire 2-alarm dispatch
        item_id4 = f"demo-fire-wake-{int(time.time())}"
        item4 = RawFeedItem(
            id=item_id4,
            feed_id="fire_rescue",
            feed_name="Wake County Fire & Rescue Dispatch",
            title="WAKE FIRE: 2-ALARM COMM FIRE",
            raw_text="STRUCTURE FIRE: 2-alarm commercial at 1200 S Saunders St. Heavy smoke venting 2nd floor, 1 occupant trapped, collapse zone established",
            metadata={"scenario_id": "SCN-001-WAREHOUSE-FIRE"},
        )
        state.bus.record_recent_item_sync(item4)
        state.bus.ingest_items.append(item4)

        # Step 4 & 5: Swarm drain -> Verification cross-corroborates -> Escalation fires!
        state.coordinator.drain(max_rounds=5)
    finally:
        state.is_demo_running = False


# ------------------------------------------------------------------
# REST API Endpoints
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the single-page emergency dispatcher operations center."""
    template_path = Path(__file__).resolve().parent / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard template not found.")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    tile_url = settings.MAP_TILE_URL
    if settings.CARTO_API_KEY:
        sep = "&" if "?" in tile_url else "?"
        tile_url = f"{tile_url}{sep}api_key={settings.CARTO_API_KEY}"

    html_content = html_content.replace("__MAP_TILE_URL__", tile_url)
    return HTMLResponse(content=html_content)


@app.get("/api/incidents")
def get_incidents():
    """Return all incidents sorted by escalation and risk."""
    return JSONResponse(content={
        "incidents": state.dashboard.get_incidents_payload(),
        "stats": state.dashboard.stats,
    })


@app.get("/api/swarm")
def get_swarm_status():
    """Return Maritime micro-VM agent telemetry and queue depths."""
    telemetry = state.coordinator.get_swarm_telemetry()
    telemetry["speed_multiplier"] = state.speed_multiplier
    telemetry["is_demo_running"] = state.is_demo_running
    return JSONResponse(content=telemetry)


@app.get("/api/eval")
def get_eval_proof():
    """Return before/after evaluation benchmark comparing Base Gemma 4 vs Gemma 4 LoRA Adapter."""
    metrics = get_cached_benchmark_metrics()
    return JSONResponse(content=metrics)


@app.post("/api/submit")
def submit_incident(req: SubmitIncidentRequest):
    """
    Manual dispatch submission with optional multimodal image attachment.
    Immediately triaged, verified, geocoded, and pushed to the live board.
    """
    if not req.raw_text or not req.raw_text.strip():
        raise HTTPException(status_code=400, detail="Dispatch raw text cannot be empty.")

    item_id = f"manual-{int(time.time()*1000)}"
    item = RawFeedItem(
        id=item_id,
        feed_id=req.feed_id or "web_upload",
        feed_name="Dispatcher Manual Entry",
        title=req.title or "Emergency Report",
        raw_text=req.raw_text.strip(),
        image_data=req.image_data,
        image_mime_type=req.image_mime_type or "image/jpeg",
    )
    state.bus.record_recent_item_sync(item)
    state.bus.ingest_items.append(item)
    state.coordinator.drain(max_rounds=4)

    return JSONResponse(content={
        "status": "success",
        "item_id": item_id,
        "message": "Incident submitted to multimodal swarm.",
        "incident": state.dashboard.get_incident(f"INC-{item_id}").summary_line if state.dashboard.get_incident(f"INC-{item_id}") else "Processing",
    })


@app.post("/api/incidents/{incident_id}/acknowledge")
def acknowledge_incident(incident_id: str):
    """Dispatcher action: Acknowledge an active emergency escalation."""
    try:
        inc = state.dashboard.acknowledge_incident(incident_id)
        return JSONResponse(content={"status": "acknowledged", "incident_id": incident_id, "human_status": inc.human_status})
    except KeyError:
        raise HTTPException(status_code=404, detail="Incident not found.")


@app.post("/api/incidents/{incident_id}/dismiss")
def dismiss_incident(incident_id: str, req: Optional[DispatchActionRequest] = None):
    """Dispatcher action: Dismiss an incident (e.g. false alarm)."""
    try:
        reason = req.reason if req and req.reason else "Dispatcher Dismissal"
        inc = state.dashboard.dismiss_incident(incident_id, reason=reason)
        return JSONResponse(content={"status": "dismissed", "incident_id": incident_id, "human_status": inc.human_status})
    except KeyError:
        raise HTTPException(status_code=404, detail="Incident not found.")


@app.post("/api/incidents/{incident_id}/dispatch")
def dispatch_incident(incident_id: str, req: Optional[DispatchActionRequest] = None):
    """Dispatcher action: Assign emergency response units."""
    try:
        units = req.units if req and req.units else ["Engine 4", "Medic 2", "Ladder 1"]
        notes = req.notes if req else None
        inc = state.dashboard.dispatch_incident(incident_id, units=units, dispatcher_notes=notes)
        return JSONResponse(content={"status": "dispatched", "incident_id": incident_id, "human_status": inc.human_status})
    except KeyError:
        raise HTTPException(status_code=404, detail="Incident not found.")


@app.post("/api/demo/run")
async def trigger_demo():
    """Trigger the 90-second scripted crisis burst scenario."""
    if state.is_demo_running:
        return JSONResponse(content={"status": "already_running", "message": "Demo scenario is already in progress."})
    state.demo_task = asyncio.create_task(run_scripted_demo_sequence())
    return JSONResponse(content={"status": "started", "message": "90-second scripted crisis scenario launched."})


@app.post("/api/demo/speed")
def set_demo_speed(req: SpeedControlRequest):
    """Update speed multiplier (0.5x, 1x, 2x, 5x, 100x for Instant)."""
    import math
    val = float(req.multiplier)
    if math.isnan(val) or math.isinf(val) or val <= 0:
        val = 1.0
    state.speed_multiplier = min(100.0, max(0.1, val))
    return JSONResponse(content={"status": "speed_updated", "speed_multiplier": state.speed_multiplier})


@app.post("/api/demo/reset")
def reset_demo():
    """Reset the board and reload baseline feeds."""
    if state.demo_task and not state.demo_task.done():
        state.demo_task.cancel()
    state.bus.ingest_items.clear()
    state.bus.verification_items.clear()
    state.bus.geocoding_items.clear()
    state.bus.dashboard_items.clear()
    state.bus.recent_items_buffer.clear()
    state.dashboard.incidents.clear()
    state.dashboard.escalated_ids.clear()
    state.dashboard._corroborated_ids.clear()
    state.dashboard.stats = {"total_processed": 0, "escalated_count": 0, "corroborated_count": 0, "routine_count": 0}
    state.populate_baseline()
    return JSONResponse(content={"status": "reset_complete", "stats": state.dashboard.stats})


# ------------------------------------------------------------------
# Real-Time SSE Stream Endpoint
# ------------------------------------------------------------------

@app.get("/api/events")
async def sse_events_stream(request: Request):
    """Server-Sent Events endpoint broadcasting live updates every 500ms."""
    async def event_generator():
        last_stat_hash = None
        while True:
            if await request.is_disconnected():
                break

            # Step swarm in background if items available
            if len(state.bus.ingest_items) > 0 or len(state.bus.verification_items) > 0 or len(state.bus.geocoding_items) > 0 or len(state.bus.dashboard_items) > 0:
                state.coordinator.step()

            payload = {
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "incidents": state.dashboard.get_incidents_payload()[:25],
                "stats": state.dashboard.stats,
                "telemetry": state.coordinator.get_swarm_telemetry(),
                "active_escalations": [
                    {
                        "incident_id": inc.incident_id,
                        "type": inc.triage.incident_type,
                        "risk": inc.triage.casualty_risk_level,
                        "cred": inc.triage.credibility_score,
                        "location": inc.triage.location_text,
                        "display_name": inc.geocoded.display_name if inc.geocoded else inc.triage.location_text,
                        "text": inc.feed_item.raw_text,
                        "corrobs": inc.verification.corroborating_feed_ids if inc.verification else [],
                        "image_data": inc.image_data,
                    }
                    for inc in state.dashboard.get_pending_escalations()
                ],
                "is_demo_running": state.is_demo_running,
                "speed_multiplier": state.speed_multiplier,
            }

            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ------------------------------------------------------------------
# WebSocket Live Stream Endpoint
# ------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if len(state.bus.ingest_items) > 0 or len(state.bus.verification_items) > 0 or len(state.bus.geocoding_items) > 0 or len(state.bus.dashboard_items) > 0:
                state.coordinator.step()

            payload = {
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "incidents": state.dashboard.get_incidents_payload()[:25],
                "stats": state.dashboard.stats,
                "telemetry": state.coordinator.get_swarm_telemetry(),
                "active_escalations": [
                    {
                        "incident_id": inc.incident_id,
                        "type": inc.triage.incident_type,
                        "risk": inc.triage.casualty_risk_level,
                        "cred": inc.triage.credibility_score,
                        "location": inc.triage.location_text,
                        "display_name": inc.geocoded.display_name if inc.geocoded else inc.triage.location_text,
                        "text": inc.feed_item.raw_text,
                        "corrobs": inc.verification.corroborating_feed_ids if inc.verification else [],
                        "image_data": inc.image_data,
                    }
                    for inc in state.dashboard.get_pending_escalations()
                ],
                "is_demo_running": state.is_demo_running,
                "speed_multiplier": state.speed_multiplier,
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing SentinelSentry Swarm baseline scenarios...")
    state.populate_baseline()
