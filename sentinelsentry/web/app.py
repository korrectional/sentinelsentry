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
from sentinelsentry.feeds.live import PRESET_FEEDS, fetch_and_parse_live_feed
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


class IngestLiveFeedRequest(BaseModel):
    url_or_preset: str
    feed_name: Optional[str] = None
    max_items: Optional[int] = 5


class DispatchActionRequest(BaseModel):
    units: Optional[List[str]] = None
    notes: Optional[str] = None
    reason: Optional[str] = None


# ------------------------------------------------------------------
# Background Swarm Runner & Scripted Demo Sequence
# ------------------------------------------------------------------

async def run_scripted_demo_sequence():
    """
    Executes the multi-stage crisis triage demonstration sequence:
    1. Ambient routine traffic (fender bender at Crabtree Valley Mall)
    2. Severe weather report (Skywarn report on storm damage in Wake County)
    3. Major crisis begins (CAD 10-70 commercial building fire at 1200 S Saunders St) - Solo Unverified
    4. Citizen eyewitness corroboration arrives on Social Media -> Corroboration detected!
    5. Wake County Fire & Rescue 2-alarm dispatch -> Emergency Escalation FIRES!
    6. EMS Scanner dispatch standby Medic 4 -> Cross-feed corroborated across 4 channels!
    7. Routine hospital transfer at WakeMed -> Demonstrates active triage alongside ongoing crisis.
    Time delay adjusts with state.speed_multiplier.
    """
    state.is_demo_running = True
    try:
        def delay(seconds: float):
            return max(0.1, seconds / max(0.1, state.speed_multiplier))

        async def step_pipeline_async():
            """Step swarm through stages with brief micro-pauses for visible queue telemetry."""
            step_pause = max(0.04, 0.18 / max(0.1, state.speed_multiplier))
            for _ in range(4):
                state.coordinator.step()
                await asyncio.sleep(step_pause)
                if (
                    len(state.bus.ingest_items) == 0
                    and len(state.bus.verification_items) == 0
                    and len(state.bus.geocoding_items) == 0
                    and len(state.bus.dashboard_items) == 0
                ):
                    break
            state.coordinator.drain(max_rounds=2)

        now_ts = int(time.time())

        # --- Step 1: Ambient Routine Call (t=0s) ---
        item_id1 = f"demo-routine-1-{now_ts}"
        item1 = RawFeedItem(
            id=item_id1,
            feed_id="cad_police",
            feed_name="Raleigh Metro 911 / Police CAD",
            title="CAD: 10-50 PD Crabtree",
            raw_text="10-50 PD ONLY CRABTREE VALLEY MALL LOT. MINOR FENDER BENDER NO INJURIES DRIVERS EXCHANGING INFO",
        )
        state.bus.record_recent_item_sync(item1)
        state.bus.ingest_items.append(item1)
        await step_pipeline_async()

        await asyncio.sleep(delay(3.0))

        # --- Step 2: Skywarn Weather Observation (t=3s) ---
        item_id2_sky = f"demo-weather-{now_ts}"
        item2_sky = RawFeedItem(
            id=item_id2_sky,
            feed_id="weather_skywarn",
            feed_name="NWS Skywarn Weather Spotters",
            title="SKYWARN: Heavy Rain & Wind Gusts",
            raw_text="SPOTTER REPORT: Sustained 40mph wind gusts and heavy downpours along US-1 corridor near Cary. Minor roadway ponding, traffic moving normally.",
        )
        state.bus.record_recent_item_sync(item2_sky)
        state.bus.ingest_items.append(item2_sky)
        await step_pipeline_async()

        await asyncio.sleep(delay(3.5))

        # --- Step 3: Crisis Erupts - Solo High-Risk CAD Dispatch (t=7s) ---
        item_id2 = f"demo-fire-cad-{now_ts}"
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
        await step_pipeline_async()

        await asyncio.sleep(delay(4.0))

        # --- Step 4: Citizen Eyewitness Post -> Cross-Feed Corroboration! (t=11s) ---
        item_id3 = f"demo-fire-social-{now_ts}"
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
        await step_pipeline_async()

        await asyncio.sleep(delay(4.0))

        # --- Step 5: Wake Fire 2-Alarm Dispatch -> ESCALATION FIRES! (t=15s) ---
        item_id4 = f"demo-fire-wake-{now_ts}"
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
        await step_pipeline_async()

        # Update initial CAD report now that cross-feed corroboration is established
        initial_cad_inc = state.dashboard.get_incident(f"INC-{item_id2}")
        if initial_cad_inc and not initial_cad_inc.is_escalated:
            initial_cad_inc.is_escalated = True
            initial_cad_inc.human_status = "ESCALATED"
            initial_cad_inc.triage.credibility_score = min(1.0, initial_cad_inc.triage.credibility_score + 0.35)
            state.dashboard.process_incident(initial_cad_inc)

        await asyncio.sleep(delay(4.0))

        # --- Step 6: EMS Scanner Standby Deployment (t=19s) ---
        item_id5_ems = f"demo-fire-ems-{now_ts}"
        item5_ems = RawFeedItem(
            id=item_id5_ems,
            feed_id="ems_scanner",
            feed_name="Triangle Regional EMS Scanner",
            title="EMS TAC-2: STANDBY SAUNDERS",
            raw_text="EMS 11 DISPATCH: Standby 1200 S Saunders St commercial 2nd alarm. Medic 4 staging on Commerce Pkwy for civilian burn victim extraction.",
            metadata={"scenario_id": "SCN-001-WAREHOUSE-FIRE"},
        )
        state.bus.record_recent_item_sync(item5_ems)
        state.bus.ingest_items.append(item5_ems)
        await step_pipeline_async()

        await asyncio.sleep(delay(3.5))

        # --- Step 7: Routine Medical Transfer -> System Handles Routine Concurrently (t=23s) ---
        item_id6_routine = f"demo-routine-2-{now_ts}"
        item6_routine = RawFeedItem(
            id=item_id6_routine,
            feed_id="ems_scanner",
            feed_name="Triangle Regional EMS Scanner",
            title="EMS: Routine Transfer WakeMed",
            raw_text="MEDIC 8: Routine non-emergency transport from WakeMed Health & Hospitals to rehab center, no lights no sirens.",
        )
        state.bus.record_recent_item_sync(item6_routine)
        state.bus.ingest_items.append(item6_routine)
        await step_pipeline_async()

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error(f"Error during scripted demo sequence: {exc}")
    finally:
        state.is_demo_running = False


def run_scripted_demo_sync():
    """Synchronously execute the full crisis scenario without asyncio sleep delays (for testing and discrete runs)."""
    state.is_demo_running = True
    try:
        now_ts = int(time.time())

        # Step 1: Routine Fender Bender
        item_id1 = f"demo-routine-1-{now_ts}"
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

        # Step 2: Skywarn Weather Observation
        item_id2_sky = f"demo-weather-{now_ts}"
        item2_sky = RawFeedItem(
            id=item_id2_sky,
            feed_id="weather_skywarn",
            feed_name="NWS Skywarn Weather Spotters",
            title="SKYWARN: Heavy Rain & Wind Gusts",
            raw_text="SPOTTER REPORT: Sustained 40mph wind gusts and heavy downpours along US-1 corridor near Cary. Minor roadway ponding, traffic moving normally.",
        )
        state.bus.record_recent_item_sync(item2_sky)
        state.bus.ingest_items.append(item2_sky)
        state.coordinator.drain(max_rounds=3)

        # Step 3: Messy High-Risk 2-Alarm Fire
        item_id2 = f"demo-fire-cad-{now_ts}"
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

        # Step 4: Corroborating Citizen Eyewitness Post
        item_id3 = f"demo-fire-social-{now_ts}"
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

        # Step 5: Wake Fire 2-alarm dispatch -> Escalation fires!
        item_id4 = f"demo-fire-wake-{now_ts}"
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

        # Step 6: EMS Scanner Standby Deployment
        item_id5_ems = f"demo-fire-ems-{now_ts}"
        item5_ems = RawFeedItem(
            id=item_id5_ems,
            feed_id="ems_scanner",
            feed_name="Triangle Regional EMS Scanner",
            title="EMS TAC-2: STANDBY SAUNDERS",
            raw_text="EMS 11 DISPATCH: Standby 1200 S Saunders St commercial 2nd alarm. Medic 4 staging on Commerce Pkwy for civilian burn victim extraction.",
            metadata={"scenario_id": "SCN-001-WAREHOUSE-FIRE"},
        )
        state.bus.record_recent_item_sync(item5_ems)
        state.bus.ingest_items.append(item5_ems)

        # Drain swarm to process all items, corroborations, and escalations
        state.coordinator.drain(max_rounds=5)

        # Ensure initial CAD fire report is also updated to escalated state
        initial_cad_inc = state.dashboard.get_incident(f"INC-{item_id2}")
        if initial_cad_inc and not initial_cad_inc.is_escalated:
            initial_cad_inc.is_escalated = True
            initial_cad_inc.human_status = "ESCALATED"
            initial_cad_inc.triage.credibility_score = min(1.0, initial_cad_inc.triage.credibility_score + 0.35)
            state.dashboard.process_incident(initial_cad_inc)
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
        tile_url = f"{tile_url}{sep}key={settings.CARTO_API_KEY}&api_key={settings.CARTO_API_KEY}"

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


@app.get("/api/feeds/presets")
def get_feed_presets():
    """Return available public emergency RSS/Atom feed presets."""
    return JSONResponse(content={"presets": list(PRESET_FEEDS.values())})


@app.post("/api/feeds/ingest-live")
def ingest_live_feed(req: IngestLiveFeedRequest):
    """
    Ingest real-time public emergency RSS or Atom stream (e.g. USGS Earthquakes, NWS Alerts, FEMA).
    Extracts GeoRSS coordinates, triages with Gemma 4, corroborates cross-feed,
    and updates the Leaflet map and live operations stream.
    """
    if not req.url_or_preset or not req.url_or_preset.strip():
        raise HTTPException(status_code=400, detail="Feed URL or preset key is required.")

    items, is_live_network, source_desc = fetch_and_parse_live_feed(
        url_or_preset=req.url_or_preset.strip(),
        custom_feed_name=req.feed_name,
        max_items=max(1, min(20, req.max_items or 5)),
    )

    if not items:
        return JSONResponse(
            status_code=200,
            content={
                "status": "warning",
                "count": 0,
                "message": "No new entries found in feed.",
                "source_desc": source_desc,
            },
        )

    for item in items:
        state.bus.record_recent_item_sync(item)
        state.bus.ingest_items.append(item)

    # Drain swarm to immediately triage, verify, geocode, and push to dashboard
    state.coordinator.drain(max_rounds=6)

    # Return summary of ingested items
    item_summaries = []
    for item in items:
        inc = state.dashboard.get_incident(f"INC-{item.id}")
        item_summaries.append({
            "item_id": item.id,
            "title": item.title,
            "feed_id": item.feed_id,
            "feed_name": item.feed_name,
            "latitude": item.metadata.get("latitude"),
            "longitude": item.metadata.get("longitude"),
            "display_name": item.metadata.get("display_name"),
            "incident_type": inc.triage.incident_type if inc else "Processing",
            "casualty_risk_level": inc.triage.casualty_risk_level if inc else 1,
            "is_escalated": inc.is_escalated if inc else False,
        })

    return JSONResponse(content={
        "status": "success",
        "count": len(items),
        "is_live_network": is_live_network,
        "source_desc": source_desc,
        "feed_name": req.feed_name or (items[0].feed_name if items else "Live RSS Feed"),
        "items": item_summaries,
    })


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
    """Trigger the scripted crisis triage demonstration scenario."""
    if state.is_demo_running:
        if state.demo_task and state.demo_task.done():
            state.is_demo_running = False
        else:
            return JSONResponse(content={"status": "already_running", "message": "Demo scenario is already in progress."})
    state.demo_task = asyncio.create_task(run_scripted_demo_sequence())
    return JSONResponse(content={"status": "started", "message": "Scripted crisis scenario launched."})


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
    state.is_demo_running = False
    state.bus.ingest_items.clear()
    state.bus.verification_items.clear()
    state.bus.geocoding_items.clear()
    state.bus.dashboard_items.clear()
    state.bus.recent_items_buffer.clear()
    state.dashboard.incidents.clear()
    state.dashboard.escalated_ids.clear()
    state.dashboard._corroborated_ids.clear()
    state.dashboard.stats = {"total_processed": 0, "escalated_count": 0, "corroborated_count": 0, "routine_count": 0}
    # Clean feed generator item buffers and agent dedup caches
    for gen in state.feed_generators.values():
        gen._items.clear()
    for ing in state.coordinator.ingest_agents:
        if hasattr(ing, "client") and hasattr(ing.client, "dedup_cache"):
            ing.client.dedup_cache.clear()
            ing.total_ingested = 0
            ing.processed_count = 0
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
