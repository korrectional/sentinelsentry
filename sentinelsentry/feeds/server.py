"""
FastAPI HTTP server serving dynamic simulated RSS feeds.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from sentinelsentry.config import settings, FeedConfig
from sentinelsentry.feeds.generator import RSSFeedGenerator
from sentinelsentry.feeds.mock_streams import CRISIS_DISPATCH_SCENARIOS


class InjectItemRequest(BaseModel):
    feed_id: str
    raw_text: str
    title: Optional[str] = None
    scenario_id: Optional[str] = "CUSTOM_INJECT"


class TriggerScenarioRequest(BaseModel):
    scenario_id: str


class FeedServerManager:
    """
    Manages in-memory RSS feed generators for all configured feeds.
    """

    def __init__(self, feed_configs: Optional[List[FeedConfig]] = None):
        configs = feed_configs or settings.FEEDS
        self.generators: Dict[str, RSSFeedGenerator] = {}
        for cfg in configs:
            gen = RSSFeedGenerator(
                feed_id=cfg.id,
                feed_name=cfg.name,
                description=cfg.description,
                feed_link=f"http://{settings.SIMULATED_FEED_SERVER_HOST}:{settings.SIMULATED_FEED_SERVER_PORT}/feeds",
            )
            # Pre-populate with baseline items
            gen.populate_from_scenarios()
            self.generators[cfg.id] = gen

    def get_feed_xml(self, feed_id: str) -> str:
        if feed_id not in self.generators:
            raise KeyError(f"Feed '{feed_id}' not found.")
        return self.generators[feed_id].to_rss_xml()

    def inject_item(self, feed_id: str, raw_text: str, title: Optional[str] = None, scenario_id: Optional[str] = None) -> str:
        if feed_id not in self.generators:
            raise KeyError(f"Feed '{feed_id}' not found.")
        return self.generators[feed_id].add_item(
            raw_text=raw_text,
            title=title,
            published_at=datetime.now(timezone.utc),
            scenario_id=scenario_id,
        )

    def trigger_scenario(self, scenario_id: str) -> List[str]:
        """Trigger all items in a named scenario across their respective feeds."""
        target_scn = next((s for s in CRISIS_DISPATCH_SCENARIOS if s["scenario_id"] == scenario_id), None)
        if not target_scn:
            raise ValueError(f"Scenario '{scenario_id}' not found.")

        created_guids = []
        for it in target_scn["items"]:
            f_id = it["feed_id"]
            if f_id in self.generators:
                guid = self.generators[f_id].add_item(
                    raw_text=it["raw_text"],
                    title=it.get("title"),
                    published_at=datetime.now(timezone.utc),
                    scenario_id=scenario_id,
                )
                created_guids.append(guid)
        return created_guids


feed_manager = FeedServerManager()
app = FastAPI(title="SentinelSentry Simulated Crisis Feed Server", version="0.1.0")


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/feeds/list")
async def list_feeds():
    return [
        {
            "id": f_id,
            "name": gen.feed_name,
            "description": gen.description,
            "items_count": len(gen.get_items()),
            "url": f"/feeds/{f_id}.xml",
        }
        for f_id, gen in feed_manager.generators.items()
    ]


@app.get("/feeds/{feed_id}.xml")
async def get_rss_feed(feed_id: str):
    # Strip .xml if included in route match
    clean_id = feed_id.replace(".xml", "")
    try:
        xml_content = feed_manager.get_feed_xml(clean_id)
        return Response(content=xml_content, media_type="application/rss+xml; charset=utf-8")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Feed '{clean_id}' does not exist.")


@app.post("/feeds/inject")
async def inject_feed_item(req: InjectItemRequest):
    try:
        guid = feed_manager.inject_item(
            feed_id=req.feed_id,
            raw_text=req.raw_text,
            title=req.title,
            scenario_id=req.scenario_id,
        )
        return {"status": "success", "guid": guid, "feed_id": req.feed_id}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/feeds/scenarios/trigger")
async def trigger_scenario_endpoint(req: TriggerScenarioRequest):
    try:
        guids = feed_manager.trigger_scenario(req.scenario_id)
        return {
            "status": "success",
            "scenario_id": req.scenario_id,
            "items_injected": len(guids),
            "guids": guids,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
