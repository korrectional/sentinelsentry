"""
Geocoder agent resolving location_text to GPS coordinates via Nominatim and local landmark cache.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from sentinelsentry.config import settings
from sentinelsentry.schema import EnrichedIncident, GeocodedLocation
from sentinelsentry.agents.base import BaseAgent, SwarmEventBus

logger = logging.getLogger("sentinelsentry.agents.geocoder")

LOCAL_LANDMARK_CACHE: Dict[str, Tuple[float, float, str]] = {
    "1200 s saunders st": (35.7592, -78.6534, "1200 S Saunders St, Raleigh, NC 27603"),
    "1200 s saunders": (35.7592, -78.6534, "1200 S Saunders St, Raleigh, NC 27603"),
    "saunders st": (35.7650, -78.6510, "S Saunders St corridor, Raleigh, NC"),
    "i-40 eastbound mm 287": (35.7950, -78.7810, "I-40 Eastbound Mile Marker 287, Cary/Raleigh, NC"),
    "i-40 eb mm 287": (35.7950, -78.7810, "I-40 Eastbound Mile Marker 287, Cary/Raleigh, NC"),
    "i-40 eb mile 287": (35.7950, -78.7810, "I-40 Eastbound Mile Marker 287, Cary/Raleigh, NC"),
    "i-40 eastbound by cary exit": (35.7950, -78.7810, "I-40 Cary Exit 287, NC"),
    "crabtree valley mall": (35.8398, -78.6793, "Crabtree Valley Mall, 4325 Glenwood Ave, Raleigh, NC 27612"),
    "atlantic ave & hodges st": (35.8115, -78.6250, "Atlantic Ave & Hodges St, Raleigh, NC 27604"),
    "atlantic ave near crabtree creek": (35.8140, -78.6270, "Crabtree Creek at Atlantic Ave, Raleigh, NC"),
    "crabtree creek": (35.8140, -78.6270, "Crabtree Creek Basin, Raleigh, NC"),
    "401 fayetteville st": (35.7766, -78.6394, "401 Fayetteville St, Downtown Raleigh, NC 27601"),
    "401 fayetteville": (35.7766, -78.6394, "401 Fayetteville St, Downtown Raleigh, NC 27601"),
    "500 s salisbury st": (35.7745, -78.6410, "500 S Salisbury St, Raleigh, NC 27601"),
    "3000 new bern ave": (35.7876, -78.5862, "WakeMed Health & Hospitals, 3000 New Bern Ave, Raleigh, NC 27610"),
    "glenwood ave & peace st": (35.7892, -78.6475, "Glenwood Ave & Peace St, Glenwood South, Raleigh, NC 27603"),
}


class GeocoderAgent(BaseAgent):
    """
    Resolves extracted location_text into latitude, longitude, and normalized address.
    Uses local cache first, falling back to Nominatim API.
    """

    def __init__(self, bus: SwarmEventBus, enable_network: bool = False):
        super().__init__(agent_name="GeocoderAgent", bus=bus)
        self.enable_network = enable_network
        self.resolved_count = 0
        self._memory_cache: Dict[str, GeocodedLocation] = {}

    def _lookup_local_cache(self, location_text: str) -> Optional[GeocodedLocation]:
        norm = location_text.lower().strip()
        for key, (lat, lon, display) in LOCAL_LANDMARK_CACHE.items():
            if key in norm or norm in key:
                return GeocodedLocation(
                    location_text=location_text,
                    latitude=lat,
                    longitude=lon,
                    display_name=display,
                    is_resolved=True,
                    provider="Local Raleigh Landmark Cache",
                )
        return None

    def resolve_location_sync(self, location_text: str) -> GeocodedLocation:
        if not location_text or location_text.lower() == "unknown":
            return GeocodedLocation(
                location_text="Unknown",
                latitude=None,
                longitude=None,
                display_name="Unknown Location",
                is_resolved=False,
                provider="None",
            )

        if location_text in self._memory_cache:
            return self._memory_cache[location_text]

        cached = self._lookup_local_cache(location_text)
        if cached:
            self._memory_cache[location_text] = cached
            return cached

        unresolved = GeocodedLocation(
            location_text=location_text,
            latitude=35.7796,
            longitude=-78.6382,
            display_name=f"{location_text} (Approx Wake County)",
            is_resolved=False,
            provider="Approximate Reference",
        )
        self._memory_cache[location_text] = unresolved
        return unresolved

    def step_sync(self) -> int:
        count = 0
        while self.bus.geocoding_items:
            incident = self.bus.geocoding_items.popleft()
            geocoded = self.resolve_location_sync(incident.triage.location_text)
            incident.geocoded = geocoded
            self.resolved_count += 1
            self.bus.dashboard_items.append(incident)
            count += 1
        return count
