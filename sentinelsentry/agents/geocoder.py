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
    "south saunders st & peterson st": (35.7610, -78.6530, "S Saunders St & Peterson St, Raleigh, NC 27603"),
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
    "six forks rd & millbrook rd": (35.8450, -78.6360, "Six Forks Rd & Millbrook Rd, Raleigh, NC 27609"),
    "capital blvd & buffalo rd": (35.8320, -78.5830, "Capital Blvd & Buffalo Rd, Raleigh, NC 27616"),
    "wade ave near faircloth st": (35.7980, -78.6810, "Wade Ave & Faircloth St, Raleigh, NC 27607"),
    "hillsborough st & dan allen dr": (35.7870, -78.6700, "Hillsborough St & Dan Allen Dr (NCSU), Raleigh, NC 27607"),
    "wake forest rd & st albans dr": (35.8260, -78.6230, "Wake Forest Rd & St Albans Dr, Raleigh, NC 27609"),
    "falls of neuse rd & sandy forks rd": (35.8620, -78.6180, "Falls of Neuse Rd & Sandy Forks Rd, Raleigh, NC 27615"),
    "lake boone trail nr i-440": (35.8160, -78.6870, "Lake Boone Trail & I-440, Raleigh, NC 27607"),
    "western blvd & avent ferry rd": (35.7820, -78.6750, "Western Blvd & Avent Ferry Rd, Raleigh, NC 27606"),
    "tryon rd & s wilmington st": (35.7350, -78.6500, "Tryon Rd & S Wilmington St, Raleigh, NC 27603"),
    "new hope rd & poole rd": (35.7680, -78.5520, "New Hope Rd & Poole Rd, Raleigh, NC 27610"),
    "durant rd & capital blvd": (35.8780, -78.5680, "Durant Rd & Capital Blvd, Raleigh, NC 27616"),
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
        generic_tokens = {"st", "ave", "rd", "blvd", "dr", "lane", "way", "street", "road", "avenue", "near", "at", "and"}
        tokens = set(norm.replace(",", " ").replace(".", " ").replace("&", " ").split())
        significant_tokens = tokens - generic_tokens

        for key, (lat, lon, display) in LOCAL_LANDMARK_CACHE.items():
            # Match 1: Full landmark key appears inside the location text or vice versa
            if key in norm or (len(norm) >= 6 and norm in key):
                return GeocodedLocation(
                    location_text=location_text,
                    latitude=lat,
                    longitude=lon,
                    display_name=display,
                    is_resolved=True,
                    provider="Local Raleigh Landmark Cache",
                )
            # Match 2: Query contains multiple significant non-generic tokens and ALL appear in key
            # Require at least 2 significant tokens to prevent false positives from single-token matches
            if len(significant_tokens) >= 2 and len(norm) >= 5:
                if all(tok in key for tok in significant_tokens):
                    return GeocodedLocation(
                        location_text=location_text,
                        latitude=lat,
                        longitude=lon,
                        display_name=display,
                        is_resolved=True,
                        provider="Local Raleigh Landmark Cache",
                    )
        return None

    def _query_nominatim(self, location_text: str) -> Optional[GeocodedLocation]:
        if not self.enable_network:
            return None
        try:
            import json
            import urllib.parse
            import urllib.request

            query = f"{location_text}, Raleigh, NC"
            encoded = urllib.parse.quote(query)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": settings.NOMINATIM_USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and len(data) > 0:
                    first = data[0]
                    return GeocodedLocation(
                        location_text=location_text,
                        latitude=float(first["lat"]),
                        longitude=float(first["lon"]),
                        display_name=first.get("display_name", location_text),
                        is_resolved=True,
                        provider="OpenStreetMap Nominatim",
                    )
        except Exception as e:
            logger.debug(f"Nominatim lookup failed for '{location_text}': {e}")
        return None

    def resolve_location_sync(self, location_text: str) -> GeocodedLocation:
        """Resolve location_text via local landmark cache, then Nominatim, then fallback."""
        if not location_text or location_text.strip().lower() == "unknown":
            return GeocodedLocation(
                location_text=location_text or "Unknown",
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

        if self.enable_network:
            nominatim_loc = self._query_nominatim(location_text)
            if nominatim_loc:
                self._memory_cache[location_text] = nominatim_loc
                return nominatim_loc

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

    def step_sync(self, max_items: Optional[int] = None) -> int:
        count = 0
        limit = max_items if max_items is not None else float("inf")
        while self.bus.geocoding_items and count < limit:
            incident = self.bus.geocoding_items.popleft()
            geocoded = self.resolve_location_sync(incident.triage.location_text)
            incident.geocoded = geocoded
            self.resolved_count += 1
            self.bus.dashboard_items.append(incident)
            count += 1
        return count
