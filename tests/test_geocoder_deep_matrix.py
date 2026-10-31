"""
Deep test matrix for the Geocoder Agent:
- Regional landmark cache resolution
- Highway mile marker and directional handling
- Zero-latency in-memory cache hits
- Unknown and malformed address safety
"""

from sentinelsentry.schema import RawFeedItem, EnrichedIncident, TriageOutput
from sentinelsentry.agents.base import SwarmEventBus
from sentinelsentry.agents.geocoder import GeocoderAgent, LOCAL_LANDMARK_CACHE


def test_geocoder_known_landmarks():
    """Verify all predefined regional landmarks resolve to exact GPS coordinates."""
    bus = SwarmEventBus()
    agent = GeocoderAgent(bus=bus, enable_network=False)

    landmarks = [
        ("1200 S Saunders St, Raleigh", 35.7592, -78.6534),
        ("I-40 Eastbound MM 287 nr Cary", 35.7950, -78.7810),
        ("Crabtree Valley Mall, 4325 Glenwood Ave", 35.8398, -78.6793),
        ("401 Fayetteville St, Downtown Raleigh", 35.7766, -78.6394),
        ("3000 New Bern Ave nr WakeMed", 35.7876, -78.5862),
    ]

    for loc_text, expected_lat, expected_lon in landmarks:
        geo = agent.resolve_location_sync(loc_text)
        assert geo.is_resolved is True
        assert abs(geo.latitude - expected_lat) < 0.001
        assert abs(geo.longitude - expected_lon) < 0.001
        assert geo.display_name != ""


def test_geocoder_case_and_punctuation_insensitivity():
    """Verify address variations with mixed case and punctuation resolve properly."""
    bus = SwarmEventBus()
    agent = GeocoderAgent(bus=bus, enable_network=False)

    queries = [
        "1200 s. saunders st., raleigh, nc",
        "1200 S SAUNDERS ST",
        "i-40 eb mm 287",
        "CRABTREE VALLEY MALL",
    ]

    for q in queries:
        geo = agent.resolve_location_sync(q)
        assert geo.is_resolved is True
        assert geo.latitude is not None
        assert geo.longitude is not None


def test_geocoder_unknown_location_fallback():
    """Verify unknown location text safely falls back without raising exceptions."""
    bus = SwarmEventBus()
    agent = GeocoderAgent(bus=bus, enable_network=False)

    geo = agent.resolve_location_sync("Unknown Nonexistent Road Nowhere")
    assert geo.latitude is not None
    assert geo.longitude is not None
    # Safe regional fallback
    assert 35.0 <= geo.latitude <= 36.5
    assert -79.5 <= geo.longitude <= -78.0


def test_geocoder_memory_cache_hit():
    """Verify second lookup of identical location hits memory cache with zero overhead."""
    bus = SwarmEventBus()
    agent = GeocoderAgent(bus=bus, enable_network=False)

    res1 = agent.resolve_location_sync("Atlantic Ave & Hodges St")
    assert "Atlantic Ave & Hodges St" in agent._memory_cache

    # Second query returns cached instance
    res2 = agent.resolve_location_sync("Atlantic Ave & Hodges St")
    assert res1.latitude == res2.latitude
    assert res1.longitude == res2.longitude
