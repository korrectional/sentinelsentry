"""
Tests for live emergency RSS / Atom feed ingestion service and API endpoints.
Validates GeoRSS coordinate extraction, preset catalog, snapshot fallbacks,
Gemma 4 swarm triage integration, and Leaflet dashboard mapping.
"""

from __future__ import annotations

import json
import pytest
from fastapi import HTTPException

from sentinelsentry.feeds.live import (
    PRESET_FEEDS,
    strip_html_tags,
    extract_georss_coordinates,
    extract_location_name,
    fetch_and_parse_live_feed,
)
from sentinelsentry.web.app import (
    app,
    state,
    reset_demo,
    get_feed_presets,
    ingest_live_feed,
    get_incidents,
    IngestLiveFeedRequest,
)
from sentinelsentry.schema import RawFeedItem, EnrichedIncident
from sentinelsentry.agents.geocoder import GeocoderAgent
from sentinelsentry.agents.base import SwarmEventBus


def test_preset_feeds_catalog():
    """Verify emergency preset catalog configuration."""
    assert "usgs_earthquakes" in PRESET_FEEDS
    assert "nws_weather_alerts" in PRESET_FEEDS
    assert "fema_emergency" in PRESET_FEEDS

    usgs = PRESET_FEEDS["usgs_earthquakes"]
    assert "earthquake.usgs.gov" in usgs["url"]
    assert usgs["name"]
    assert usgs["badge"]


def test_strip_html_tags():
    """Verify HTML markup stripping from messy RSS summaries."""
    raw = "<p>Emergency alert: <b>High water</b> at creek.<br/>Stay indoors &amp; safe!</p>"
    clean = strip_html_tags(raw)
    assert clean == "Emergency alert: High water at creek. Stay indoors safe!"
    assert strip_html_tags("") == ""


def test_georss_coordinate_extraction():
    """Verify coordinate extraction from parsed GeoRSS elements."""
    # Test GeoJSON Point representation (lon, lat)
    mock_entry_point = {
        "where": {"type": "Point", "coordinates": (-117.525, 35.662)}
    }
    lat, lon = extract_georss_coordinates(mock_entry_point)
    assert lat == pytest.approx(35.662, abs=0.001)
    assert lon == pytest.approx(-117.525, abs=0.001)

    # Test direct geo_lat / geo_long fields
    mock_entry_direct = {
        "geo_lat": "35.839",
        "geo_long": "-78.680",
    }
    lat2, lon2 = extract_georss_coordinates(mock_entry_direct)
    assert lat2 == pytest.approx(35.839, abs=0.001)
    assert lon2 == pytest.approx(-78.680, abs=0.001)

    # Test raw text regex fallback
    mock_raw = "<georss:point>19.3820 -155.2650</georss:point>"
    lat3, lon3 = extract_georss_coordinates({}, raw_entry_text=mock_raw)
    assert lat3 == pytest.approx(19.382, abs=0.001)
    assert lon3 == pytest.approx(-155.265, abs=0.001)


def test_extract_location_name():
    """Verify human-readable location extraction from USGS and NWS formats."""
    usgs_title = "M 4.8 - 14 km ENE of Ridgecrest, California"
    loc1 = extract_location_name(usgs_title, "Moderate earthquake")
    assert "Ridgecrest, California" in loc1

    nws_title = "Flash Flood Warning issued"
    nws_summary = "Flash Flood Warning for Wake County, Crabtree Creek Basin"
    loc2 = extract_location_name(nws_title, nws_summary)
    assert "Wake County" in loc2


def test_fetch_and_parse_usgs_preset():
    """Verify parsing of USGS real-time earthquake preset with GeoRSS coordinates."""
    items, is_live, source_desc = fetch_and_parse_live_feed("usgs_earthquakes", max_items=3)
    assert len(items) == 3
    for it in items:
        assert isinstance(it, RawFeedItem)
        assert it.id.startswith("rss-")
        assert it.title
        assert it.metadata.get("is_live_stream") is True
        assert "latitude" in it.metadata
        assert "longitude" in it.metadata
        assert it.metadata["latitude"] is not None
        assert it.metadata["longitude"] is not None


def test_fetch_and_parse_nws_preset():
    """Verify parsing of NOAA/NWS active weather alerts preset."""
    items, is_live, source_desc = fetch_and_parse_live_feed("nws_weather_alerts", max_items=2)
    assert len(items) == 2
    for it in items:
        assert isinstance(it, RawFeedItem)
        assert "Warning" in it.title or "Watch" in it.title or len(it.title) > 5
        assert it.metadata.get("latitude") is not None


def test_fetch_and_parse_fema_preset():
    """Verify parsing of FEMA emergency announcements preset."""
    items, is_live, source_desc = fetch_and_parse_live_feed("fema_emergency", max_items=2)
    assert len(items) == 2
    for it in items:
        assert isinstance(it, RawFeedItem)
        assert "FEMA" in it.raw_text or "Emergency" in it.raw_text or len(it.title) > 5


def test_api_feed_presets_endpoint():
    """Test GET /api/feeds/presets returns catalog."""
    resp = get_feed_presets()
    assert resp.status_code == 200
    data = json.loads(resp.body.decode("utf-8"))
    assert "presets" in data
    preset_ids = [p["id"] for p in data["presets"]]
    assert "usgs_earthquakes" in preset_ids
    assert "nws_weather_alerts" in preset_ids


def test_api_post_ingest_live_usgs():
    """Test POST /api/feeds/ingest-live drains through swarm and appears on dashboard."""
    req = IngestLiveFeedRequest(
        url_or_preset="usgs_earthquakes",
        feed_name="USGS Live Test",
        max_items=3,
    )
    resp = ingest_live_feed(req)
    assert resp.status_code == 200
    data = json.loads(resp.body.decode("utf-8"))
    assert data["status"] == "success"
    assert data["count"] == 3
    assert len(data["items"]) == 3

    first_item = data["items"][0]
    assert first_item["latitude"] is not None
    assert first_item["longitude"] is not None

    # Check dashboard payload
    inc_resp = get_incidents()
    assert inc_resp.status_code == 200
    inc_data = json.loads(inc_resp.body.decode("utf-8"))
    incident_ids = [i["incident_id"] for i in inc_data["incidents"]]
    assert f"INC-{first_item['item_id']}" in incident_ids


def test_api_post_ingest_empty_url_validation():
    """Test POST /api/feeds/ingest-live validates non-empty URL/preset."""
    with pytest.raises(HTTPException) as exc_info:
        ingest_live_feed(IngestLiveFeedRequest(url_or_preset="   "))
    assert exc_info.value.status_code == 400
    assert "required" in exc_info.value.detail.lower()


def test_geocoder_preserves_georss_coordinates():
    """Test GeocoderAgent retains exact GeoRSS coordinates when provided in feed item metadata."""
    bus = SwarmEventBus()
    geocoder = GeocoderAgent(bus=bus)

    feed_item = RawFeedItem(
        id="georss-test-1",
        feed_id="usgs_live",
        feed_name="USGS Live",
        title="M 4.5 Ridgecrest",
        raw_text="M 4.5 earthquake Ridgecrest CA",
        metadata={
            "latitude": 35.662,
            "longitude": -117.525,
            "display_name": "Ridgecrest, Kern County, CA",
            "provider": "GeoRSS Source Stream",
        },
    )
    from sentinelsentry.schema import TriageOutput
    incident = EnrichedIncident(
        incident_id="INC-georss-test-1",
        feed_item=feed_item,
        triage=TriageOutput(
            incident_type="Severe Weather / Flooding",
            casualty_risk_level=3,
            credibility_score=0.95,
            location_text="Ridgecrest, CA",
        ),
    )

    bus.geocoding_items.append(incident)
    geocoder.step_sync()

    assert len(bus.dashboard_items) == 1
    processed = bus.dashboard_items[0]
    assert processed.geocoded is not None
    assert processed.geocoded.latitude == pytest.approx(35.662, abs=0.001)
    assert processed.geocoded.longitude == pytest.approx(-117.525, abs=0.001)
    assert processed.geocoded.is_resolved is True
    assert processed.geocoded.provider == "GeoRSS Source Stream"
