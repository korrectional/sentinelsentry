"""
Unit tests for RSS Feed Generator and RSS Feed Client deduplication.
"""

import pytest
from datetime import datetime, timezone
from sentinelsentry.config import FeedConfig
from sentinelsentry.feeds.generator import RSSFeedGenerator
from sentinelsentry.feeds.client import RSSFeedClient, DeduplicationCache
from sentinelsentry.feeds.server import FeedServerManager


def test_rss_generator_generates_valid_xml():
    gen = RSSFeedGenerator(
        feed_id="test_police",
        feed_name="Test Police CAD",
        description="Test Feed for CAD dispatch",
        feed_link="http://localhost:8765/feeds",
    )
    guid = gen.add_item(
        raw_text="10-50 PI at Old Wake Forest & Millbrook, veh flipped smoke showing 2 pinned",
        title="CAD 911: 10-50 PI Old Wake Forest",
        scenario_id="TEST-001",
    )
    xml_str = gen.to_rss_xml()

    assert "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" in xml_str
    assert "<rss version=\"2.0\"" in xml_str
    assert "<title>Test Police CAD</title>" in xml_str
    assert "10-50 PI at Old Wake Forest" in xml_str
    assert f"<guid isPermaLink=\"false\">{guid}</guid>" in xml_str
    assert "<category>TEST-001</category>" in xml_str


def test_rss_client_parse_and_deduplication():
    gen = RSSFeedGenerator(
        feed_id="test_fire",
        feed_name="Test Fire Rescue",
        description="Test Feed for Fire dispatch",
    )
    guid1 = gen.add_item(
        raw_text="STRUCTURE FIRE: 2-alarm commercial at 1200 S Saunders St",
        title="Fire Alert Saunders",
    )

    xml_content = gen.to_rss_xml()

    config = FeedConfig(
        id="test_fire",
        name="Test Fire Rescue",
        description="Test Feed",
        url="http://test/fire.xml",
    )
    client = RSSFeedClient(config)

    # First parse: should discover 1 item
    items1 = client.parse_feed_content(xml_content)
    assert len(items1) == 1
    assert items1[0].id == guid1
    assert items1[0].feed_id == "test_fire"
    assert "1200 S Saunders St" in items1[0].raw_text

    # Second parse with same XML: should return 0 items because of deduplication
    items2 = client.parse_feed_content(xml_content)
    assert len(items2) == 0

    # Add a new item to the feed generator
    guid2 = gen.add_item(
        raw_text="GAS LEAK: 401 Fayetteville St downtown",
        title="Gas Leak Alert",
    )
    xml_updated = gen.to_rss_xml()

    # Third parse: should only discover the 1 new item
    items3 = client.parse_feed_content(xml_updated)
    assert len(items3) == 1
    assert items3[0].id == guid2
    assert "401 Fayetteville St" in items3[0].raw_text


def test_deduplication_cache_lru_eviction():
    cache = DeduplicationCache(max_size=3)
    cache.mark_seen("item_1")
    cache.mark_seen("item_2")
    cache.mark_seen("item_3")

    assert cache.is_seen("item_1")
    assert cache.is_seen("item_2")
    assert cache.is_seen("item_3")

    # Add 4th item, oldest item ('item_1') should be evicted
    cache.mark_seen("item_4")
    assert not cache.is_seen("item_1")
    assert cache.is_seen("item_4")


def test_feed_server_manager_initialization():
    manager = FeedServerManager()
    assert "cad_police" in manager.generators
    assert "fire_rescue" in manager.generators
    assert "ems_scanner" in manager.generators
    assert "weather_skywarn" in manager.generators
    assert "social_crisis" in manager.generators

    xml_cad = manager.get_feed_xml("cad_police")
    assert "<rss version=\"2.0\"" in xml_cad
    assert "CAD" in xml_cad

    # Test dynamic injection
    injected_guid = manager.inject_item(
        feed_id="cad_police",
        raw_text="10-33 OFFICER CALL FOR ASSISTANCE AT 500 S SALISBURY",
        title="CAD 911: 10-33 CALL FOR ASSIST",
    )
    assert len(injected_guid) > 0
    updated_xml = manager.get_feed_xml("cad_police")
    assert injected_guid in updated_xml
