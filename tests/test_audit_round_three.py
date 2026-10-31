"""
Tests from Audit Round 3:
Validates geocoder landmark precision (anti-false-positive matching),
RSS XML None-safety, summary_line null-safety, and benchmark empty-data validation.
"""

import pytest
import tempfile
from pathlib import Path
from sentinelsentry.schema import (
    TriageOutput,
    RawFeedItem,
    GeocodedLocation,
    EnrichedIncident,
)
from sentinelsentry.agents.base import SwarmEventBus
from sentinelsentry.agents.geocoder import GeocoderAgent
from sentinelsentry.feeds.generator import RSSFeedGenerator
from sentinelsentry.eval.benchmark import BenchmarkRunner


def test_geocoder_anti_false_positive_matching():
    """Generic street names like 'Oakwood St' or 'Main Ave' must NOT falsely match 'Saunders St'."""
    bus = SwarmEventBus()
    ga = GeocoderAgent(bus, enable_network=False)

    # Generic street ending in 'St' should NOT match '1200 S Saunders St'
    g_oakwood = ga.resolve_location_sync("Oakwood St")
    assert not g_oakwood.is_resolved
    assert "Saunders" not in g_oakwood.display_name

    # Generic street ending in 'Ave' should NOT match 'Atlantic Ave' or '3000 New Bern Ave'
    g_main = ga.resolve_location_sync("Main Ave")
    assert not g_main.is_resolved

    # Legitimate landmark with significant tokens MUST match
    g_saunders = ga.resolve_location_sync("1200 S Saunders St")
    assert g_saunders.is_resolved
    assert "Saunders" in g_saunders.display_name


def test_rss_generator_handles_none_title_and_link():
    """RSS XML generation must not throw AttributeError when link or scenario is None."""
    gen = RSSFeedGenerator(
        feed_id="test_safe",
        feed_name="Safe Generator Test",
        description="Testing None-safety",
    )
    # Explicitly pass None for optional fields
    guid = gen.add_item(
        raw_text="Test incident text",
        title=None,
        link=None,
        scenario_id=None,
    )
    xml_output = gen.to_rss_xml()
    assert "<rss version=\"2.0\"" in xml_output
    assert f"<guid isPermaLink=\"false\">{guid}</guid>" in xml_output
    assert "Test incident text" in xml_output


def test_summary_line_null_safety():
    """summary_line property must format gracefully without NoneType errors."""
    raw = RawFeedItem(id="raw-null", feed_id="cad_police", feed_name="CAD", raw_text="Sample dispatch")
    triage = TriageOutput(incident_type="Structure Fire", casualty_risk_level=4, credibility_score=0.9, location_text=None)

    # 1. Without geocoded object
    inc1 = EnrichedIncident(incident_id="INC-1", feed_item=raw, triage=triage, geocoded=None)
    assert "[ROUTINE] [Structure Fire]" in inc1.summary_line
    assert "Loc:Unknown" in inc1.summary_line

    # 2. With geocoded object where display_name is None
    geo_none = GeocodedLocation(location_text="Unknown", display_name=None, is_resolved=False)
    inc2 = EnrichedIncident(incident_id="INC-2", feed_item=raw, triage=triage, geocoded=geo_none)
    assert "Loc:Unknown" in inc2.summary_line


def test_benchmark_runner_empty_file_raises_value_error():
    """BenchmarkRunner must raise ValueError when given an empty test file."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
        f.write("\n")
        temp_path = Path(f.name)

    try:
        runner = BenchmarkRunner(temp_path)
        with pytest.raises(ValueError, match="No test samples found"):
            runner.evaluate()
    finally:
        temp_path.unlink(missing_ok=True)
