"""
Adversarial test matrix for Gemma 4 triage inference engine:
- All 8 incident taxonomy types
- Police 10-codes & radio shorthand
- Visual damage & multimodal base64 variations
- Boundary invariants: risk in [1, 5], credibility in [0.0, 1.0]
"""

import json
from sentinelsentry.schema import IncidentType, TriageOutput
from sentinelsentry.agents.triage import LocalTriageEngine


def test_triage_all_eight_taxonomy_categories():
    """Verify engine correctly categorizes distinct crisis patterns into the 8 taxonomy classes."""
    engine = LocalTriageEngine()

    test_cases = [
        ("10-70 COMM BLDG 1200 S SAUNDERS ST FLAMES VENTING ROOF", IncidentType.STRUCTURE_FIRE),
        ("10-50 PI MULTI-VEHICLE ROLLOVER I-40 EB MM 287 JAWS IN SERVICE", IncidentType.VEHICLE_ACCIDENT),
        ("CARDIAC ARREST 65YO MALE CPR IN PROGRESS WAKEMED EMS EN ROUTE", IncidentType.MEDICAL_EMERGENCY),
        ("FLASH FLOOD RAPID SWIFTWATER ATLANTIC AVE ROADS SUBMERGED", IncidentType.SEVERE_WEATHER),
        ("10-75 HAZMAT AMMONIA TANKER LEAK VAPOR CLOUD LEVEL A", IncidentType.HAZMAT),
        ("10-33 CODE 3 ACTIVE SHOOTER SHOTS FIRED SALISBURY ST", IncidentType.CIVIL_UNREST),
        ("GAS MAIN RUPTURE BY BACKHOE ROARING HISS EVACUATING 500FT", IncidentType.GAS_LEAK),
        ("COMMERCIAL WATERFLOW ALARM ACCIDENTAL TRIP TECHNICIAN TESTING", IncidentType.NON_EMERGENCY),
    ]

    for raw_text, expected_type in test_cases:
        triage = engine.predict(raw_text)
        assert triage.incident_type == expected_type.value, (
            f"Failed for '{raw_text}': got '{triage.incident_type}', expected '{expected_type.value}'"
        )
        assert 1 <= triage.casualty_risk_level <= 5
        assert 0.0 <= triage.credibility_score <= 1.0
        assert triage.location_text != ""


def test_triage_multimodal_variations():
    """Verify multimodal input works with various image data formats (svg, base64, data URIs)."""
    engine = LocalTriageEngine()

    # 1. SVG Data URI
    svg_uri = "data:image/svg+xml;utf8,<svg><rect width='100' height='100' fill='red'/><text>FIRE</text></svg>"
    out1 = engine.predict(
        raw_text="Warehouse fire with heavy smoke on Saunders St",
        image_data=svg_uri,
        feed_source="citizen_photo",
    )
    assert out1.incident_type == "Structure Fire"
    assert out1.casualty_risk_level >= 4
    assert out1.credibility_score >= 0.70

    # 2. Base64 payload with damage markers
    base64_data = "data:image/jpeg;base64,/9j/4AAQSkZJRg_submerged_car_water_rescue_evidence"
    out2 = engine.predict(
        raw_text="Car washed off road at Crabtree Creek",
        image_data=base64_data,
        feed_source="spotter",
    )
    assert out2.incident_type == "Severe Weather / Flooding"
    assert out2.casualty_risk_level >= 3

    # 3. None / empty image data falls back gracefully
    out3 = engine.predict(
        raw_text="Routine vehicle check parking lot no damage",
        image_data=None,
        feed_source="cad",
    )
    assert 1 <= out3.casualty_risk_level <= 2


def test_triage_location_number_isolation():
    """Verify street numbers (e.g. 1200, 500, 401) are isolated from 10-codes (10-70, 10-50)."""
    engine = LocalTriageEngine()

    # 10-70 with 1200 S Saunders
    t1 = engine.predict("10-70 COMM BLDG 1200 S SAUNDERS ST FLAMES")
    assert "1200" in t1.location_text
    assert "Saunders" in t1.location_text
    assert "70" not in t1.location_text.split()[0]  # First token must not be 70

    # 10-33 with 500 S Salisbury
    t2 = engine.predict("10-33 CODE 3 ACTIVE 500 S SALISBURY ST")
    assert "500" in t2.location_text
    assert "Salisbury" in t2.location_text
    assert "33" not in t2.location_text.split()[0]


def test_triage_strict_json_serialization():
    """Verify all outputs strictly validate as standard JSON conforming to the schema."""
    engine = LocalTriageEngine()
    samples = [
        "10-50 PI I-40 EB MM 287",
        "Minor leak near water heater non-emergency",
        "UNKNOWN EMERGENCY SITUATION ON GLENWOOD AVE",
        "Heavy smoke coming from apartment kitchen",
    ]

    for s in samples:
        triage = engine.predict(s)
        json_str = triage.to_strict_json()
        parsed = json.loads(json_str)

        assert "incident_type" in parsed
        assert "casualty_risk_level" in parsed
        assert "credibility_score" in parsed
        assert "location_text" in parsed

        # Invariant checks
        assert parsed["casualty_risk_level"] in [1, 2, 3, 4, 5]
        assert 0.0 <= parsed["credibility_score"] <= 1.0
        assert isinstance(parsed["location_text"], str)
