"""
Data models and strict JSON schemas for SentinelSentry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class IncidentType(str, Enum):
    STRUCTURE_FIRE = "Structure Fire"
    VEHICLE_ACCIDENT = "Vehicle Accident / Rescue"
    MEDICAL_EMERGENCY = "Medical Emergency"
    SEVERE_WEATHER = "Severe Weather / Flooding"
    HAZMAT = "Hazardous Materials"
    CIVIL_UNREST = "Civil Unrest / Active Threat"
    GAS_LEAK = "Infrastructure Failure / Gas Leak"
    NON_EMERGENCY = "False Alarm / Non-Emergency"
    UNKNOWN = "Unknown / Other"


class TriageOutput(BaseModel):
    """
    Strict output schema emitted by Gemma 2B LoRA adapter:
    {"incident_type": "...", "casualty_risk_level": 1-5, "credibility_score": 0-1, "location_text": "..."}
    """
    incident_type: str = Field(
        ...,
        description="Classified incident category (e.g. Structure Fire, Vehicle Accident, Flooding)"
    )
    casualty_risk_level: int = Field(
        ...,
        ge=1,
        le=5,
        description="Estimated risk of casualties / severe harm on an integer scale 1 (minimal) to 5 (extreme / mass casualty)"
    )
    credibility_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Assessed credibility score of the report between 0.0 (unverified rumor) and 1.0 (official confirmed source)"
    )
    location_text: str = Field(
        ...,
        description="Raw or normalized location string extracted from the text, or 'Unknown' if not mentioned"
    )

    @field_validator("casualty_risk_level", mode="before")
    @classmethod
    def clamp_risk(cls, v: Any) -> int:
        if isinstance(v, (float, str)):
            try:
                v = int(round(float(v)))
            except (ValueError, TypeError):
                v = 1
        return max(1, min(5, int(v)))

    @field_validator("credibility_score", mode="before")
    @classmethod
    def clamp_credibility(cls, v: Any) -> float:
        if isinstance(v, (int, str)):
            try:
                v = float(v)
            except (ValueError, TypeError):
                v = 0.5
        return max(0.0, min(1.0, round(float(v), 2)))

    @field_validator("location_text", mode="before")
    @classmethod
    def sanitize_location(cls, v: Any) -> str:
        if v is None:
            return "Unknown"
        v_str = str(v).strip()
        return v_str if v_str else "Unknown"

    def to_strict_json(self) -> str:
        """Serialize to compact, single-line strict JSON matching the Gemma training target."""
        return json.dumps(
            {
                "incident_type": self.incident_type,
                "casualty_risk_level": self.casualty_risk_level,
                "credibility_score": self.credibility_score,
                "location_text": self.location_text,
            },
            separators=(",", ": ")
        )


class RawFeedItem(BaseModel):
    """
    Raw item pulled by Ingest Agents from RSS / dispatch feeds.
    """
    id: str = Field(..., description="Unique deterministic identifier (hash or GUID)")
    feed_id: str = Field(..., description="Feed identifier, e.g. 'cad_police', 'fire_rescue'")
    feed_name: str = Field(..., description="Human-readable feed title")
    title: str = Field("", description="Title from RSS item or brief dispatch headline")
    raw_text: str = Field(..., description="Messy dispatch or report text")
    published_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Publication or broadcast timestamp"
    )
    link: Optional[str] = Field(None, description="Original URL or resource link")
    guid: Optional[str] = Field(None, description="Feed item GUID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional feed metadata")


class GeocodedLocation(BaseModel):
    """
    Coordinates and address resolved by Geocoder Agent.
    """
    location_text: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    display_name: Optional[str] = None
    is_resolved: bool = False
    provider: str = "Nominatim / Cache"


class CorroborationReport(BaseModel):
    """
    Output from Verification Agent when casualty_risk_level >= 4.
    """
    triggered: bool = False
    original_credibility: float = 0.5
    adjusted_credibility: float = 0.5
    corroborating_feed_ids: List[str] = Field(default_factory=list)
    corroborating_item_ids: List[str] = Field(default_factory=list)
    confidence_delta: float = 0.0
    verification_notes: str = ""
    verified_at: Optional[datetime] = None


class EnrichedIncident(BaseModel):
    """
    Fully enriched incident sent to the Human Dispatcher live board.
    """
    incident_id: str
    feed_item: RawFeedItem
    triage: TriageOutput
    geocoded: Optional[GeocodedLocation] = None
    verification: Optional[CorroborationReport] = None
    is_escalated: bool = False
    human_status: str = "PENDING_REVIEW"  # PENDING_REVIEW, DISPATCHED, DISMISSED, ESCALATED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def summary_line(self) -> str:
        loc = (self.geocoded.display_name if (self.geocoded and self.geocoded.is_resolved and self.geocoded.display_name) else self.triage.location_text) or "Unknown"
        esc = "[ESCALATED]" if self.is_escalated else "[ROUTINE]"
        return f"{esc} [{self.triage.incident_type}] Risk:{self.triage.casualty_risk_level}/5 | Cred:{self.triage.credibility_score:.2f} | Loc:{loc}"
