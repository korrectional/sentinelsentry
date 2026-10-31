"""
Verification agent triggered when casualty_risk_level >= 4.
Performs cross-feed corroboration and adjusts credibility score.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Tuple

from sentinelsentry.config import settings
from sentinelsentry.schema import RawFeedItem, EnrichedIncident, CorroborationReport
from sentinelsentry.agents.base import BaseAgent, SwarmEventBus

logger = logging.getLogger("sentinelsentry.agents.verification")


class VerificationAgent(BaseAgent):
    """
    Triggered when casualty_risk_level >= 4.
    Searches sliding window across other feeds for corroborating evidence,
    adjusts credibility score, and triggers escalation flags.
    """

    def __init__(self, bus: SwarmEventBus):
        super().__init__(agent_name="VerificationAgent", bus=bus)
        self.verifications_completed = 0
        self.escalations_triggered = 0

    LOCATION_STOPWORDS = {
        "near", "with", "from", "road", "street", "avenue", "lane", "drive",
        "blvd", "pkwy", "trail", "highway", "interstate", "structure", "residential",
        "commercial", "building", "fire", "smoke", "flames", "flame", "alarm",
        "water", "flood", "flooding", "patient", "medical", "assist", "accident",
        "crash", "rollover", "trapped", "pinned", "occupant", "unaccounted",
        "police", "officer", "medic", "engine", "ladder", "units", "code",
        "scene", "progress", "evacuation", "evac", "report", "reported",
        "vehicle", "damage", "lines", "active", "threat", "severe",
        "weather", "shots", "shooting", "hazard", "fumes", "leak",
        "alert", "critical", "heavy", "minor", "inside", "outside",
        "someone", "people", "everyone", "sirens", "downtown", "north",
        "south", "east", "west", "approx", "county", "raleigh", "cary",
    }

    def _extract_location_tokens(self, text: str) -> set[str]:
        words = set()
        clean_text = text.replace(",", " ").replace(".", " ").replace("&", " ").replace("/", " ").replace("-", " ")
        for token in clean_text.split():
            clean = token.strip().lower()
            if len(clean) >= 3 and clean not in self.LOCATION_STOPWORDS:
                words.add(clean)
        return words

    TYPE_CRISIS_KEYWORDS = {
        "Structure Fire": {"fire", "smoke", "flames", "burn", "alarm", "explosion", "explosions", "warehouse", "building", "hazmat", "standby", "2-alarm", "2nd alarm"},
        "Vehicle Accident / Rescue": {"accident", "crash", "rollover", "extrication", "highway", "interstate", "traffic", "trauma", "patients", "car", "suv", "truck", "standstill", "lanes", "jaws"},
        "Severe Weather / Flooding": {"flood", "flooding", "water", "creek", "rain", "swiftwater", "rescue", "gauge", "weather", "storm", "submerged", "windshields"},
        "Infrastructure Failure / Gas Leak": {"gas", "leak", "odor", "smell", "fumes", "evac", "evacuating", "rupture", "pipeline", "rotten"},
        "Civil Unrest / Active Threat": {"shots", "shooting", "gun", "threat", "fight", "disturbance", "police", "crowd", "popping", "running"},
        "Medical Emergency": {"cpr", "medic", "cardiac", "trauma", "unresponsive", "patient", "patients", "rescue", "ems", "burn", "burns"},
        "Hazardous Materials": {"hazmat", "chemical", "fumes", "diesel", "spill", "ammonia", "drum"},
    }

    def _find_corroborations(self, incident: EnrichedIncident, recent_items: List[RawFeedItem]) -> List[Tuple[RawFeedItem, float]]:
        target_feed = incident.feed_item.feed_id
        target_time = incident.feed_item.published_at

        target_loc_text = incident.triage.location_text.lower().strip()
        has_valid_target_loc = target_loc_text != "unknown" and len(target_loc_text) > 3
        if not has_valid_target_loc:
            # Cannot cross-corroborate an incident with an unknown location
            return []

        target_loc_tokens = self._extract_location_tokens(incident.triage.location_text)
        if not target_loc_tokens:
            return []

        target_keywords = self.TYPE_CRISIS_KEYWORDS.get(incident.triage.incident_type, set())

        corroborating = []
        for cand in recent_items:
            if cand.feed_id == target_feed:
                continue

            dt_seconds = abs((cand.published_at - target_time).total_seconds())
            if dt_seconds > settings.CORROBORATION_TIME_WINDOW_SECONDS:
                continue

            # Prevent weather spotter feeds from corroborating non-weather crimes
            if cand.feed_id == "weather_skywarn" and incident.triage.incident_type != "Severe Weather / Flooding":
                continue

            # Check location overlap
            cand_tokens = self._extract_location_tokens(cand.raw_text)
            loc_overlap = bool(target_loc_tokens.intersection(cand_tokens))

            # Semantic / scenario compatibility check
            same_scenario = bool(
                cand.metadata.get("scenario_id")
                and cand.metadata.get("scenario_id") == incident.feed_item.metadata.get("scenario_id")
            )
            cand_text_lower = cand.raw_text.lower()
            semantic_overlap = any(kw in cand_text_lower for kw in target_keywords) if target_keywords else False
            is_official_cross_call = cand.feed_id in ("cad_police", "fire_rescue", "ems_scanner") and incident.feed_item.feed_id in ("cad_police", "fire_rescue", "ems_scanner")

            if loc_overlap and (same_scenario or semantic_overlap or is_official_cross_call):
                weight = 0.25 if cand.feed_id in ("cad_police", "fire_rescue", "ems_scanner") else 0.15
                corroborating.append((cand, weight))

        return corroborating

    def verify_incident_sync(self, incident: EnrichedIncident) -> EnrichedIncident:
        recent_items = self.bus.get_recent_items_snapshot_sync()
        corroborators = self._find_corroborations(incident, recent_items)

        original_cred = incident.triage.credibility_score
        corrob_feed_ids = [c[0].feed_id for c in corroborators]
        corrob_item_ids = [c[0].id for c in corroborators]

        if corroborators:
            total_delta = sum(c[1] for c in corroborators)
            adjusted_cred = min(1.0, original_cred + total_delta)
            notes = (
                f"VERIFIED: Corroborated by {len(corroborators)} cross-stream report(s) "
                f"({', '.join(set(corrob_feed_ids))}). Credibility boosted from {original_cred:.2f} to {adjusted_cred:.2f}."
            )
            report = CorroborationReport(
                triggered=True,
                original_credibility=original_cred,
                adjusted_credibility=adjusted_cred,
                corroborating_feed_ids=corrob_feed_ids,
                corroborating_item_ids=corrob_item_ids,
                confidence_delta=total_delta,
                verification_notes=notes,
                verified_at=datetime.now(timezone.utc),
            )
        else:
            adjusted_cred = original_cred
            notes = f"UNVERIFIED: Solo feed alert from {incident.feed_item.feed_id}. No cross-feed corroboration found within {settings.CORROBORATION_TIME_WINDOW_SECONDS}s."
            report = CorroborationReport(
                triggered=True,
                original_credibility=original_cred,
                adjusted_credibility=adjusted_cred,
                corroborating_feed_ids=[],
                corroborating_item_ids=[],
                confidence_delta=0.0,
                verification_notes=notes,
                verified_at=datetime.now(timezone.utc),
            )

        incident.triage.credibility_score = adjusted_cred
        incident.verification = report

        is_unknown_loc = incident.triage.location_text.lower().strip() == "unknown"
        is_unknown_type = incident.triage.incident_type == "Unknown / Other"

        if is_unknown_loc or is_unknown_type:
            incident.is_escalated = False
            if is_unknown_loc and is_unknown_type:
                incident.human_status = "NEEDS REVIEW (Unknown Type & Location)"
            elif is_unknown_loc:
                incident.human_status = "NEEDS REVIEW (Location Unresolved)"
            else:
                incident.human_status = "NEEDS REVIEW (Unclassified Emergency)"
        elif incident.triage.casualty_risk_level >= settings.HIGH_RISK_THRESHOLD and adjusted_cred >= settings.ESCALATION_MIN_CREDIBILITY:
            incident.is_escalated = True
            incident.human_status = "ESCALATED"
            self.escalations_triggered += 1
        else:
            incident.is_escalated = False
            incident.human_status = "ROUTINE"

        self.verifications_completed += 1
        return incident

    def step_sync(self) -> int:
        count = 0
        while self.bus.verification_items:
            incident = self.bus.verification_items.popleft()
            verified = self.verify_incident_sync(incident)
            self.bus.geocoding_items.append(verified)
            count += 1
        return count
