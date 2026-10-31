"""
Alert and dashboard agent consolidating incidents for human dispatcher decision support.
Simulates Maritime micro-VM isolation for the central alert hub.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table

from sentinelsentry.schema import EnrichedIncident
from sentinelsentry.agents.base import BaseAgent, SwarmEventBus

logger = logging.getLogger("sentinelsentry.agents.dashboard")


class AlertDashboardAgent(BaseAgent):
    """
    Consolidates enriched incidents into a live decision-support board for the human dispatcher.
    Provides visual alerts, escalation logs, and review queues.
    """

    def __init__(self, bus: SwarmEventBus, console: Optional[Console] = None):
        super().__init__(
            agent_name="AlertDashboardAgent",
            bus=bus,
            micro_vm_id="maritime-vm-alerter-01",
            agent_role="Dispatcher Decision-Support Hub",
        )
        self.console = console or Console(legacy_windows=False, force_terminal=True)
        self.incidents: Dict[str, EnrichedIncident] = {}
        self.escalated_ids: List[str] = []
        self._corroborated_ids: set = set()
        self.stats = {
            "total_processed": 0,
            "escalated_count": 0,
            "corroborated_count": 0,
            "routine_count": 0,
        }

    def process_incident(self, incident: EnrichedIncident) -> None:
        is_new = incident.incident_id not in self.incidents
        was_escalated = not is_new and self.incidents[incident.incident_id].is_escalated
        self.incidents[incident.incident_id] = incident

        if is_new:
            self.stats["total_processed"] += 1
            if incident.is_escalated:
                self.stats["escalated_count"] += 1
                if incident.incident_id not in self.escalated_ids:
                    self.escalated_ids.append(incident.incident_id)
            else:
                self.stats["routine_count"] += 1
        else:
            if incident.is_escalated and not was_escalated:
                self.stats["escalated_count"] += 1
                self.stats["routine_count"] = max(0, self.stats["routine_count"] - 1)
                if incident.incident_id not in self.escalated_ids:
                    self.escalated_ids.append(incident.incident_id)

        if incident.verification and incident.verification.corroborating_feed_ids:
            if incident.incident_id not in self._corroborated_ids:
                self._corroborated_ids.add(incident.incident_id)
                self.stats["corroborated_count"] += 1

    def step_sync(self, max_items: Optional[int] = None) -> int:
        count = 0
        limit = max_items if max_items is not None else float("inf")
        while self.bus.dashboard_items and count < limit:
            incident = self.bus.dashboard_items.popleft()
            self.process_incident(incident)
            count += 1

        self.processed_count = self.stats["total_processed"]
        self.record_heartbeat("working" if count > 0 else "idle")
        return count

    def render_board_table(self, limit: int = 10) -> Table:
        table = Table(
            title="[ALERT] SentinelSentry: Live Dispatcher Decision-Support Board",
            header_style="bold cyan",
            show_lines=True,
            expand=False,
        )
        table.add_column("Time", justify="center", style="dim", min_width=8, max_width=10, no_wrap=True)
        table.add_column("Feed", justify="center", min_width=10, max_width=15, no_wrap=True)
        table.add_column("Incident Type", style="bold", min_width=14, max_width=22)
        table.add_column("Risk", justify="center", min_width=7, max_width=9, no_wrap=True)
        table.add_column("Cred", justify="center", min_width=5, max_width=6, no_wrap=True)
        table.add_column("Location", min_width=14, max_width=24)
        table.add_column("Status / Corroboration", min_width=16, max_width=30)

        sorted_incidents = sorted(
            self.incidents.values(),
            key=lambda inc: (inc.is_escalated, inc.triage.casualty_risk_level, inc.created_at),
            reverse=True,
        )[:limit]

        for inc in sorted_incidents:
            t = inc.created_at.strftime("%H:%M:%S")
            feed_badge = inc.feed_item.feed_id

            risk_val = inc.triage.casualty_risk_level
            if risk_val >= 4:
                risk_str = f"[bold red]{risk_val}/5 (!)[/bold red]"
            elif risk_val == 3:
                risk_str = f"[yellow]{risk_val}/5[/yellow]"
            else:
                risk_str = f"[green]{risk_val}/5[/green]"

            cred_val = inc.triage.credibility_score
            cred_str = f"[bold]{cred_val:.2f}[/bold]" if cred_val >= 0.8 else f"{cred_val:.2f}"

            loc_str = inc.geocoded.display_name if (inc.geocoded and inc.geocoded.is_resolved) else inc.triage.location_text
            if len(loc_str) > 23:
                loc_str = loc_str[:21] + ".."

            if inc.is_escalated:
                corrob_text = "[bold red]** ESCALATED **[/bold red]"
                if inc.verification and inc.verification.corroborating_feed_ids:
                    corrobs = ", ".join(inc.verification.corroborating_feed_ids)
                    corrob_text += f"\n[yellow]Corrob: {corrobs}[/yellow]"
                else:
                    corrob_text += "\n[dim]Solo extreme alert[/dim]"
            elif "NEEDS REVIEW" in inc.human_status:
                corrob_text = f"[bold yellow]{inc.human_status}[/bold yellow]"
            else:
                corrob_text = f"[dim green]{inc.human_status}[/dim green]"

            table.add_row(
                t,
                feed_badge,
                inc.triage.incident_type,
                risk_str,
                cred_str,
                loc_str,
                corrob_text,
            )

        return table

    def _find_incident(self, incident_id: str) -> Optional[EnrichedIncident]:
        """Find incident by ID, accepting with or without 'INC-' prefix."""
        if incident_id in self.incidents:
            return self.incidents[incident_id]
        if f"INC-{incident_id}" in self.incidents:
            return self.incidents[f"INC-{incident_id}"]
        if incident_id.startswith("INC-") and incident_id[4:] in self.incidents:
            return self.incidents[incident_id[4:]]
        return None

    def get_incident(self, incident_id: str) -> Optional[EnrichedIncident]:
        return self._find_incident(incident_id)

    def get_pending_escalations(self) -> List[EnrichedIncident]:
        """Returns incidents requiring urgent human dispatcher action."""
        return [
            inc for inc in self.incidents.values()
            if inc.is_escalated
        ]

    def acknowledge_incident(self, incident_id: str) -> EnrichedIncident:
        """Dispatcher action: Acknowledge an active emergency escalation."""
        inc = self._find_incident(incident_id)
        if not inc:
            raise KeyError(f"Incident '{incident_id}' not found.")
        actual_id = inc.incident_id
        inc.is_escalated = False
        inc.human_status = "ACKNOWLEDGED"
        inc.metadata["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
        if actual_id in self.escalated_ids:
            self.escalated_ids.remove(actual_id)
        self.stats["escalated_count"] = max(0, self.stats["escalated_count"] - 1)
        self.stats["routine_count"] += 1
        logger.info(f"[Dispatcher] Acknowledged incident {actual_id}")
        return inc

    def dispatch_incident(
        self,
        incident_id: str,
        units: Optional[List[str]] = None,
        dispatcher_notes: Optional[str] = None,
    ) -> EnrichedIncident:
        """Dispatcher action: Assign units and mark incident as DISPATCHED."""
        inc = self._find_incident(incident_id)
        if not inc:
            raise KeyError(f"Incident '{incident_id}' not found.")
        actual_id = inc.incident_id
        assigned = units or ["Engine 4", "Medic 2", "Ladder 1"]
        inc.human_status = f"DISPATCHED ({', '.join(assigned)})"
        inc.is_escalated = False
        if actual_id in self.escalated_ids:
            self.escalated_ids.remove(actual_id)
        if dispatcher_notes:
            inc.metadata["dispatcher_notes"] = dispatcher_notes
        inc.metadata["dispatched_units"] = assigned
        inc.metadata["dispatched_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"[Dispatcher] Dispatched incident {actual_id} to {assigned}")
        return inc

    def dismiss_incident(self, incident_id: str, reason: str = "False Alarm") -> EnrichedIncident:
        """Dispatcher action: Dismiss incident without dispatch."""
        inc = self._find_incident(incident_id)
        if not inc:
            raise KeyError(f"Incident '{incident_id}' not found.")
        actual_id = inc.incident_id
        inc.human_status = f"DISMISSED ({reason})"
        inc.is_escalated = False
        if actual_id in self.escalated_ids:
            self.escalated_ids.remove(actual_id)
        inc.metadata["dismissed_reason"] = reason
        inc.metadata["dismissed_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"[Dispatcher] Dismissed incident {actual_id}: {reason}")
        return inc

    def escalate_incident(self, incident_id: str, reason: str = "Supervisor Override") -> EnrichedIncident:
        """Dispatcher action: Manually escalate an incident for priority response."""
        inc = self._find_incident(incident_id)
        if not inc:
            raise KeyError(f"Incident '{incident_id}' not found.")
        actual_id = inc.incident_id
        inc.is_escalated = True
        inc.human_status = f"MANUALLY ESCALATED ({reason})"
        inc.metadata["manual_escalation_reason"] = reason
        if actual_id not in self.escalated_ids:
            self.escalated_ids.append(actual_id)
        logger.info(f"[Dispatcher] Manually escalated incident {actual_id}: {reason}")
        return inc

    def render_incident_detail(self, incident_id: str) -> str:
        """Render a formatted string or card detailing a specific incident."""
        inc = self._find_incident(incident_id)
        if not inc:
            return f"Incident {incident_id} not found."
        lines = [
            f"=== INCIDENT DETAIL: {inc.incident_id} ===",
            f"Type: {inc.triage.incident_type}",
            f"Casualty Risk Level: {inc.triage.casualty_risk_level}/5",
            f"Credibility Score: {inc.triage.credibility_score:.2f}",
            f"Location: {inc.triage.location_text}",
            f"Status: {inc.human_status}",
            f"Escalated: {inc.is_escalated}",
            f"Feed: {inc.feed_item.feed_id} ({inc.feed_item.feed_name})",
            f"Raw Text: {inc.feed_item.raw_text}",
        ]
        if inc.verification and inc.verification.corroborating_feed_ids:
            lines.append(f"Corroborating Feeds: {', '.join(inc.verification.corroborating_feed_ids)}")
        if "dispatched_units" in inc.metadata:
            lines.append(f"Dispatched Units: {', '.join(inc.metadata['dispatched_units'])}")
        return "\n".join(lines)

    def get_incidents_payload(self) -> List[Dict[str, Any]]:
        """Serialize incidents for the web dashboard."""
        result = []
        for inc in sorted(self.incidents.values(), key=lambda x: (x.is_escalated, x.triage.casualty_risk_level, x.created_at), reverse=True):
            lat = inc.geocoded.latitude if (inc.geocoded and inc.geocoded.latitude is not None) else 35.7796
            lon = inc.geocoded.longitude if (inc.geocoded and inc.geocoded.longitude is not None) else -78.6382
            result.append({
                "incident_id": inc.incident_id,
                "feed_id": inc.feed_item.feed_id,
                "feed_name": inc.feed_item.feed_name,
                "title": inc.feed_item.title or inc.triage.incident_type,
                "raw_text": inc.feed_item.raw_text,
                "time": inc.created_at.strftime("%H:%M:%S"),
                "incident_type": inc.triage.incident_type,
                "casualty_risk_level": inc.triage.casualty_risk_level,
                "credibility_score": round(inc.triage.credibility_score, 2),
                "location_text": inc.triage.location_text,
                "display_name": inc.geocoded.display_name if (inc.geocoded and inc.geocoded.display_name) else inc.triage.location_text,
                "latitude": lat,
                "longitude": lon,
                "is_geocoded": bool(inc.geocoded and inc.geocoded.is_resolved),
                "is_escalated": inc.is_escalated,
                "human_status": inc.human_status,
                "image_data": inc.image_data,
                "image_mime_type": inc.image_mime_type,
                "corroborating_feed_ids": inc.verification.corroborating_feed_ids if inc.verification else [],
                "confidence_delta": inc.verification.confidence_delta if inc.verification else 0.0,
                "verification_notes": inc.verification.verification_notes if inc.verification else "",
            })
        return result
