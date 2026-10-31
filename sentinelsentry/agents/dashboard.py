"""
Alert and dashboard agent consolidating incidents for human dispatcher decision support.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
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
        super().__init__(agent_name="AlertDashboardAgent", bus=bus)
        self.console = console or Console(legacy_windows=False, force_terminal=True)
        self.incidents: Dict[str, EnrichedIncident] = {}
        self.escalated_ids: List[str] = []
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
            self.stats["corroborated_count"] += 1

    def step_sync(self) -> int:
        count = 0
        while self.bus.dashboard_items:
            incident = self.bus.dashboard_items.popleft()
            self.process_incident(incident)
            count += 1
        return count

    def render_board_table(self, limit: int = 10) -> Table:
        table = Table(
            title="[ALERT] SentinelSentry: Live Dispatcher Decision-Support Board",
            header_style="bold cyan",
            show_lines=True,
        )
        table.add_column("Time", justify="center", style="dim", width=10)
        table.add_column("Feed", justify="center", width=14)
        table.add_column("Incident Type", style="bold", width=22)
        table.add_column("Risk", justify="center", width=8)
        table.add_column("Cred", justify="center", width=8)
        table.add_column("Location", width=25)
        table.add_column("Status / Corroboration", width=35)

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
            if len(loc_str) > 24:
                loc_str = loc_str[:22] + ".."

            if inc.is_escalated:
                corrob_text = f"[bold red]** ESCALATED **[/bold red]\n"
                if inc.verification and inc.verification.corroborating_feed_ids:
                    corrob_text += f"[yellow]Corrob: {','.join(inc.verification.corroborating_feed_ids)}[/yellow]"
                else:
                    corrob_text += "[dim]Solo extreme alert[/dim]"
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
