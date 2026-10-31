"""
SentinelSentry Quick Smoke Test Runner.
Executes an end-to-end pipeline pass through Ingest, Triage, Verification,
Geocoding, and Live Dashboard Decision Support.
"""

import sys

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.panel import Panel

from sentinelsentry.config import FeedConfig
from sentinelsentry.agents.base import SwarmEventBus, SwarmCoordinator
from sentinelsentry.agents.ingest import IngestAgent
from sentinelsentry.agents.triage import TriageAgent, LocalTriageEngine
from sentinelsentry.agents.verification import VerificationAgent
from sentinelsentry.agents.geocoder import GeocoderAgent
from sentinelsentry.agents.dashboard import AlertDashboardAgent
from sentinelsentry.feeds.generator import RSSFeedGenerator


def main():
    console = Console(force_terminal=True, legacy_windows=False)
    console.print(
        Panel.fit(
            "[bold cyan]SentinelSentry End-to-End Smoke Test[/bold cyan]\n"
            "[dim]Testing Ingest -> Gemma Triage -> Cross-Feed Verification -> Geocoder -> Live Board[/dim]",
            title="[bold green]RUNNING SMOKE TEST[/bold green]",
        )
    )

    bus = SwarmEventBus()
    coordinator = SwarmCoordinator(bus=bus)

    # 1. Setup simulated feed generator with test incidents
    gen_fire = RSSFeedGenerator(feed_id="fire_rescue", feed_name="Wake Fire Dispatch", description="Fire ops")
    gen_police = RSSFeedGenerator(feed_id="cad_police", feed_name="Raleigh 911 CAD", description="Police CAD")

    # Add corroborating high-severity warehouse fire items
    gen_fire.add_item(
        raw_text="STRUCTURE FIRE: 2-alarm commercial at 1200 S Saunders St, heavy smoke venting through roof, occupant trapped",
        title="2-Alarm Fire Saunders",
        scenario_id="SCN-001",
    )
    gen_police.add_item(
        raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF EVAC IN PROGRESS 1 SUBJ INSIDE",
        title="CAD: 10-70 Saunders",
        scenario_id="SCN-001",
    )
    # Add routine non-emergency call
    gen_police.add_item(
        raw_text="10-50 PD ONLY CRABTREE VALLEY MALL LOT. MINOR FENDER BENDER NO INJURIES",
        title="CAD: 10-50 PD Crabtree",
        scenario_id="SCN-ROUTINE",
    )

    # 2. Register Agents
    cfg_fire = FeedConfig(id="fire_rescue", name="Wake Fire", description="Fire", url="http://local/fire")
    cfg_police = FeedConfig(id="cad_police", name="Raleigh CAD", description="CAD", url="http://local/police")

    coordinator.register_ingest_agent(IngestAgent(cfg_fire, bus, custom_fetcher=lambda: gen_fire.to_raw_feed_items()))
    coordinator.register_ingest_agent(IngestAgent(cfg_police, bus, custom_fetcher=lambda: gen_police.to_raw_feed_items()))
    coordinator.register_triage_worker(TriageAgent(bus, worker_id=1, engine=LocalTriageEngine()))
    coordinator.register_triage_worker(TriageAgent(bus, worker_id=2, engine=LocalTriageEngine()))
    coordinator.set_verification_agent(VerificationAgent(bus))
    coordinator.set_geocoder_agent(GeocoderAgent(bus, enable_network=False))
    dashboard_agent = AlertDashboardAgent(bus, console=console)
    coordinator.set_dashboard_agent(dashboard_agent)

    # 3. Drain the pipeline
    console.print("[dim]Stepping multi-agent coordinator through queues...[/dim]\n")
    rounds = coordinator.drain(max_rounds=5)

    # 4. Display Live Decision Board
    console.print(dashboard_agent.render_board_table(limit=5))

    console.print("\n" + "-" * 70)
    console.print(f"[bold cyan]Pipeline Summary:[/bold cyan]")
    console.print(f" • Total Ingested & Triaged: [bold]{dashboard_agent.stats['total_processed']}[/bold]")
    console.print(f" • Corroborated Cross-Feed:  [bold]{dashboard_agent.stats['corroborated_count']}[/bold]")
    console.print(f" • Escalations Triggered:    [bold red]{dashboard_agent.stats['escalated_count']}[/bold red]")
    console.print(f" • Routine Review Items:     [bold green]{dashboard_agent.stats['routine_count']}[/bold green]")
    console.print("-" * 70)

    # Sanity checks
    assert dashboard_agent.stats["total_processed"] == 3, "Expected 3 processed items"
    assert dashboard_agent.stats["escalated_count"] >= 1, "Expected at least 1 escalated item"
    console.print("[bold green][PASSED] All pipeline stages verified successfully.[/bold green]\n")


if __name__ == "__main__":
    main()
