"""
SentinelSentry Live Multi-Agent Swarm Demo.
Launches the real-time Emergency Dispatcher Operations Center (FastAPI + SSE + Leaflet.js)
powered by a Gemma 4 LoRA adapter swarm running inside Maritime micro-VMs.

Usage:
    python run_demo.py               # Starts web dashboard & automatically opens browser
    python run_demo.py --cli         # Headless terminal CLI simulation with Rich tables
    python run_demo.py --no-browser  # Starts web server without auto-opening browser
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from rich.console import Console
from rich.panel import Panel

# Ensure utf-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sentinelsentry.config import settings
from sentinelsentry.schema import RawFeedItem
from sentinelsentry.feeds.generator import RSSFeedGenerator
from sentinelsentry.agents.base import SwarmEventBus, SwarmCoordinator
from sentinelsentry.agents.ingest import IngestAgent
from sentinelsentry.agents.triage import TriageAgent, LocalTriageEngine
from sentinelsentry.agents.verification import VerificationAgent
from sentinelsentry.agents.geocoder import GeocoderAgent
from sentinelsentry.agents.dashboard import AlertDashboardAgent


def run_cli_demo():
    """Headless CLI mode running discrete simulation with Rich terminal tables."""
    console = Console(force_terminal=True, legacy_windows=False)
    console.print(
        Panel.fit(
            "[bold white on blue] SentinelSentry: LoRA-Tuned Gemma 4 for Crisis Triage [/bold white on blue]\n"
            "[dim]A lightweight Gemma 4 adapter turns messy dispatch text into strict JSON,\n"
            "and a swarm of Maritime micro-VM agents routes, verifies, and escalates incidents.[/dim]",
            title="[bold yellow]STARTING SENTINELSENTRY SWARM[/bold yellow]",
        )
    )

    bus = SwarmEventBus()
    coordinator = SwarmCoordinator(bus=bus)

    # 1. Initialize Simulated Feed Generators
    feed_gens = {}
    for cfg in settings.FEEDS:
        gen = RSSFeedGenerator(
            feed_id=cfg.id,
            feed_name=cfg.name,
            description=cfg.description,
        )
        gen.populate_from_scenarios()
        feed_gens[cfg.id] = gen

        # Register Ingest Agent for each feed
        ingest_agent = IngestAgent(
            feed_config=cfg,
            bus=bus,
            custom_fetcher=lambda g=gen: g.to_raw_feed_items(),
        )
        coordinator.register_ingest_agent(ingest_agent)

    # 2. Register Triage, Verification, Geocoder, and Dashboard Agents
    triage_worker1 = TriageAgent(bus=bus, worker_id=1, engine=LocalTriageEngine())
    triage_worker2 = TriageAgent(bus=bus, worker_id=2, engine=LocalTriageEngine())
    adapter_info = triage_worker1.engine.get_adapter_info()

    verification_agent = VerificationAgent(bus=bus)
    geocoder_agent = GeocoderAgent(bus=bus, enable_network=False)
    dashboard_agent = AlertDashboardAgent(bus=bus, console=console)

    coordinator.register_triage_worker(triage_worker1)
    coordinator.register_triage_worker(triage_worker2)
    coordinator.set_verification_agent(verification_agent)
    coordinator.set_geocoder_agent(geocoder_agent)
    coordinator.set_dashboard_agent(dashboard_agent)

    console.print("\n[bold green][+] Swarm initialized: 5 Ingest Agents, 2 Triage Workers, 1 Verification Agent, 1 Geocoder, 1 Live Board[/bold green]")
    console.print(f"[dim]Triage Engine: {adapter_info['mode']} | Base: {adapter_info['base_model']} (LoRA r={adapter_info['lora_r']}, alpha={adapter_info['lora_alpha']})[/dim]")
    if adapter_info["has_safetensors"]:
        console.print(f"[dim]Adapter Weights: adapter_model.safetensors ({adapter_info['safetensors_size_bytes']:,} bytes)[/dim]")
    console.print("[dim]Simulating parallel feed ingestion and multi-stage triage inside Maritime micro-VMs...[/dim]\n")

    # 3. Step through initial feed discovery
    coordinator.drain(max_rounds=5)

    console.print(dashboard_agent.render_board_table(limit=8))

    # 4. Inject a live dynamic crisis burst (Scenario: 2-Alarm Warehouse Fire)
    console.print("\n" + "=" * 80)
    console.print("[bold red][!] TRIGGERING LIVE CRISIS BURST: SCN-001 Warehouse Fire with Trapped Occupant[/bold red]")
    console.print("=" * 80)

    # Injecting into Police CAD and Social Eyewitness simultaneously
    guid1 = feed_gens["cad_police"].add_item(
        raw_text="10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF EVAC IN PROGRESS 1 SUBJ INSIDE",
        title="CAD 911: 10-70 Saunders",
        scenario_id="SCN-001",
    )
    guid2 = feed_gens["social_crisis"].add_item(
        raw_text="Holy smoke huge explosions coming from warehouse district on Saunders St downtown Raleigh sirens everywhere!!",
        title="Citizen Report: Saunders Explosion",
        scenario_id="SCN-001",
    )

    console.print(f"[cyan]-> Injected item into 'cad_police' feed: {guid1}[/cyan]")
    console.print(f"[cyan]-> Injected item into 'social_crisis' feed: {guid2}[/cyan]")

    # Run swarm cycle to process burst
    coordinator.drain(max_rounds=4)

    console.print("\n[bold yellow][UPDATED DISPATCHER BOARD AFTER TRIAGE & CORROBORATION][/bold yellow]")
    console.print(dashboard_agent.render_board_table(limit=10))

    console.print("\n" + "-" * 80)
    console.print("[bold cyan]Swarm Operational Metrics:[/bold cyan]")
    console.print(f" • Total Ingested & Triaged: [bold]{dashboard_agent.stats['total_processed']}[/bold]")
    console.print(f" • Corroborated Cross-Feed:  [bold]{dashboard_agent.stats['corroborated_count']}[/bold]")
    console.print(f" • Escalations Fired:        [bold red]{dashboard_agent.stats['escalated_count']}[/bold red]")
    console.print(f" • Routine Review Items:     [bold green]{dashboard_agent.stats['routine_count']}[/bold green]")
    console.print("-" * 80)
    console.print("[bold green][OK] SentinelSentry triage swarm demonstrated successfully.[/bold green]\n")


def run_web_demo(host: str = "127.0.0.1", port: int = 8000, auto_open: bool = True):
    """One-command full web operations center launcher."""
    import uvicorn

    console = Console(force_terminal=True, legacy_windows=False)
    url = f"http://{host}:{port}"

    console.print(
        Panel.fit(
            f"[bold white on red] SentinelSentry: Live Dispatcher Operations Center [/bold white on red]\n\n"
            f"[bold yellow]Model:[/bold yellow]        Google Gemma 4 (`{settings.BASE_MODEL_NAME}`) + LoRA Adapter\n"
            f"[bold yellow]Architecture:[/bold yellow] 10 Maritime Micro-VM Swarm Workers (Ingest, Triage, Verify, Geocode, Alert)\n"
            f"[bold yellow]Web Hub:[/bold yellow]      [link={url}][bold cyan]{url}[/bold cyan][/link]\n"
            f"[bold yellow]Features:[/bold yellow]     Leaflet Incident Map • SSE Live Updates • Escalation Banner • Multimodal Vision",
            title="[bold green]SENTINELSENTRY EMERGENCY OPERATIONS DEMO[/bold green]",
        )
    )

    if auto_open:
        def open_browser():
            time.sleep(1.2)
            try:
                console.print(f"[bold green][*] Automatically launching browser to {url}...[/bold green]")
                webbrowser.open(url)
            except Exception as e:
                console.print(f"[yellow]Could not open browser automatically: {e}. Please visit {url} directly.[/yellow]")

        thread = threading.Thread(target=open_browser, daemon=True)
        thread.start()

    console.print(f"[dim]Starting Uvicorn web server on {host}:{port}... Press Ctrl+C to terminate.[/dim]\n")
    uvicorn.run("sentinelsentry.web.app:app", host=host, port=port, log_level="info", reload=False)


def main():
    parser = argparse.ArgumentParser(description="Launch the SentinelSentry Emergency Operations Swarm Demo.")
    parser.add_argument("--cli", action="store_true", help="Run in headless terminal CLI mode with Rich table.")
    parser.add_argument("--host", type=str, default=settings.WEB_HOST, help="Web dashboard host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=settings.WEB_PORT, help="Web dashboard port (default: 8000).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically on startup.")
    args = parser.parse_args()

    if args.cli:
        run_cli_demo()
    else:
        run_web_demo(host=args.host, port=args.port, auto_open=not args.no_browser)


if __name__ == "__main__":
    main()
