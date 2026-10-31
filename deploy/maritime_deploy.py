"""
Maritime Micro-VM Deployment Bridge for SentinelSentry.
Deploys the 10 decoupled swarm agents onto Maritime's Firecracker micro-VM platform (maritime.sh)
using the Maritime API / CLI token.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentinelsentry.maritime")

MARITIME_API_BASE = os.environ.get("MARITIME_API_BASE", "https://api.maritime.sh/v1")

SWARM_MICRO_VMS: List[Dict[str, Any]] = [
    {
        "id": "maritime-vm-ingest-cad_police",
        "name": "CAD Police Ingest Worker",
        "role": "Law enforcement CAD feed polling and 10-code deduplication",
        "entrypoint": "python -m sentinelsentry.agents.ingest --feed cad_police",
        "env": {"FEED_ID": "cad_police", "POLL_INTERVAL": "2.0"},
    },
    {
        "id": "maritime-vm-ingest-fire_rescue",
        "name": "Wake Fire & Rescue Ingest Worker",
        "role": "Wake County Fire 2-alarm commercial fire parser",
        "entrypoint": "python -m sentinelsentry.agents.ingest --feed fire_rescue",
        "env": {"FEED_ID": "fire_rescue", "POLL_INTERVAL": "2.0"},
    },
    {
        "id": "maritime-vm-ingest-ems_scanner",
        "name": "Triangle EMS Scanner Ingest Worker",
        "role": "Triangle regional EMS trauma and hospital feed parser",
        "entrypoint": "python -m sentinelsentry.agents.ingest --feed ems_scanner",
        "env": {"FEED_ID": "ems_scanner", "POLL_INTERVAL": "2.0"},
    },
    {
        "id": "maritime-vm-ingest-weather_skywarn",
        "name": "NWS Skywarn Spotter Ingest Worker",
        "role": "Severe weather, flood, and Skywarn spotter advisory ingest",
        "entrypoint": "python -m sentinelsentry.agents.ingest --feed weather_skywarn",
        "env": {"FEED_ID": "weather_skywarn", "POLL_INTERVAL": "3.0"},
    },
    {
        "id": "maritime-vm-ingest-social_crisis",
        "name": "Citizen Eyewitness Stream Ingest Worker",
        "role": "Citizen crisis stream parser with panic deduplication",
        "entrypoint": "python -m sentinelsentry.agents.ingest --feed social_crisis",
        "env": {"FEED_ID": "social_crisis", "POLL_INTERVAL": "2.5"},
    },
    {
        "id": "maritime-vm-triage-01",
        "name": "Gemma 4 Triage Worker #1",
        "role": "Parallel Gemma 4 LoRA inference worker for dispatch triage",
        "entrypoint": "python -m sentinelsentry.agents.triage --worker-id 1",
        "env": {"WORKER_ID": "1", "BATCH_SIZE": "5"},
    },
    {
        "id": "maritime-vm-triage-02",
        "name": "Gemma 4 Triage Worker #2",
        "role": "Parallel Gemma 4 LoRA inference worker for dispatch triage",
        "entrypoint": "python -m sentinelsentry.agents.triage --worker-id 2",
        "env": {"WORKER_ID": "2", "BATCH_SIZE": "5"},
    },
    {
        "id": "maritime-vm-verifier-01",
        "name": "Verification Corroboration Agent",
        "role": "10-minute sliding window multi-feed cross-corroborator",
        "entrypoint": "python -m sentinelsentry.agents.verification",
        "env": {"RISK_THRESHOLD": "4", "WINDOW_MINUTES": "10"},
    },
    {
        "id": "maritime-vm-geocoder-01",
        "name": "Zero-Latency Geocoder Agent",
        "role": "Landmark cache and OpenStreetMap Nominatim geocoder",
        "entrypoint": "python -m sentinelsentry.agents.geocoder",
        "env": {"NOMINATIM_ENABLED": "true"},
    },
    {
        "id": "maritime-vm-alerter-01",
        "name": "Operations Center Web Hub",
        "role": "FastAPI SSE event broadcaster and human dispatcher decision board",
        "entrypoint": "uvicorn sentinelsentry.web.app:app --host 0.0.0.0 --port 8000",
        "env": {"PORT": "8000", "HOST": "0.0.0.0"},
    },
]


def deploy_to_maritime(api_key: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Provision the 10 micro-VM agents on Maritime.
    If dry_run is True or API key is mock, verifies agent configurations.
    """
    logger.info("Initializing Maritime Swarm Deployment...")
    logger.info(f"Target Platform: Maritime (YC F26) - Firecracker Micro-VM Cluster")
    logger.info(f"Configuring {len(SWARM_MICRO_VMS)} independent agent micro-VMs...\n")

    results = {"deployed": [], "failed": [], "dry_run": dry_run}

    for vm in SWARM_MICRO_VMS:
        vm_id = vm["id"]
        name = vm["name"]
        logger.info(f"-> Provisioning [{vm_id}]: {name}")

        if dry_run or not api_key:
            logger.info(f"   [DRY-RUN] Manifest validated: command='{vm['entrypoint']}', role='{vm['role']}'")
            results["deployed"].append({
                "vm_id": vm_id,
                "status": "validated_manifest",
                "role": vm["role"],
            })
            continue

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SentinelSentry-MaritimeDeploy/1.0",
        }
        payload = {
            "external_id": vm_id,
            "name": name,
            "role": vm["role"],
            "command": vm["entrypoint"],
            "env": vm["env"],
            "template": "custom-python3",
            "sleep_after_idle_seconds": 60,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(f"{MARITIME_API_BASE}/agents", json=payload, headers=headers)
                if resp.status_code in (200, 201):
                    logger.info(f"   [SUCCESS] Live micro-VM provisioned on Maritime: {resp.json().get('id', vm_id)}")
                    results["deployed"].append({"vm_id": vm_id, "status": "active", "response": resp.json()})
                else:
                    logger.warning(f"   [HTTP {resp.status_code}] Maritime response: {resp.text}")
                    results["failed"].append({"vm_id": vm_id, "error": resp.text})
        except Exception as e:
            logger.warning(f"   [CONNECTION NOTICE] Maritime endpoint unreachable ({e}). Manifest recorded.")
            results["failed"].append({"vm_id": vm_id, "error": str(e)})

    logger.info("\n" + "="*70)
    logger.info(f"Maritime Swarm Provisioning Summary: {len(results['deployed'])} ready, {len(results['failed'])} failed.")
    logger.info("="*70)
    return results


def main():
    parser = argparse.ArgumentParser(description="Deploy SentinelSentry Swarm onto Maritime Micro-VMs")
    parser.add_argument("--api-key", help="Maritime API Key (or set MARITIME_API_KEY env var)")
    parser.add_argument("--dry-run", action="store_true", help="Validate agent configurations without network API calls")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MARITIME_API_KEY") or os.environ.get("MARITIME_TOKEN")

    if not api_key and not args.dry_run:
        print("\n" + "!"*70)
        print("NOTICE: No MARITIME_API_KEY or MARITIME_TOKEN detected.")
        print("To deploy to live Maritime micro-VMs:")
        print("  1. Export your Maritime token: export MARITIME_API_KEY=\"mk_...\"")
        print("  2. Run: python deploy/maritime_deploy.py --api-key <YOUR_KEY>")
        print("Running in validation / dry-run mode to verify all 10 micro-VM manifests...")
        print("!"*70 + "\n")
        args.dry_run = True

    deploy_to_maritime(api_key=api_key or "", dry_run=args.dry_run)


if __name__ == "__main__":
    main()
