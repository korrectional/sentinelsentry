"""
Configuration settings for SentinelSentry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, model_validator


class FeedConfig(BaseModel):
    id: str
    name: str
    description: str
    poll_interval_seconds: float = 3.0
    is_simulated: bool = True
    url: str = ""
    weight_bias_credibility: float = 0.8  # official CAD has higher initial credibility than social


class Settings(BaseModel):
    # Project paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Optional[Path] = None
    FEEDS_DIR: Optional[Path] = None
    TESTS_DIR: Optional[Path] = None

    @model_validator(mode='after')
    def _resolve_dynamic_paths(self) -> Settings:
        """Resolve derived paths at instance creation time (not class definition time)."""
        if self.DATA_DIR is None:
            local_data = self.BASE_DIR / "data"
            temp_data = Path(os.environ.get("TEMP", "/tmp")) / "sentinelsentry_data"
            self.DATA_DIR = local_data if local_data.exists() else temp_data
        if self.FEEDS_DIR is None:
            self.FEEDS_DIR = self.BASE_DIR / "feeds"
        if self.TESTS_DIR is None:
            self.TESTS_DIR = self.BASE_DIR / "tests"
        return self

    # Model & LoRA settings
    BASE_MODEL_NAME: str = "google/gemma-2b-it"
    LORA_TARGET_MODULES: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    ADAPTER_OUTPUT_DIR: str = "adapter_model"
    HF_REPO_ID: str = "sentinelsentry/gemma-2b-crisis-triage"

    # Agent thresholds
    HIGH_RISK_THRESHOLD: int = 4  # Triggers Verification Agent when casualty_risk_level >= 4
    ESCALATION_MIN_CREDIBILITY: float = 0.6  # Auto-escalates for human dispatcher if risk >= 4 and credibility >= 0.6
    CORROBORATION_TIME_WINDOW_SECONDS: float = 600  # 10 minutes cross-feed window
    NOMINATIM_USER_AGENT: str = "sentinelsentry-crisis-dispatch/0.1"
    SIMULATED_FEED_SERVER_PORT: int = 8765
    SIMULATED_FEED_SERVER_HOST: str = "127.0.0.1"

    # Default simulated feeds
    FEEDS: List[FeedConfig] = [
        FeedConfig(
            id="cad_police",
            name="Raleigh Metro 911 / Police CAD",
            description="Law enforcement computer-aided dispatch lines with 10-codes and shorthand",
            poll_interval_seconds=2.0,
            url="http://127.0.0.1:8765/feeds/cad-police.xml",
            weight_bias_credibility=0.90,
        ),
        FeedConfig(
            id="fire_rescue",
            name="Wake County Fire & Rescue Dispatch",
            description="Fire operations, structure alarms, vehicle extrications, and hazmat telemetry",
            poll_interval_seconds=2.5,
            url="http://127.0.0.1:8765/feeds/fire-rescue.xml",
            weight_bias_credibility=0.95,
        ),
        FeedConfig(
            id="ems_scanner",
            name="Triangle Regional EMS Scanner Transcript",
            description="Emergency medical service radio transcripts, triage tags, and trauma alerts",
            poll_interval_seconds=2.0,
            url="http://127.0.0.1:8765/feeds/ems-scanner.xml",
            weight_bias_credibility=0.90,
        ),
        FeedConfig(
            id="weather_skywarn",
            name="NWS Skywarn Spotter Network",
            description="Severe convective storm, flash flooding, and downed infrastructure reports",
            poll_interval_seconds=3.0,
            url="http://127.0.0.1:8765/feeds/weather-skywarn.xml",
            weight_bias_credibility=0.85,
        ),
        FeedConfig(
            id="social_crisis",
            name="Citizen Eyewitness Crisis Stream (X/Reddit)",
            description="Unfiltered public social reports with high noise, varied phrasing, and rapid eyewitness bursts",
            poll_interval_seconds=1.5,
            url="http://127.0.0.1:8765/feeds/social-crisis.xml",
            weight_bias_credibility=0.45,
        ),
    ]


settings = Settings()
