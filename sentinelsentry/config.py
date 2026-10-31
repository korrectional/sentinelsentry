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
            pkg_data = self.BASE_DIR / "sentinelsentry" / "data"
            local_data = self.BASE_DIR / "data"
            self.DATA_DIR = pkg_data if pkg_data.exists() else (local_data if local_data.exists() else pkg_data)
        if self.FEEDS_DIR is None:
            pkg_feeds = self.BASE_DIR / "sentinelsentry" / "feeds"
            local_feeds = self.BASE_DIR / "feeds"
            self.FEEDS_DIR = pkg_feeds if pkg_feeds.exists() else local_feeds
        if self.TESTS_DIR is None:
            self.TESTS_DIR = self.BASE_DIR / "tests"
        return self

    # Model & LoRA settings (Gemma 4 Smallest Variant)
    BASE_MODEL_NAME: str = "google/gemma-4-e2b-it"
    LORA_TARGET_MODULES: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    ADAPTER_OUTPUT_DIR: str = "adapter_model"
    _LOCAL_ADAPTER: Path = BASE_DIR / "adapter_model"
    _TEMP_ADAPTER: Path = Path(os.environ.get("TEMP", "/tmp")) / "sentinelsentry_weights"
    ADAPTER_DIR: Path = _LOCAL_ADAPTER if (_LOCAL_ADAPTER / "adapter_model.safetensors").exists() else (_TEMP_ADAPTER if (_TEMP_ADAPTER / "adapter_model.safetensors").exists() else _LOCAL_ADAPTER)
    HF_REPO_ID: str = "sentinelsentry/gemma-4-crisis-triage"

    # Agent thresholds
    HIGH_RISK_THRESHOLD: int = 4  # Triggers Verification Agent when casualty_risk_level >= 4
    ESCALATION_MIN_CREDIBILITY: float = 0.6  # Auto-escalates for human dispatcher if risk >= 4 and credibility >= 0.6
    CORROBORATION_TIME_WINDOW_SECONDS: float = 600  # 10 minutes cross-feed window
    NOMINATIM_USER_AGENT: str = "sentinelsentry-crisis-dispatch/0.1"
    SIMULATED_FEED_SERVER_PORT: int = 8765
    SIMULATED_FEED_SERVER_HOST: str = "127.0.0.1"

    # Web Dashboard Server
    WEB_DASHBOARD_HOST: str = "127.0.0.1"
    WEB_DASHBOARD_PORT: int = 8000
    WEB_HOST: str = "127.0.0.1"
    WEB_PORT: int = 8000

    # Map & Tile Settings (Free DarkMatter public tiles by default; CARTO API Key optional)
    CARTO_API_KEY: Optional[str] = os.environ.get("CARTO_API_KEY")
    MAP_TILE_URL: str = os.environ.get(
        "MAP_TILE_URL",
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
    )

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
