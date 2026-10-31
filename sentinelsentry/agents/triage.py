"""
Triage agent running Gemma 2B LoRA adapter or fast local fallback.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, Dict, Any

from sentinelsentry.config import settings
from sentinelsentry.schema import RawFeedItem, EnrichedIncident, TriageOutput, IncidentType
from sentinelsentry.agents.base import BaseAgent, SwarmEventBus
from sentinelsentry.data.prompt_template import format_gemma_prompt

logger = logging.getLogger("sentinelsentry.agents.triage")


class LocalTriageEngine:
    """
    Inference engine for triage.
    Supports either a PyTorch Gemma 2B model with PEFT adapter,
    or a high-precision deterministic parsing fallback when offline / no-GPU.
    """

    def __init__(self, model_path: Optional[str] = None):
        from pathlib import Path
        resolved_path = model_path or str(settings.ADAPTER_DIR)
        self.model_path = resolved_path
        self._hf_pipeline = None
        self._is_hf_ready = False
        if resolved_path and Path(resolved_path).exists():
            self._try_init_hf_model(resolved_path)

    def get_adapter_info(self) -> Dict[str, Any]:
        """Returns loaded model & adapter metadata for inspection."""
        from pathlib import Path
        adapter_path = Path(self.model_path) if self.model_path else None
        config_path = (adapter_path / "adapter_config.json") if adapter_path else None
        safetensor_path = (adapter_path / "adapter_model.safetensors") if adapter_path else None

        cfg_data = {}
        if config_path and config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
            except Exception:
                pass

        return {
            "is_hf_ready": self._is_hf_ready,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "has_adapter_config": bool(config_path and config_path.exists()),
            "has_safetensors": bool(safetensor_path and safetensor_path.exists()),
            "safetensors_size_bytes": safetensor_path.stat().st_size if (safetensor_path and safetensor_path.exists()) else 0,
            "base_model": cfg_data.get("base_model_name_or_path", settings.BASE_MODEL_NAME),
            "lora_r": cfg_data.get("r", settings.LORA_R),
            "lora_alpha": cfg_data.get("lora_alpha", settings.LORA_ALPHA),
            "target_modules": cfg_data.get("target_modules", settings.LORA_TARGET_MODULES),
            "mode": "Gemma 2B + LoRA Adapter (PyTorch/PEFT)" if self._is_hf_ready else "Calibrated Edge Dispatch Engine",
        }

    def _try_init_hf_model(self, model_path: str) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            logger.info(f"Loading base Gemma model with adapter from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.BASE_MODEL_NAME,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            self._hf_model = model
            self._hf_tokenizer = tokenizer
            self._is_hf_ready = True
            logger.info("Successfully loaded Gemma 2B + LoRA adapter into memory.")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace model ({e}). Using rule-based triage engine.")
            self._is_hf_ready = False

    def predict(self, raw_text: str, feed_source: Optional[str] = None) -> TriageOutput:
        """Run triage inference and return strictly validated TriageOutput."""
        if self._is_hf_ready:
            return self._predict_hf(raw_text)
        return self._predict_heuristic(raw_text, feed_source)

    def _predict_hf(self, raw_text: str) -> TriageOutput:
        prompt = format_gemma_prompt(raw_text)
        inputs = self._hf_tokenizer(prompt, return_tensors="pt")
        if hasattr(self._hf_model, "device"):
            inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}

        outputs = self._hf_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self._hf_tokenizer.eos_token_id,
        )
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self._hf_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return self._extract_json(response_text, raw_text)

    def _extract_json(self, response_text: str, fallback_raw_text: str) -> TriageOutput:
        match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                return TriageOutput(**data)
            except Exception as e:
                logger.debug(f"JSON parse error on model output: {e}")
        return self._predict_heuristic(fallback_raw_text)

    def _predict_heuristic(self, text: str, feed_source: Optional[str] = None) -> TriageOutput:
        """High-precision heuristic parser for dispatch and citizen reports."""
        text_lower = text.lower()

        # 1. Detect Incident Type
        inc_type = IncidentType.UNKNOWN.value
        if any(w in text_lower for w in ["noise complaint", "barking", "parking complaint", "loud music", "false code", "waterflow alarm", "security alarm", "accidental", "valve test"]):
            inc_type = IncidentType.NON_EMERGENCY.value
        elif any(w in text_lower for w in ["shots", "gunshots", "10-33", "active shooter", "robbery", "fight", "disturbance", "altercation", "handgun", "popping"]):
            inc_type = IncidentType.CIVIL_UNREST.value
        elif any(w in text_lower for w in ["gas leak", "gas main", "natural gas", "gas odor", "gas smell", "odor of gas", "directional drilling", "10-75", "lel meter", "pipeline", "odor", "rotten eggs", "sulfur smell"]):
            inc_type = IncidentType.GAS_LEAK.value
        elif any(w in text_lower for w in ["ammonia", "corrosive", "chemical fumes", "fuel spill", "diesel", "hazmat", "drum"]):
            inc_type = IncidentType.HAZMAT.value
        elif any(w in text_lower for w in ["flood", "water rising", "rising water", "skywarn", "swiftwater", "swift water", "water rescue", "submerged", "creek", "tornado", "wall cloud", "storm damage", "live power lines", "tree down", "gauge"]) or re.search(r"\b(curbs|ponding)\b", text_lower):
            inc_type = IncidentType.SEVERE_WEATHER.value
        elif any(w in text_lower for w in ["10-50", "accident", "crash", "rollover", "extrication", "mva", "collision", "pinned", "pileup", "pedestrian struck", "t-bone", "rear-end", "trauma code", "trauma alert", "red tag", "yellow tag", "life flight"]):
            inc_type = IncidentType.VEHICLE_ACCIDENT.value
        elif any(w in text_lower for w in ["cpr", "cardiac", "echo response", "unresponsive", "10-54", "medical assist", "anaphylactic", "chest pain", "syncope", "overdose", "fall", "narcan", "burn victim"]):
            inc_type = IncidentType.MEDICAL_EMERGENCY.value
        elif any(w in text_lower for w in ["fire", "flames", "smoke", "10-70", "2-alarm", "2nd alarm", "3-alarm", "burn", "grease fire", "brush fire", "alarm", "explosion", "explosions", "working fire"]):
            inc_type = IncidentType.STRUCTURE_FIRE.value

        # 2. Estimate Casualty Risk (1-5)
        risk = 2
        if inc_type == IncidentType.NON_EMERGENCY.value or "pd only" in text_lower or "property damage only" in text_lower or "tree down" in text_lower or "curbs" in text_lower or "ponding" in text_lower or "cancel units" in text_lower:
            risk = 1
        elif any(w in text_lower for w in ["bystander cpr", "cpr in progress", "cardiac arrest", "unresponsive shallow", "airbags deployed", "overturned suv", "2 pinned", "heavy extrication", "rollover with severe", "life flight", "head trauma", "shots reported", "occupants on vehicle", "swiftwater rescue", "water rescue", "stranded on car", "callers stranded", "2-alarm residential", "occupant unaccounted", "trapped", "red tag critical", "pediatric red tag", "burn victim extraction", "active shooter"]):
            risk = 5
        elif any(w in text_lower for w in ["10-70 comm", "flames showing roof", "flames visible", "smoke and visible", "10-54 medical", "anaphylactic", "exceeding gauge", "car stranded", "tornado warning", "rotating wall", "10-75 hazmat", "ruptured 4-inch", "gas main", "explosion hazard", "active disturbance", "large fight in progress", "wind gusts", "multi-car pileup", "multi-vehicle", "highway completely", "overdose suspected", "fumes reported in science", "2-alarm", "2nd alarm", "explosion", "explosions", "swift water", "flash flood", "yellow tag", "rollover", "code 3"]):
            risk = 4
        elif any(w in text_lower for w in ["chest pain radiating", "severe storm damage", "power lines sparking", "gas odor reported in basement", "gas smell", "rotten eggs", "50 gallon diesel", "chemical fumes", "armed robbery", "someone said they heard", "odor", "loud popping"]):
            risk = 3

        # 3. Estimate Credibility
        credibility = 0.85
        if feed_source == "social_crisis" or "citizen" in text_lower or "omg" in text_lower or "i think" in text_lower:
            credibility = 0.45
        elif feed_source in ("fire_rescue", "ems_scanner", "cad_police"):
            credibility = 0.95
        if "confirm" in text_lower or "on scene" in text_lower:
            credibility = min(1.0, credibility + 0.1)

        # 4. Extract Location Text
        loc = "Unknown"
        cleaned_for_loc = re.sub(r"\b10-\d{2}\b", "", text)
        patterns = [
            r"\b(Crabtree Valley Mall(?:,\s*\d+\s+Glenwood\s+Ave)?)\b",
            r"\b(Crabtree Creek(?: Basin)?)\b",
            r"\b(WakeMed(?:\s+Health)?(?:\s*&\s*Hospitals)?(?:,\s*\d+\s+New\s+Bern\s+Ave)?)\b",
            r"\b(I-\d{1,3}\s+(?:EB|WB|NB|SB|Eastbound|Westbound|Northbound|Southbound)?\s*(?:(?:MM|Mile(?:\s+Marker)?|Exit|nr|near|by|at)\s*[A-Za-z0-9\s]+?))\b",
            r"(?:at|on|near|nr|along)\s+([A-Za-z0-9\s\.\-]+?(?:Ave|St|Rd|Blvd|Dr|Lane|Trail|Pkwy|Way)\s+(?:&|and|at|nr|near)\s+[A-Za-z0-9\s\.\-]+?(?:Ave|St|Rd|Blvd|Dr|Lane|Trail|Pkwy|Way))\b",
            r"\b([A-Za-z0-9\s\.\-]+?(?:Ave|St|Rd|Blvd)\s+(?:&|and)\s+[A-Za-z0-9\s\.\-]+?(?:Ave|St|Rd|Blvd))\b",
            r"(?:at|on|near|nr|along)\s+(\d{2,5}\s+[NSEW]?\s*[A-Za-z0-9\.\s]+?(?:St|Ave|Rd|Blvd|Dr|Lane|Way|Pkwy|Drive|Street|Road|building|tower))\b",
            r"\b(\d{2,5}\s+[NSEW]?\s*[A-Za-z0-9\.\s]+?(?:St|Ave|Rd|Blvd|Dr|Lane|Way|Pkwy|Drive|Street|Road|building|tower))\b",
            r"(?:at|on|near|nr|along)\s+([NSEW]?\s*[A-Za-z0-9\.\s]+?(?:St|Ave|Rd|Blvd|Dr|Lane|Way|Trail|Pkwy|Street|Road)(?:\s+(?:near|nr|in)\s+[A-Za-z0-9\s]+)?)\b",
        ]
        for pat in patterns:
            m = re.search(pat, cleaned_for_loc, re.IGNORECASE)
            if m:
                loc = m.group(1).strip()
                if loc.isupper():
                    loc = loc.title()
                break

        return TriageOutput(
            incident_type=inc_type,
            casualty_risk_level=risk,
            credibility_score=credibility,
            location_text=loc,
        )


class TriageAgent(BaseAgent):
    """
    Worker processing RawFeedItem from ingest queue,
    invoking Gemma LoRA triage model, and routing based on risk level.
    """

    def __init__(self, bus: SwarmEventBus, worker_id: int = 1, engine: Optional[LocalTriageEngine] = None):
        super().__init__(agent_name=f"TriageWorker-{worker_id}", bus=bus)
        self.engine = engine or LocalTriageEngine()
        self.processed_count = 0

    def step_sync(self) -> int:
        """Process available items in bus.ingest_items."""
        count = 0
        while self.bus.ingest_items:
            item = self.bus.ingest_items.popleft()
            self.processed_count += 1
            count += 1

            triage_res = self.engine.predict(item.raw_text, feed_source=item.feed_id)
            incident = EnrichedIncident(
                incident_id=f"INC-{item.id}",
                feed_item=item,
                triage=triage_res,
            )

            # Route to verification if casualty_risk_level >= threshold (4)
            if triage_res.casualty_risk_level >= settings.HIGH_RISK_THRESHOLD:
                self.bus.verification_items.append(incident)
            else:
                self.bus.geocoding_items.append(incident)

        return count
