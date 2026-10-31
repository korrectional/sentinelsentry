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
        self.model_path = model_path
        self._hf_pipeline = None
        self._is_hf_ready = False
        if model_path:
            self._try_init_hf_model(model_path)

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
        if any(w in text_lower for w in ["fire", "flames", "smoke", "10-70", "alarm", "burn"]):
            if "alarm" in text_lower and any(w in text_lower for w in ["waterflow", "accidental", "false", "test", "maintenance"]):
                inc_type = IncidentType.NON_EMERGENCY.value
            else:
                inc_type = IncidentType.STRUCTURE_FIRE.value
        elif any(w in text_lower for w in ["10-50", "accident", "crash", "rollover", "extrication", "mva", "collision", "pinned"]):
            inc_type = IncidentType.VEHICLE_ACCIDENT.value
        elif any(w in text_lower for w in ["flood", "water rising", "skywarn", "swiftwater", "submerged", "creek overflowing"]):
            inc_type = IncidentType.SEVERE_WEATHER.value
        elif any(w in text_lower for w in ["gas leak", "hazmat", "10-75", "lel meter", "pipeline", "odor of gas"]):
            inc_type = IncidentType.GAS_LEAK.value
        elif any(w in text_lower for w in ["cpr", "cardiac", "echo response", "unresponsive", "10-54", "medical"]):
            inc_type = IncidentType.MEDICAL_EMERGENCY.value
        elif any(w in text_lower for w in ["shots", "gunshots", "10-33", "active shooter", "weapon"]):
            inc_type = IncidentType.CIVIL_UNREST.value

        # 2. Estimate Casualty Risk (1-5)
        risk = 2
        if inc_type == IncidentType.NON_EMERGENCY.value or "pd only" in text_lower or "property damage only" in text_lower:
            risk = 1
        elif any(w in text_lower for w in ["cpr in progress", "unaccounted for", "red tag", "ped trapped", "jaws in service", "trapped", "collapse zone", "multi-vehicle rollover", "swiftwater", "lel meters reading 70%"]):
            risk = 5 if any(w in text_lower for w in ["cpr", "trapped", "red tag", "unaccounted"]) else 4
        elif any(w in text_lower for w in ["injuries", "medic en route", "code 3", "2-alarm"]):
            risk = 4
        elif any(w in text_lower for w in ["standstill", "odor", "popping"]):
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
        patterns = [
            r"(\d{2,5}\s+[A-Za-z0-9\.\s]+(?:St|Ave|Rd|Blvd|Dr|Lane|Way|Pkwy|Drive|Street|Road))",
            r"(I-\d{1,3}\s+(?:EB|WB|NB|SB|Eastbound|Westbound)?[^,\.\n]+)",
            r"([A-Za-z\s]+(?:Ave|St|Rd|Blvd)\s+(?:&|and|nr|near|at)\s+[A-Za-z\s]+(?:Ave|St|Rd|Blvd|Dr))",
            r"(Crabtree Valley Mall[^,\.\n]*)",
            r"(Crabtree Creek[^,\.\n]*)",
            r"(Glenwood Ave & Peace St[^,\.\n]*)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                loc = m.group(1).strip()
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
