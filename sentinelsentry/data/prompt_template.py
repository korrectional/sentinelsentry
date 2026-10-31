"""
Prompt formatting templates for Gemma 4 instruction tuning and multimodal triage inference.
"""

from typing import Optional

SYSTEM_INSTRUCTION = """You are SentinelSentry, an expert emergency dispatch AI triage model powered by Gemma 4.
Convert messy, abbreviated emergency dispatch reports, citizen eyewitness text, and optional scene imagery into a single strict JSON object.

Output JSON Schema:
{
  "incident_type": "string",
  "casualty_risk_level": 1-5,
  "credibility_score": 0.0-1.0,
  "location_text": "string"
}

Standard incident categories:
- Structure Fire
- Vehicle Accident / Rescue
- Medical Emergency
- Severe Weather / Flooding
- Hazardous Materials
- Civil Unrest / Active Threat
- Infrastructure Failure / Gas Leak
- False Alarm / Non-Emergency

Rules:
1. Output ONLY the JSON object. No preamble, no explanation, no markdown backticks.
2. casualty_risk_level must be an integer between 1 and 5.
3. credibility_score must be a float between 0.0 and 1.0.
4. location_text must be the exact street address, intersection, or landmark, or 'Unknown' if none is found.
5. If an image is provided, integrate visual damage indicators (smoke, water levels, structural collapse, injuries) to refine risk and credibility.
"""


def format_gemma_prompt(raw_text: str, has_image: bool = False) -> str:
    """Format prompt for Gemma 4 chat and multimodal models."""
    img_context = "\n[Attached Scene Image Included]" if has_image else ""
    return (
        f"<start_of_turn>user\n"
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Input dispatch text:{img_context}\n"
        f"{raw_text.strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def format_training_example(raw_text: str, target_json: str, has_image: bool = False) -> str:
    """Format full prompt + response for causal language model training."""
    prompt = format_gemma_prompt(raw_text, has_image=has_image)
    return f"{prompt}{target_json.strip()}<end_of_turn>"
