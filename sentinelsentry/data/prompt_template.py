"""
Prompt formatting templates for Gemma 2B instruction tuning and triage inference.
"""

SYSTEM_INSTRUCTION = """You are SentinelSentry, an expert emergency dispatch AI triage model.
Convert messy, abbreviated emergency dispatch reports and citizen eyewitness text into a single strict JSON object.

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
"""


def format_gemma_prompt(raw_text: str) -> str:
    """Format prompt for Gemma 2B chat model."""
    return (
        f"<start_of_turn>user\n"
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Input dispatch text:\n"
        f"{raw_text.strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def format_training_example(raw_text: str, target_json: str) -> str:
    """Format full prompt + response for causal language model training."""
    prompt = format_gemma_prompt(raw_text)
    return f"{prompt}{target_json.strip()}<end_of_turn>"
