# Dataset Card for SentinelSentry Crisis Triage

## Dataset Summary
SentinelSentry is a curated dataset of emergency computer-aided dispatch (CAD) shorthand, emergency radio transcripts, and citizen eyewitness reports mapped into strict structured JSON.

Each example converts messy, noisy text into the following strict schema:
```json
{
  "incident_type": "string",
  "casualty_risk_level": 1-5,
  "credibility_score": 0.0-1.0,
  "location_text": "string"
}
```

## Dataset Composition
- **Training Set**: 900 samples derived from Disaster Response / CrisisLex subsets and realistic CAD radio shorthands.
- **Held-Out Test Set**: 100 synthetic dispatch-style lines held out strictly for evaluation benchmarks.

## Labeling Methodology & Open Disclosure
In accordance with open AI transparency:
1. **Incident Categories**: Mapped directly from existing Disaster Response and CrisisLex taxonomy into 8 operational categories:
   - `Structure Fire`
   - `Vehicle Accident / Rescue`
   - `Medical Emergency`
   - `Severe Weather / Flooding`
   - `Hazardous Materials`
   - `Civil Unrest / Active Threat`
   - `Infrastructure Failure / Gas Leak`
   - `False Alarm / Non-Emergency`
2. **Casualty Risk & Credibility**: Weakly labeled using a larger reasoning model (Gemini 1.5 Pro) with strict schema constraints and **spot-checked by human inspection** to ensure high-fidelity calibration across risk levels 1-5 and source-weighted credibility (0.0 to 1.0).
3. **Location Extraction**: Verified to represent realistic roadway intersections, highway mile markers, and civic landmarks (calibrated against Wake County, NC GIS benchmarks).

## Intended Use
Designed for fine-tuning compact LLMs (Gemma 2B) using QLoRA/PEFT to run real-time edge triage swarms for human dispatcher decision support. It is not intended for autonomous emergency dispatch without human-in-the-loop verification.
