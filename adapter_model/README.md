---
license: gemma
base_model: google/gemma-4-e2b-it
tags:
- peft
- lora
- gemma-4
- crisis-response
- emergency-dispatch
- 911-triage
- multi-agent
- multimodal
pipeline_tag: text-generation
library_name: peft
datasets:
- sentinelsentry/crisis-lex-dispatch-subset
---

# SentinelSentry: LoRA-Tuned Gemma 4 for Emergency Dispatch Triage

<div align="center">

[![License: Gemma](https://img.shields.io/badge/License-Gemma-blue.svg)](https://ai.google.dev/gemma/terms)
[![PEFT: LoRA](https://img.shields.io/badge/PEFT-LoRA_r16_a32-brightgreen.svg)](https://github.com/huggingface/peft)
[![Base: Gemma 4](https://img.shields.io/badge/Base_Model-google%2Fgemma--4--e2b--it-orange.svg)](https://huggingface.co/google/gemma-4-e2b-it)
[![Architecture: Maritime Swarm](https://img.shields.io/badge/Architecture-Maritime_Micro--VM_Swarm-purple.svg)](https://github.com/sentinelsentry/sentinelsentry)

</div>

## Model Description

**SentinelSentry** is a compact, high-precision QLoRA/PEFT adapter for **Gemma 4** (`google/gemma-4-e2b-it`). It is engineered to transform noisy, unstructured, multi-agency emergency streams (911 CAD radio shorthands, fire telemetry, EMS scanner transcripts, and citizen eyewitness reports with optional incident photos) into strictly validated JSON crisis triage records for a human dispatcher decision-support swarm running in Maritime micro-VMs.

### One-Liner
> A small Gemma 4 adapter turns messy dispatch text into strict JSON, and a swarm of concurrent agents routes, verifies, and escalates incidents for a human dispatcher.

---

## Output Schema

The model is strictly trained to emit valid, unadorned JSON adhering to the following schema:

```json
{
  "incident_type": "Structure Fire | Vehicle Accident / Rescue | Medical Emergency | Severe Weather / Flooding | Hazardous Materials | Civil Unrest / Active Threat | Infrastructure Failure / Gas Leak | False Alarm / Non-Emergency",
  "casualty_risk_level": 1, 2, 3, 4, or 5,
  "credibility_score": 0.0 to 1.0,
  "location_text": "Extracted street, intersection, or civic landmark"
}
```

---

## Performance & Proof on 100 Held-Out Samples

| Metric | Base Gemma 4 (Zero-Shot) | SentinelSentry Gemma 4 LoRA | Improvement |
|---|:---:|:---:|:---:|
| **Strict JSON Validity Rate** | 62.0% | **100.0%** | **+38.0%** |
| **Incident-Type Accuracy** | 0.0% | **63.0%** | **+63.0%** |
| **Casualty Risk MAE (Scale 1–5)** | 2.15 | **1.06** | **-1.09 error** |

---

## Multimodal Support

The SentinelSentry Gemma 4 pipeline supports text-only dispatch lines as well as multimodal inputs with attached crisis imagery (e.g., flooded roads, structural fires, vehicle entrapment). When image data is provided, visual damage indicators refine casualty risk assessment and elevate credibility scores.

---

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-e2b-it")
model = PeftModel.from_pretrained(base_model, "sentinelsentry/gemma-4-crisis-triage")
```
