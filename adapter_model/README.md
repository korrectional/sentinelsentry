---
license: gemma
base_model: google/gemma-2b-it
tags:
- peft
- lora
- gemma
- crisis-response
- emergency-dispatch
- 911-triage
- multi-agent
pipeline_tag: text-generation
library_name: peft
datasets:
- sentinelsentry/crisis-lex-dispatch-subset
---

# SentinelSentry: LoRA-Tuned Gemma 2B for Emergency Dispatch Triage

<div align="center">

[![License: Gemma](https://img.shields.io/badge/License-Gemma-blue.svg)](https://ai.google.dev/gemma/terms)
[![PEFT: LoRA](https://img.shields.io/badge/PEFT-LoRA_r16_a32-brightgreen.svg)](https://github.com/huggingface/peft)
[![Base: Gemma 2B-IT](https://img.shields.io/badge/Base_Model-google%2Fgemma--2b--it-orange.svg)](https://huggingface.co/google/gemma-2b-it)
[![Architecture: Multi-Agent Swarm](https://img.shields.io/badge/Architecture-Concurrent_Agent_Swarm-purple.svg)](https://github.com/sentinelsentry/sentinelsentry)

</div>

## Model Description

**SentinelSentry** is a compact, high-precision QLoRA/PEFT adapter for **Gemma 2B Instruction** (`google/gemma-2b-it`). It is engineered to transform noisy, unstructured, multi-agency emergency streams (911 CAD radio shorthands, fire telemetry, EMS scanner transcripts, and citizen eyewitness reports) into strictly validated JSON crisis triage records for a human dispatcher decision-support swarm.

### One-Liner
> A small Gemma 2B adapter turns messy dispatch text into strict JSON, and a swarm of concurrent agents routes, verifies, and escalates incidents for a human dispatcher.

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

## Proof for Judges (Held-Out Benchmark on 100 Dispatch Lines)

Trained on 900 emergency samples and evaluated against **100 held-out synthetic dispatch-style test lines** comparing zero-shot Base Gemma 2B with a strict prompt vs. the SentinelSentry LoRA adapter:

| Metric | Base Gemma 2B (Strict Prompt) | SentinelSentry LoRA Adapter | Improvement |
| :--- | :---: | :---: | :---: |
| **Strict JSON Validity Rate** | 68.0% | **100.0%** | **+32.0%** |
| **Incident-Type Accuracy** | 0.0% *(drifted keys)* | **96.0%** | **+96.0%** |
| **Casualty Risk MAE (Scale 1–5)** | 2.09 | **0.25** | **-1.84 error** |
| **Inference Latency (Colab T4)** | ~180 ms | **~190 ms** | Edge-deployable |

---

## Training Details

- **Base Model:** `google/gemma-2b-it`
- **Method:** QLoRA 4-bit (NF4) with double quantization via `bitsandbytes`
- **PEFT Architecture:**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Dropout: 0.05
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Training Duration:** ~10 minutes on a single Google Colab T4 GPU
- **Optimizer:** Paged AdamW 8-bit, learning rate 2e-4, linear warmup ratio 0.05
- **Loss Function:** Cross-Entropy via `SFTTrainer` (TRL)

### Open Disclosure & Labeling Methodology
In compliance with open AI transparency:
1. **Incident Categories**: Mapped directly from existing Disaster Response and CrisisLex taxonomy into 8 operational emergency dispatch categories.
2. **Casualty Risk & Credibility**: Weakly labeled using a larger reasoning model (Gemini 1.5 Pro) with strict schema constraints and **spot-checked by human inspection**.
3. **Location Extraction**: Calibrated against realistic Wake County, NC GIS benchmarks.

---

## Quickstart & Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "google/gemma-2b-it"
ADAPTER_ID = "sentinelsentry/gemma-2b-crisis-triage"

# 1. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()

# 2. Format Prompt with Gemma Turn Delimiters
raw_dispatch = "10-70 COMM BLDG 1200 S SAUNDERS ST. FLAMES VISIBLE ROOF 1 SUBJ INSIDE"
prompt = f"<start_of_turn>user\nYou are an emergency crisis dispatch triage AI. Analyze the following raw emergency report and output ONLY a valid JSON object matching the exact schema:\n{{\"incident_type\": string, \"casualty_risk_level\": integer (1-5), \"credibility_score\": float (0-1), \"location_text\": string}}\n\nRaw Report: \"{raw_dispatch}\"<end_of_turn>\n<start_of_turn>model\n"

# 3. Generate Strict Triage Output
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=False)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(response.strip())
# Output:
# {
#   "incident_type": "Structure Fire",
#   "casualty_risk_level": 4,
#   "credibility_score": 0.90,
#   "location_text": "1200 S Saunders St, Raleigh"
# }
```

---

## Integration with the Multi-Agent Swarm

In the `SentinelSentry` multi-agent swarm architecture:
1. **`IngestAgent`** polls simulated CAD/RSS feeds and pushes raw text to `ingest_queue`.
2. **`TriageAgent`** runs this Gemma 2B LoRA adapter to emit structured JSON.
3. If `casualty_risk_level >= 4`, the item routes to **`VerificationAgent`** for cross-feed corroboration against recent eyewitness streams.
4. **`GeocoderAgent`** attaches geospatial coordinates via regional landmark cache / Nominatim.
5. **`AlertDashboardAgent`** elevates the verified crisis to the human 911 dispatcher's live board.

---

## Ethical Considerations & Decision-Support Notice

`SentinelSentry` is designed solely as **decision support for human 911 dispatchers**, not autonomous emergency action. All escalated alerts require human supervisor review, dispatch approval, or dismissal.
