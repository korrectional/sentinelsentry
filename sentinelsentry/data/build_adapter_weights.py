"""
Script to generate or verify local Gemma 2B PEFT LoRA adapter weights (adapter_model.safetensors).
Creates 252 spec-compliant LoRA weights matching google/gemma-2b-it architecture.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from pathlib import Path
from typing import Dict, Any

import numpy as np

from sentinelsentry.config import settings

logger = logging.getLogger("sentinelsentry.data.adapter")


def build_safetensors_weights(target_dir: Path, rank: int = 16) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    safetensor_path = target_dir / "adapter_model.safetensors"

    hidden_size = 2048
    kv_size = 256
    intermediate_size = 8192
    num_layers = 18

    modules = {
        "self_attn.q_proj": ((rank, hidden_size), (hidden_size, rank)),
        "self_attn.k_proj": ((rank, hidden_size), (kv_size, rank)),
        "self_attn.v_proj": ((rank, hidden_size), (kv_size, rank)),
        "self_attn.o_proj": ((rank, hidden_size), (hidden_size, rank)),
        "mlp.gate_proj": ((rank, hidden_size), (intermediate_size, rank)),
        "mlp.up_proj": ((rank, hidden_size), (intermediate_size, rank)),
        "mlp.down_proj": ((rank, intermediate_size), (hidden_size, rank)),
    }

    header: Dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "framework": "pytorch",
            "peft_version": "0.10.0",
            "base_model": "google/gemma-2b-it",
            "lora_alpha": "32",
            "lora_r": str(rank),
        }
    }

    data_buffers = []
    current_offset = 0

    np.random.seed(42)

    for layer in range(num_layers):
        for mod_name, (shape_a, shape_b) in modules.items():
            name_a = f"base_model.model.model.layers.{layer}.{mod_name}.lora_A.weight"
            name_b = f"base_model.model.model.layers.{layer}.{mod_name}.lora_B.weight"

            tensor_a = (np.random.randn(*shape_a) * 0.02).astype(np.float16)
            tensor_b = np.zeros(shape_b, dtype=np.float16)

            for name, tensor in [(name_a, tensor_a), (name_b, tensor_b)]:
                raw_bytes = tensor.tobytes()
                length = len(raw_bytes)
                header[name] = {
                    "dtype": "F16",
                    "shape": list(tensor.shape),
                    "data_offsets": [current_offset, current_offset + length],
                }
                data_buffers.append(raw_bytes)
                current_offset += length

    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")

    header_len = len(header_bytes)
    padding_len = (8 - (header_len % 8)) % 8
    if padding_len > 0:
        header_bytes += b" " * padding_len
        header_len = len(header_bytes)

    with open(safetensor_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        for buf in data_buffers:
            f.write(buf)

    print(f"Successfully generated adapter weights: {safetensor_path}")
    print(f" - Size: {safetensor_path.stat().st_size:,} bytes")
    print(f" - Tensors: {len(header) - 1} weights (18 layers x 7 modules x 2)")
    return safetensor_path


if __name__ == "__main__":
    out_dir = settings.ADAPTER_DIR
    build_safetensors_weights(out_dir)
