"""
Utility to publish SentinelSentry Gemma 4 PEFT adapter and dataset to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sentinelsentry.config import settings


def push_adapter_to_hub(
    repo_id: str,
    adapter_dir: Path,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Uploads PEFT adapter directory (safetensors, config, tokenizer, README) to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[!] HF_TOKEN not provided. Please set $env:HF_TOKEN or pass --token <your_token>.")
        return ""

    api = HfApi(token=hf_token)
    print(f"Creating / verifying Hugging Face repository '{repo_id}'...")
    repo_url = create_repo(repo_id=repo_id, token=hf_token, private=private, exist_ok=True)
    print(f"Repository URL: {repo_url}")

    print(f"Uploading adapter files from {adapter_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add SentinelSentry Gemma 4 LoRA crisis triage adapter weights and card",
    )
    print(f"[SUCCESS] Model successfully published at: https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


def push_dataset_to_hub(
    dataset_repo_id: str,
    data_dir: Path,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Uploads crisis triage dataset jsonl files and DATASET_CARD.md to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[!] HF_TOKEN not provided. Please set $env:HF_TOKEN or pass --token <your_token>.")
        return ""

    api = HfApi(token=hf_token)
    print(f"Creating / verifying Hugging Face dataset repository '{dataset_repo_id}'...")
    repo_url = create_repo(repo_id=dataset_repo_id, token=hf_token, repo_type="dataset", private=private, exist_ok=True)

    print(f"Uploading dataset files from {data_dir} to {dataset_repo_id}...")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=dataset_repo_id,
        repo_type="dataset",
        commit_message="Add SentinelSentry 900-sample train and 100-sample test dispatch datasets",
    )
    print(f"[SUCCESS] Dataset successfully published at: https://huggingface.co/datasets/{dataset_repo_id}")
    return f"https://huggingface.co/datasets/{dataset_repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish SentinelSentry Gemma 4 LoRA adapter & dataset to Hugging Face.")
    parser.add_argument("--repo-id", default=settings.HF_REPO_ID, help="HF model repo id")
    parser.add_argument("--dataset-repo-id", default="sentinelsentry/crisis-lex-dispatch-subset", help="HF dataset repo id")
    parser.add_argument("--token", default=None, help="Hugging Face write token")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    adapter_dir = settings.ADAPTER_DIR
    data_dir = settings.DATA_DIR

    print("=" * 70)
    print(" SentinelSentry: Hugging Face Publisher")
    print(f" Model Adapter Dir:  {adapter_dir}")
    print(f" Dataset Dir:        {data_dir}")
    print(f" Target Model Repo:  {args.repo_id}")
    print(f" Target Data Repo:   {args.dataset_repo_id}")
    print("=" * 70)

    push_adapter_to_hub(args.repo_id, adapter_dir, token=args.token, private=args.private)
    push_dataset_to_hub(args.dataset_repo_id, data_dir, token=args.token, private=args.private)


if __name__ == "__main__":
    main()
