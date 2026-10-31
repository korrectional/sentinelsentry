"""
Utility to publish SentinelSentry Gemma 2B PEFT adapter and dataset to Hugging Face Hub.
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
        commit_message="Add SentinelSentry Gemma 2B LoRA crisis triage adapter weights and card",
    )
    print(f"[SUCCESS] Model successfully published at: https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


def push_dataset_to_hub(
    dataset_repo_id: str,
    data_dir: Path,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Uploads training and held-out test dataset jsonl files and dataset card to Hugging Face."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[!] HF_TOKEN not provided. Skipping dataset upload.")
        return ""

    api = HfApi(token=hf_token)
    print(f"Creating / verifying Hugging Face Dataset repository '{dataset_repo_id}'...")
    create_repo(repo_id=dataset_repo_id, repo_type="dataset", token=hf_token, private=private, exist_ok=True)

    print(f"Uploading dataset files from {data_dir} to {dataset_repo_id}...")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=dataset_repo_id,
        repo_type="dataset",
        commit_message="Add SentinelSentry 900 train + 100 held-out test crisis dispatch dataset",
    )
    print(f"[SUCCESS] Dataset successfully published at: https://huggingface.co/datasets/{dataset_repo_id}")
    return f"https://huggingface.co/datasets/{dataset_repo_id}"


def main():
    parser = argparse.ArgumentParser(description="Publish SentinelSentry model and dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", default=settings.HF_REPO_ID, help="HF model repo (e.g. sentinelsentry/gemma-2b-crisis-triage)")
    parser.add_argument("--token", default=None, help="Hugging Face personal access token")
    parser.add_argument("--adapter-dir", default=str(settings.ADAPTER_DIR), help="Directory containing adapter files")
    parser.add_argument("--dataset-repo-id", default="sentinelsentry/crisis-lex-dispatch-subset", help="HF dataset repo")
    parser.add_argument("--data-dir", default=str(settings.DATA_DIR), help="Directory containing train and test jsonl files")
    parser.add_argument("--dataset-only", action="store_true", help="Only push dataset")
    parser.add_argument("--model-only", action="store_true", help="Only push model")

    args = parser.parse_args()

    adapter_path = Path(args.adapter_dir)
    data_path = Path(args.data_dir)

    if not args.dataset_only:
        push_adapter_to_hub(repo_id=args.repo_id, adapter_dir=adapter_path, token=args.token)

    if not args.model_only:
        push_dataset_to_hub(dataset_repo_id=args.dataset_repo_id, data_dir=data_path, token=args.token)


if __name__ == "__main__":
    main()
