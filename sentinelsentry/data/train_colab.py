"""
Reproducible QLoRA training script for Gemma 2B on Colab T4 (~10 mins training time).
Trains PEFT adapter to output strict crisis triage JSON.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
MODEL_ID = "google/gemma-2b-it"
OUTPUT_DIR = "./sentinelsentry-gemma-2b-adapter"
TRAIN_FILE = "train_crisis_triage.jsonl"
EVAL_FILE = "test_heldout_dispatch.jsonl"

def main():
    print(f"--- SentinelSentry Gemma 2B QLoRA Training ---")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. 4-bit Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load Tokenizer & Quantized Base Model
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Load Dataset
    print(f"Loading training data from {TRAIN_FILE}...")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": EVAL_FILE})

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        dataset_text_field="training_text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # 7. Save Final Adapter Weights
    print(f"Saving PEFT adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! Ready to evaluate or push to Hugging Face Hub.")

if __name__ == "__main__":
    main()
