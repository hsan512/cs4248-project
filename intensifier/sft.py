import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import pandas as pd
from datasets import Dataset

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SPECIAL_TOKENS = ["<USER>", "<URL>", "<TAG>"]

@dataclass
class SFTConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    train_csv: str = "data/train.csv"
    input_col: str = "sentiment_text"
    target_col: str = "intensified_text"
    output_dir: str = "./outputs/sft"

    max_len: int = 512
    lr: float = 2e-5
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 2

    load_in_4bit: bool = True
    use_bf16: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    val_ratio: float = 0.05
    seed: int = 42
    save_steps: int = 200
    logging_steps: int = 20

class CompletionOnlyCollator:
    def __init__(self, response_template: str, tokenizer):
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.printed = False 

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        
        for i in range(len(labels)):
            instance_ids = labels[i].tolist()
            
            # 1. Decode the whole sequence to find exactly where the text is
            full_decoded = self.tokenizer.decode(instance_ids)
            
            # 2. Find the character position of the template
            template_start_char = full_decoded.find(self.response_template)
            
            if template_start_char != -1:
                # Find the character position where the actual response begins
                response_start_char = template_start_char + len(self.response_template)
                
                # 3. Map that character position back to a token index
                tokenized_offsets = self.tokenizer(
                    full_decoded, 
                    return_offsets_mapping=True, 
                    add_special_tokens=False
                )["offset_mapping"]
                
                idx = -1
                for j, (start, end) in enumerate(tokenized_offsets):
                    if start >= response_start_char:
                        idx = j
                        break
                
                if idx != -1:
                    labels[i, :idx] = -100
                else:
                    labels[i, :] = -100
            else:
                labels[i, :] = -100
                
        batch["labels"] = labels
        if not self.printed:
            self._debug_log(batch)
            self.printed = True
        return batch

    def _debug_log(self, batch):
        return batch

    def _debug_log(self, batch):
        print("\n" + "="*50)
        print("DEBUG: SFT COLLATOR CHECK")
        label_ids = batch["labels"][0].clone()
        # Filter out the mask value (-100)
        actual_trained_ids = [tid for tid in label_ids.tolist() if tid != -100]
        trained_text = self.tokenizer.decode(actual_trained_ids, skip_special_tokens=False)
        
        print(f"Trained Portion (Loss is calculated ONLY on this):")
        print(f"'{trained_text}'")
        print("="*50 + "\n")


def build_sft_prompt(tokenizer, source_text: str) -> str:
    source_text = str(source_text).strip()

    if source_text.startswith("[POS]"):
        token = "POS"
        plain = source_text[len("[POS]"):].strip()
    elif source_text.startswith("[NEG]"):
        token = "NEG"
        plain = source_text[len("[NEG]"):].strip()
    else:
        raise ValueError(f"Unknown sentiment token in source_text: {source_text[:40]}")

    direction_word = "positive" if token == "POS" else "negative"

    messages = [
        {
            "role": "user",
            "content": (
                "You are rewriting a tweet into a more emotionally intense version.\n"
                f"Target direction: {direction_word}.\n"
                "Preserve tweet style and informal noise. Keep masked swears like **** unchanged.\n"
                "Do not explain. Output only the rewritten tweet.\n\n"
                f"Original tweet: {plain}"
            )
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_examples(tokenizer, df: pd.DataFrame, input_col: str, target_col: str) -> Dataset:
    rows: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        prompt = build_sft_prompt(tokenizer, row[input_col])
        target = str(row[target_col]).strip()

        if not target:
            continue

        rows.append({
            "prompt": prompt,
            "target": target,
        })

    if not rows:
        raise ValueError("No valid training rows found after preprocessing.")

    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def make_tokenize_fn(tokenizer, cfg: SFTConfig):
    def tokenize_example(example: Dict[str, str]) -> Dict[str, List[int]]:
        full_text = example["prompt"] + example["target"] + (tokenizer.eos_token or "")
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=cfg.max_len,
            add_special_tokens=False,
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    return tokenize_example


def create_model_and_tokenizer(cfg: SFTConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # ---- ADD SPECIAL TOKENS ----
    special_tokens_dict = {
        "additional_special_tokens": SPECIAL_TOKENS
    }

    num_added = tokenizer.add_special_tokens(special_tokens_dict)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if cfg.load_in_4bit:
        compute_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quant_cfg,
        dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        device_map="auto",
    )

    # ---- IMPORTANT: resize embeddings AFTER loading ----
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()

    return model, tokenizer

def split_dataset(ds: Dataset, val_ratio: float, seed: int):
    if val_ratio <= 0.0:
        return ds, None

    if len(ds) < 2:
        return ds, None

    val_size = max(1, int(len(ds) * val_ratio))
    if val_size >= len(ds):
        val_size = len(ds) - 1

    split = ds.train_test_split(test_size=val_size, seed=seed)
    return split["train"], split["test"]


def main(cfg: SFTConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    df = pd.read_csv(cfg.train_csv)

    for col in [cfg.input_col, cfg.target_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found columns: {list(df.columns)}")

    df = df[[cfg.input_col, cfg.target_col]].dropna().copy()
    df[cfg.input_col] = df[cfg.input_col].astype(str)
    df[cfg.target_col] = df[cfg.target_col].astype(str)

    model, tokenizer = create_model_and_tokenizer(cfg)

    tokenize_fn = make_tokenize_fn(tokenizer, cfg)

    ds = build_examples(tokenizer, df, cfg.input_col, cfg.target_col)
    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)

    train_ds, val_ds = split_dataset(ds, cfg.val_ratio, cfg.seed)

    response_template = "<|assistant|>\n" 
    
    data_collator = CompletionOnlyCollator(
        response_template=response_template, 
        tokenizer=tokenizer
    )

    has_bf16 = cfg.use_bf16
    has_fp16 = not cfg.use_bf16

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=has_bf16,
        fp16=has_fp16,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=cfg.save_steps if val_ds is not None else None,
        load_best_model_at_end=True if val_ds is not None else False,
        metric_for_best_model="eval_loss" if val_ds is not None else None,
        greater_is_better=False if val_ds is not None else None,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    model.print_trainable_parameters()
    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(cfg.output_dir, "sft_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    print(f"Saved SFT model to: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--input_col", type=str, default="sentiment_text")
    parser.add_argument("--target_col", type=str, default="intensified_text")
    parser.add_argument("--output_dir", type=str, default="./outputs/sft")

    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=20)

    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)

    parser.add_argument("--use_bf16", dest="use_bf16", action="store_true")
    parser.add_argument("--no_use_bf16", dest="use_bf16", action="store_false")
    parser.set_defaults(use_bf16=True)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    cfg = SFTConfig(
        model_name=args.model_name,
        train_csv=args.train_csv,
        input_col=args.input_col,
        target_col=args.target_col,
        output_dir=args.output_dir,
        max_len=args.max_len,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        load_in_4bit=args.load_in_4bit,
        use_bf16=args.use_bf16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    main(cfg)