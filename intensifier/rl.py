import os
import re
import random
from dataclasses import dataclass
from typing import List, Tuple

from emot import emot
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

emo = emot()

# =========================
# Config
# =========================

@dataclass
class RLConfig:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    sft_path: str = "./outputs/sft/final"
    classifier_name: str = "./outputs/best_model"

    train_csv: str = "data/train.csv"
    text_col: str = "sentiment_text"
    output_dir: str = "./outputs/rl"

    max_prompt_len: int = 256  # Increased slightly for chat template overhead
    max_new_tokens: int = 256

    batch_size: int = 4
    total_steps: int = 3000
    epochs: int = 1

    learning_rate: float = 5e-7
    value_learning_rate: float = 1e-5
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    kl_coef: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    gamma: float = 1.0
    gae_lambda: float = 0.95

    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    repetition_penalty_weight: float = 0.10

    use_bf16: bool = True
    load_in_4bit: bool = True

    save_every: int = 500
    seed: int = 42


# =========================
# Utils & Preprocessing
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8):
    mask = mask.float()
    if dim is None:
        return (x * mask).sum() / (mask.sum() + eps)
    return (x * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)

def whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    mean = masked_mean(values, mask)
    var = masked_mean((values - mean) ** 2, mask)
    return (values - mean) / torch.sqrt(var + eps)

def is_bad_emoticon_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 6): min(len(text), end + 6)]
    if re.search(r"\d+(?:\.\d+)?\s*%\)", window) or \
       re.search(r"\d+(?:\.\d+)?\s*:\)", window) or \
       re.search(r"\d+(?:\.\d+)?\s*:-\)", window):
        return True
    left = text[start - 1] if start > 0 else ""
    return left.isdigit() or left == "%"

def extract_emojis_with_placeholders(text):
    if not text: return ""
    text = str(text)
    found = []
    
    emoji_info = emo.emoji(text)
    if emoji_info and "value" in emoji_info:
        for mean, loc in zip(emoji_info["mean"], emoji_info["location"]):
            found.append({"start": loc[0], "end": loc[1], "label": f"*{mean}*"})

    emoticon_info = emo.emoticons(text)
    if emoticon_info and "mean" in emoticon_info:
        for mean, loc in zip(emoticon_info["mean"], emoticon_info["location"]):
            if not is_bad_emoticon_context(text, loc[0], loc[1]):
                found.append({"start": loc[0], "end": loc[1], "label": f"*{mean}*"})

    found.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    filtered, occupied_until = [], -1
    for item in found:
        if item["start"] >= occupied_until:
            filtered.append(item)
            occupied_until = item["end"]

    pieces, last = [], 0
    for item in filtered:
        pieces.append(text[last:item["start"]])
        pieces.append(f" {item['label']} ")
        last = item["end"]
    pieces.append(text[last:])
    return re.sub(r"\s+", " ", "".join(pieces)).strip()

# =========================
# Prompting with Chat Template
# =========================

def build_prompt(tokenizer, text: str) -> str | None:
    text = str(text).strip()
    if text.startswith("[POS]"):
        direction, plain = "positive", text[len("[POS]"):].strip()
    elif text.startswith("[NEG]"):
        direction, plain = "negative", text[len("[NEG]"):].strip()
    else:
        return None

    messages = [{
        "role": "user",
        "content": (
            "You are rewriting a tweet into a more emotionally intense version.\n"
            f"Target direction: {direction}.\n"
            "Preserve tweet style and informal noise. Keep masked swears like **** unchanged.\n"
            "Do not explain. Output only the rewritten tweet.\n\n"
            f"Original tweet: {plain}"
        )
    }]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load_prompts(cfg: RLConfig, tokenizer) -> List[str]:
    df = pd.read_csv(cfg.train_csv)
    prompts = [build_prompt(tokenizer, row[cfg.text_col]) for _, row in df.iterrows()]
    prompts = [p for p in prompts if p is not None]
    if not prompts: raise ValueError("No valid prompts found.")
    return prompts

# =========================
# Model Components
# =========================

class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.value = nn.Linear(hidden_size, 1, bias=True, device=device, dtype=dtype)
        
        # Initialize to zero so the initial VF loss doesn't explode the LoRA weights
        torch.nn.init.zeros_(self.value.weight)
        torch.nn.init.zeros_(self.value.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.value(hidden_states).squeeze(-1)

class RewardScorer:
    def __init__(self, classifier_name: str, decode_tokenizer, device: torch.device, repetition_penalty_weight: float):
        self.device = device
        self.decode_tokenizer = decode_tokenizer
        self.repetition_penalty_weight = repetition_penalty_weight
        self.cls_tok = AutoTokenizer.from_pretrained(classifier_name)
        self.cls_mod = AutoModelForSequenceClassification.from_pretrained(classifier_name).to(device).eval()
        
        if self.cls_tok.pad_token is None:
            self.cls_tok.pad_token = self.cls_tok.eos_token
        
        label_map = {str(v).lower(): int(k) for k, v in self.cls_mod.config.id2label.items()}
        self.neg_idx = label_map["negative"]
        self.neu_idx = label_map["neutral"]
        self.pos_idx = label_map["positive"]

    def _extract_response(self, full_text: str) -> Tuple[str, str]:
        direction = "POS" if "direction: positive" in full_text.lower() else "NEG"
        # Use the specific ChatML tag as the split point
        parts = full_text.split("<|assistant|>\n")
        if len(parts) > 1:
            # Strip the EOS token </s> if present
            response = parts[-1].split("</s>")[0].strip()
        else:
            response = ""
        return direction, response

    @torch.no_grad()
    def score_batch(self, full_sequences: torch.Tensor) -> torch.Tensor:
        decoded = self.decode_tokenizer.batch_decode(full_sequences, skip_special_tokens=False)
        directions, cleaned_responses = [], []

        for text in decoded:
            dir_label, resp = self._extract_response(text)
            directions.append(dir_label)
            cleaned = extract_emojis_with_placeholders(resp)
            cleaned_responses.append(cleaned if cleaned.strip() else ".")

        toks = self.cls_tok(cleaned_responses, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        logits = self.cls_mod(**toks).logits
        
        # Convert logits to probabilities to make the "penalties" easier to tune
        probs = torch.softmax(logits, dim=-1)
        
        scores = []
        for i, d in enumerate(directions):
            p_pos = probs[i, self.pos_idx]
            p_neg = probs[i, self.neg_idx]
            p_neu = probs[i, getattr(self, 'neu_idx', 1)] # Ensure neu_idx in __init__

            if d == "POS":
                # Reward Positive, Penalize Neutral (0.5 weight) and Negative (1.0 weight)
                s = p_pos - (0.5 * p_neu) - p_neg
            else:
                # Reward Negative, Penalize Neutral (0.5 weight) and Positive (1.0 weight)
                s = p_neg - (0.5 * p_neu) - p_pos
            
            # Apply repetition penalty
            words = cleaned_responses[i].split()
            if len(words) > 0:
                penalty = (len(words) - len(set(words))) / len(words)
                s -= (self.repetition_penalty_weight * penalty)
            
            scores.append(s)

        return torch.stack(scores)

# =========================
# Core Logic
# =========================

def gather_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

@torch.no_grad()
def generate_batch(model, tokenizer, prompts: List[str], cfg: RLConfig, device: torch.device):
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=cfg.max_prompt_len, return_tensors="pt").to(device)
    generated = model.generate(
        **enc, max_new_tokens=cfg.max_new_tokens, do_sample=cfg.do_sample,
        temperature=cfg.temperature, top_p=cfg.top_p,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return enc["input_ids"], enc["attention_mask"], generated[:, enc["input_ids"].shape[1]:]

def build_full_sequences(prompt_ids, response_ids, pad_token_id):
    full = torch.cat([prompt_ids, response_ids], dim=1)
    attn = (full != pad_token_id).long()
    resp_mask = torch.zeros_like(full, dtype=torch.float32)
    resp_mask[:, prompt_ids.size(1):] = (response_ids != pad_token_id).float()
    return full, attn, resp_mask

def forward_policy_and_value(model, value_head, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    return out.logits, value_head(out.hidden_states[-1])

def compute_token_logps(logits, input_ids, response_mask):
    logps = gather_log_probs(logits[:, :-1, :], input_ids[:, 1:])
    return logps * response_mask[:, 1:]

def compute_gae(rewards, values, mask, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
    for t in reversed(range(rewards.size(1))):
        delta = rewards[:, t] + gamma * next_values[:, t] * mask[:, t] - values[:, t]
        lastgaelam = delta + gamma * gae_lambda * mask[:, t] * lastgaelam
        advantages[:, t] = lastgaelam
    return advantages, advantages + values

# =========================
# Main Execution
# =========================

def main():
    cfg = RLConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16

    # Load Tokenizer with SFT Special Tokens
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_path)
    # Ensure PAD is set for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Required for batch generation

    # Memory Alignment
    q_cfg = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=model_dtype
    ) if cfg.load_in_4bit else None

    def load_model(path, is_trainable=True):
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, 
            quantization_config=q_cfg, 
            device_map="auto"
        )
        # resize_token_embeddings MUST happen before PeftModel load
        base.resize_token_embeddings(len(tokenizer))
        if is_trainable and cfg.load_in_4bit:
            base = prepare_model_for_kbit_training(base)
        return PeftModel.from_pretrained(base, path, is_trainable=is_trainable)

    policy = load_model(cfg.sft_path, is_trainable=True).train()
    ref_model = load_model(cfg.sft_path, is_trainable=False).eval()

    value_head = ValueHead(policy.config.hidden_size, torch.float32, device).train()
    reward_scorer = RewardScorer(cfg.classifier_name, tokenizer, device, cfg.repetition_penalty_weight)

    # Optimizer with specific LR for Value Head
    optimizer = torch.optim.AdamW([
        {"params": [p for p in policy.parameters() if p.requires_grad], "lr": cfg.learning_rate},
        {"params": value_head.parameters(), "lr": cfg.value_learning_rate}
    ], eps=1e-5)

    prompts = load_prompts(cfg, tokenizer)
    
    # Training Loop
    for step in range(1, cfg.total_steps + 1):
        batch_prompts = random.sample(prompts, k=cfg.batch_size)

        # Rollout
        with torch.no_grad():
            p_ids, p_attn, r_ids = generate_batch(policy, tokenizer, batch_prompts, cfg, device)
            full_ids, full_attn, resp_mask = build_full_sequences(p_ids, r_ids, tokenizer.pad_token_id)
            
            # Get old policy and ref logprobs
            policy.eval()
            old_logits, old_v_full = forward_policy_and_value(policy, value_head, full_ids, full_attn)
            ref_logits = ref_model(input_ids=full_ids, attention_mask=full_attn).logits
            policy.train()

            old_logps = compute_token_logps(old_logits, full_ids, resp_mask)
            ref_logps = compute_token_logps(ref_logits, full_ids, resp_mask)
            
            # Reward Calculation
            scores = reward_scorer.score_batch(full_ids)
            kl = old_logps - ref_logps
            rewards = -cfg.kl_coef * kl

            # Terminal Reward Placement
            for i in range(full_ids.size(0)):
                # Find the actual last token of the response (before padding)
                # resp_mask has 1s for the response tokens
                actual_length = int(resp_mask[i].sum().item())
                if actual_length > 0:
                    # Place reward on the absolute final token of that sequence
                    rewards[i, actual_length - 1] += scores[i]

            # GAE
            old_values = old_v_full[:, 1:] * resp_mask[:, 1:]
            token_mask = resp_mask[:, 1:]
            advantages, returns = compute_gae(rewards, old_values, token_mask, cfg.gamma, cfg.gae_lambda)
            
            # Whitening
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update
        for _ in range(cfg.epochs):
            logits, v_full = forward_policy_and_value(policy, value_head, full_ids, full_attn)
            logps = compute_token_logps(logits, full_ids, resp_mask)
            values = v_full[:, 1:] * token_mask

            # Policy Loss
            ratio = torch.exp(logps - old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * advantages
            pg_loss = -masked_mean(torch.min(surr1, surr2), token_mask)

            # Value Loss (Clipped)
            v_clipped = old_values + torch.clamp(values - old_values, -cfg.value_clip_range, cfg.value_clip_range)
            vf_loss = 0.5 * masked_mean(torch.max((values - returns)**2, (v_clipped - returns)**2), token_mask)

            loss = pg_loss + cfg.vf_coef * vf_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

        # Monitoring
        if step % 10 == 0:
            print(f"Step {step} | Reward: {scores.mean().item():.3f} | KL: {kl.mean().item():.3f} | Loss: {loss.item():.4f}")
        
        if step % cfg.save_every == 0:
            checkpoint_dir = os.path.join(cfg.output_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save the LoRA adapters (Policy)
            policy.save_pretrained(checkpoint_dir)
            
            # Save the Value Head (since it's not part of the LoRA config)
            torch.save(value_head.state_dict(), os.path.join(checkpoint_dir, "value_head.pt"))
            
            # Save the tokenizer (so it stays with the model)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_dir}")
    # --- After the loop (Final Save) ---
    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    # Save final LoRA weights
    policy.save_pretrained(final_dir)
    # Save final Value Head
    torch.save(value_head.state_dict(), os.path.join(final_dir, "value_head.pt"))
    # Save tokenizer
    tokenizer.save_pretrained(final_dir)
    
    print(f"Training complete. Final model saved to: {final_dir}")

if __name__ == "__main__":
    main()