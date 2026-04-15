import os
import torch
import pandas as pd
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==============================
# CONFIG
# ==============================
EVAL_MODEL_NAME = "meta-llama/Meta-Llama-3-8B" 
OUTPUT_DIR = "eval_outputs"
MAX_INPUT_LEN = 512 
PPL_BATCH_SIZE = 4   # Adjust based on VRAM (4-8 is usually safe for 8B model)
HF_TOKEN = "your_hf_token" 

CSV_FILES = {
    "BASE": os.path.join(OUTPUT_DIR, "base_generations.csv"),
    "SFT": os.path.join(OUTPUT_DIR, "sft_generations.csv"),
    "RL": os.path.join(OUTPUT_DIR, "rl_generations.csv"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# CORE PPL ENGINE
# ==============================
def compute_perplexity(model, tokenizer, texts, batch_size=1, desc="Perplexity"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for i in tqdm.tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = [str(t) for t in texts[i:i + batch_size] if t]
        if not batch: continue

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
            return_tensors="pt",
        ).to(device)

        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        
        # Ensure labels for padding are -100 to be ignored by CrossEntropyLoss
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100 

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            
            # Correctly accumulate total log-likelihood (loss * valid_tokens)
            num_tokens = (labels != -100).sum().item()
            if num_tokens > 0:
                total_loss += outputs.loss.item() * num_tokens
                total_tokens += num_tokens

    if total_tokens == 0: return float('nan')
    # Final PPL = exp(Total Sum of Losses / Total Number of Tokens)
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

# ==============================
# MAIN
# ==============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_report = {}

    print(f"Loading Evaluator Model: {EVAL_MODEL_NAME}...")
    
    # Load with 4-bit quantization if you have < 24GB VRAM
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        EVAL_MODEL_NAME, 
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        # quantization_config=bnb_config # Uncomment if you hit Memory Error
    )
    
    tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Standard for loss calculation

    # 1. COMPUTE GLOBAL BASELINES (Human Text)
    print("\n--- Computing Global Baselines ---")
    
    # SST-5 Baseline
    sst5 = load_dataset("SetFit/sst5", split="test")
    ppl_sst5 = compute_perplexity(model, tokenizer, sst5["text"], PPL_BATCH_SIZE, "Ref: SST-5")
    summary_report["GLOBAL_SST5"] = ppl_sst5

    # IMDb Baseline (Subset of 1000 for efficiency)
    imdb = load_dataset("stanfordnlp/imdb", split="test")
    ppl_imdb = compute_perplexity(model, tokenizer, imdb["text"][:1000], PPL_BATCH_SIZE, "Ref: IMDb")
    summary_report["GLOBAL_IMDB"] = ppl_imdb
    # PPL of the human reference text specifically in that file
    ppl_ref = compute_perplexity(model, tokenizer, df["reference_text"].tolist(), PPL_BATCH_SIZE, f"Ref: {model_key}")
    summary_report[f"TWEET_Reference"] = ppl_ref

    # 2. COMPUTE MODEL GENERATIONS
    print("\n--- Computing Model-Specific Scores ---")
    
    for model_key, path in CSV_FILES.items():
        if not os.path.exists(path):
            print(f"Skipping {model_key}: File not found at {path}")
            continue

        df = pd.read_csv(path).dropna(subset=["generated_text", "reference_text"])
        
        # PPL of the text our model produced
        ppl_gen = compute_perplexity(model, tokenizer, df["generated_text"].tolist(), PPL_BATCH_SIZE, f"Gen: {model_key}")
        
        summary_report[f"{model_key}_Generated"] = ppl_gen

    # 3. FINAL SUMMARY OUTPUT
    final_output = "\n" + "="*50 + "\n"
    final_output += "FINAL PERPLEXITY EVALUATION (Evaluator: Llama-3-8B)\n"
    final_output += "="*50 + "\n\n"
    
    # Header
    final_output += f"{'Category':<30} | {'Perplexity':<12}\n"
    final_output += "-"*45 + "\n"
    
    for key, val in summary_report.items():
        final_output += f"{key:<30} | {val:>10.4f}\n"
    
    final_output += "="*50 + "\n"
    
    # Save to one master file
    master_log_path = os.path.join(OUTPUT_DIR, "comprehensive_ppl_results.txt")
    with open(master_log_path, "w") as f:
        f.write(final_output)
    
    print(final_output)
    print(f"Full results saved to: {master_log_path}")

if __name__ == "__main__":
    main()