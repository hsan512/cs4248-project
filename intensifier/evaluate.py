import re
import torch
import pandas as pd
import tqdm

from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score

from utils.clean_text import preprocess_pipeline

tqdm.tqdm.pandas()

# ==============================
# CONFIG
# ==============================
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GEN_MODEL_PATH = "outputs/ppo_custom/ppo_final"   # SFT / RL adapter path
CLS_MODEL_NAME = "classifier/best_model"    # sentiment classifier

INPUT_COL = "sentiment_text"
TARGET_COL = "intensified_text"
LABEL_COL = "sentiment"

MAX_INPUT_LEN = 128
MAX_NEW_TOKENS = 32
GEN_BATCH_SIZE = 16
CLS_BATCH_SIZE = 32

DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def map_true_label_to_full_id(label_name: str, label2id_dict: dict):
    """
    Normalizes the label string (e.g., 'Positive') and finds its 
    corresponding integer ID in the classifier's mapping.
    """
    norm = normalize_label_name(label_name)
    
    # Check for direct matches or common variations
    for k, v in label2id_dict.items():
        if normalize_label_name(k) == norm:
            return v
            
    # Fallback/Safety Check
    raise ValueError(f"Label '{label_name}' not found in classifier mapping: {label2id_dict}")


# 1. Load tokenizer from the SFT/PPO path to get custom special tokens
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_PATH)
if gen_tokenizer.pad_token is None:
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
gen_tokenizer.padding_side = "left"

# 2. Load Base Model and RESIZE before applying LoRA
gen_base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
gen_base_model.resize_token_embeddings(len(gen_tokenizer))

# 3. Load the PPO Adapter
gen_model = PeftModel.from_pretrained(gen_base_model, GEN_MODEL_PATH)
gen_model.eval()


# ==============================
# Load Classifier
# ==============================
cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
cls_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME).to(device)
cls_model.eval()

id2label = {int(k): v for k, v in cls_model.config.id2label.items()}
label2id = {v.lower(): int(k) for k, v in cls_model.config.id2label.items()}


# ==============================
# Helpers
# ==============================
def normalize_label_name(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    return x

def find_sentiment_indices(id2label_dict):
    neg_idx = None
    neu_idx = None
    pos_idx = None

    for idx, name in id2label_dict.items():
        norm = normalize_label_name(name)

        if norm in {"negative", "neg"}:
            neg_idx = idx
        elif norm in {"neutral", "neu"}:
            neu_idx = idx
        elif norm in {"positive", "pos"}:
            pos_idx = idx

    if neg_idx is None or pos_idx is None:
        raise ValueError(
            f"Could not find positive/negative label indices from classifier id2label={id2label_dict}"
        )

    return neg_idx, neu_idx, pos_idx

neg_idx, neu_idx, pos_idx = find_sentiment_indices(id2label)


def extract_clean_text(x):
    x = str(x)

    if x.startswith("[POS] "):
        label = "positive"
        text = x[len("[POS] "):]
    elif x.startswith("[NEG] "):
        label = "negative"
        text = x[len("[NEG] "):]
    else:
        raise ValueError(f"Unknown label token in input: {x[:40]}")

    return text, label


def normalize_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# ==============================
# Prompting (FIXED to match PPO)
# ==============================
def build_prompt_eval(tokenizer, text: str, direction: str) -> str:
    messages = [{
        "role": "user",
        "content": (
            "You are rewriting a tweet into a more emotionally intense version.\n"
            f"Target direction: {direction}.\n"
            "Preserve tweet style and informal noise. Keep masked swears like **** unchanged.\n"
            "Do not explain. Output only the rewritten tweet.\n\n"
            f"Original tweet: {text}"
        )
    }]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_response_eval(full_output: str) -> str:
    """Matches the splitting logic used in PPO Training"""
    parts = full_output.split("<|assistant|>\n")
    if len(parts) > 1:
        # Strip EOS and leading/trailing whitespace
        return parts[-1].split("</s>")[0].strip()
    return full_output.strip()


def distinct_n(texts, n=1):
    total = 0
    uniq = set()

    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            continue

        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        total += len(ngrams)
        uniq.update(ngrams)

    return len(uniq) / total if total > 0 else 0.0


def safe_preprocess_for_classifier(text):
    """
    Apply the same cleaning pipeline used for classifier training, if possible.
    Falls back to raw text if preprocess_pipeline errors.
    """
    try:
        cleaned = preprocess_pipeline(str(text))
        if isinstance(cleaned, tuple):
            # in case your pipeline returns (text, username, url) or similar
            cleaned = cleaned[0]
        cleaned = str(cleaned).strip()
        return cleaned if cleaned else str(text)
    except Exception:
        return str(text)


def map_true_label_to_binary_id(label: str, neg_idx: int, pos_idx: int):
    norm = normalize_label_name(label)
    if norm in {"negative", "neg"}:
        return neg_idx
    if norm in {"positive", "pos"}:
        return pos_idx
    return None


# ==============================
# Generation
# ==============================
def generate_texts(model, tokenizer, inputs, labels, batch_size=8):
    generations = []
    
    for i in tqdm.tqdm(range(0, len(inputs), batch_size), desc="Generating"):
        batch_inputs = inputs[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        
        # Build prompts using the exact chat template from training
        batch_prompts = [build_prompt_eval(tokenizer, txt, lbl) for txt, lbl in zip(batch_inputs, batch_labels)]

        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode without skipping special tokens so we can split on <|assistant|>
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

        for full_text in decoded:
            clean_resp = extract_response_eval(full_text)
            generations.append(clean_resp if clean_resp else ".")

    return generations


# ==============================
# Classifier scoring
# ==============================
def classifier_predict_full(model, tokenizer, texts, batch_size=32):
    pred_ids = []

    for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Classifier scoring"):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [safe_preprocess_for_classifier(x) for x in batch_texts]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1)

        pred_ids.extend(preds.cpu().tolist())

    return pred_ids


# ==============================
# Generative metrics
# ==============================
def compute_bleu(references, hypotheses):
    refs = [[ref.split()] for ref in references]
    hyps = [hyp.split() for hyp in hypotheses]
    smoothie = SmoothingFunction().method4
    return corpus_bleu(refs, hyps, smoothing_function=smoothie)


def compute_rouge(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_list, rouge2_list, rougeL_list = [], [], []

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1_list.append(scores["rouge1"].fmeasure)
        rouge2_list.append(scores["rouge2"].fmeasure)
        rougeL_list.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": sum(rouge1_list) / len(rouge1_list),
        "rouge2": sum(rouge2_list) / len(rouge2_list),
        "rougeL": sum(rougeL_list) / len(rougeL_list),
    }


def compute_exact_match(references, hypotheses):
    matches = [
        int(normalize_text(ref) == normalize_text(hyp))
        for ref, hyp in zip(references, hypotheses)
    ]
    return sum(matches) / len(matches)


def compute_bertscore(references, hypotheses):
    P, R, F1 = bertscore_score(
        hypotheses,
        references,
        lang="en",
        verbose=True,
        device=device.type,
    )
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }

# ==============================
# Load test dataset
# ==============================
df = pd.read_csv("test.csv").dropna(subset=[INPUT_COL, TARGET_COL])

# Use your existing extraction logic
tmp = df[INPUT_COL].progress_apply(extract_clean_text)
# Assigning to specific names to match the generator call below
df[["clean_input", "label_name"]] = pd.DataFrame(tmp.tolist(), index=df.index)

df = df.dropna(subset=["clean_input", TARGET_COL, "label_name"]).reset_index(drop=True)

# Prepare lists for metrics
inputs = df["clean_input"].tolist()
references = df[TARGET_COL].tolist()
labels = df["label_name"].tolist()

# Map strings to the integer IDs the classifier expects
y_true = [map_true_label_to_full_id(l, label2id) for l in labels]

# ==============================
# Generate predictions (FIXED)
# ==============================
generated_texts = generate_texts(
    gen_model, 
    gen_tokenizer, 
    inputs,      # Matches the list created above
    labels,      # Matches the list created above
    batch_size=GEN_BATCH_SIZE
)

# ==============================
# Save outputs
# ==============================
result_df = pd.DataFrame({
    "input_text": inputs,
    "reference_text": references,
    "generated_text": generated_texts,
    "sentiment": labels,
})
result_df.to_csv("generative_eval_results.csv", index=False, encoding="utf-8")
print("\nSaved detailed results to generative_eval_results.csv")


# ==============================
# Compute generative metrics
# ==============================
bleu = compute_bleu(references, generated_texts)
rouge = compute_rouge(references, generated_texts)
exact_match = compute_exact_match(references, generated_texts)
bertscore = compute_bertscore(references, generated_texts)

distinct_1 = distinct_n(generated_texts, n=1)
distinct_2 = distinct_n(generated_texts, n=2)

length_ratios = []
copy_flags = []

for src, gen in zip(inputs, generated_texts):
    src_len = max(len(src.split()), 1)
    gen_len = len(gen.split())
    length_ratios.append(gen_len / src_len)
    copy_flags.append(int(normalize_text(src) == normalize_text(gen)))

avg_length_ratio = sum(length_ratios) / len(length_ratios)
copy_rate = sum(copy_flags) / len(copy_flags)


# ==============================
# Sentiment evaluation on generated text
# ==============================
y_pred_sent = classifier_predict_full(
    cls_model,
    cls_tokenizer,
    generated_texts,
    batch_size=CLS_BATCH_SIZE,
)

all_labels = sorted(id2label.keys())

sent_acc = accuracy_score(y_true, y_pred_sent)

sent_macro_f1 = f1_score(
    y_true,
    y_pred_sent,
    average="macro",
)
posneg_to_neu = sum(
    1 for t, p in zip(y_true, y_pred_sent)
    if t in [neg_idx, pos_idx] and p == neu_idx
)
total_posneg = sum(1 for t in y_true if t in [neg_idx, pos_idx])

neu_leak_rate = posneg_to_neu / total_posneg if total_posneg > 0 else 0.0

# ==============================
# Print results
# ==============================


print("\n===== Neutral Leakage =====")
print(f"POS/NEG predicted as NEU: {posneg_to_neu}/{total_posneg}")
print(f"Neutral Leakage Rate:     {neu_leak_rate:.4f}")
print("\n===== Generative Task Evaluation =====")
print(f"BLEU:                     {bleu:.4f}")
print(f"ROUGE-1 F1:               {rouge['rouge1']:.4f}")
print(f"ROUGE-2 F1:               {rouge['rouge2']:.4f}")
print(f"ROUGE-L F1:               {rouge['rougeL']:.4f}")
print(f"BERTScore Precision:      {bertscore['bertscore_precision']:.4f}")
print(f"BERTScore Recall:         {bertscore['bertscore_recall']:.4f}")
print(f"BERTScore F1:             {bertscore['bertscore_f1']:.4f}")
print(f"Exact Match:              {exact_match:.4f}")
print(f"Distinct-1:               {distinct_1:.4f}")
print(f"Distinct-2:               {distinct_2:.4f}")
print(f"Avg Length Ratio:         {avg_length_ratio:.4f}")
print(f"Copy Rate:                {copy_rate:.4f}")

print("\n===== Sentiment on Generated Text =====")
print(f"Sentiment Accuracy:       {sent_acc:.4f}")
print(f"Sentiment Macro-F1:       {sent_macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred_sent,
    labels=all_labels,
    target_names=[id2label[i] for i in all_labels],
    digits=4,
))