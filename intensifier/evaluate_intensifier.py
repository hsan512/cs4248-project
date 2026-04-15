import gc
import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
import tqdm
from datasets import load_dataset

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score
from emot import emot
from collections import Counter

emo = emot()

tqdm.tqdm.pandas()

# =========================
# Emoji / emoticon cleanup
# =========================
def is_bad_emoticon_context(text: str, start: int, end: int) -> bool:
    """
    Filter false-positive emoticons like:
    - 36.7 %)
    - 50%)
    - 12:)
    """
    window = text[max(0, start - 6): min(len(text), end + 6)]

    if re.search(r"\d+(?:\.\d+)?\s*%\)", window):
        return True

    if re.search(r"\d+(?:\.\d+)?\s*:\)", window):
        return True

    if re.search(r"\d+(?:\.\d+)?\s*:-\)", window):
        return True

    left = text[start - 1] if start > 0 else ""
    if left.isdigit() or left == "%":
        return True

    return False


def extract_emojis_with_placeholders(text):
    if text is None:
        return ""

    text = str(text)
    found = []

    # --- detect unicode emojis ---
    emoji_info = emo.emoji(text)
    if emoji_info and "value" in emoji_info and "location" in emoji_info:
        for mean, loc in zip(emoji_info["mean"], emoji_info["location"]):
            label = f"*{mean}*" if mean else ""
            found.append({
                "start": loc[0],
                "end": loc[1],
                "label": label,
            })

    # --- detect emoticons ---
    emoticon_info = emo.emoticons(text)
    if emoticon_info and "mean" in emoticon_info and "location" in emoticon_info:
        for mean, loc in zip(emoticon_info["mean"], emoticon_info["location"]):
            if is_bad_emoticon_context(text, loc[0], loc[1]):
                continue
            label = f"*{mean}*" if mean else ""
            found.append({
                "start": loc[0],
                "end": loc[1],
                "label": label,
            })

    found.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    filtered = []
    occupied_until = -1

    for item in found:
        if item["start"] < occupied_until:
            continue
        filtered.append(item)
        occupied_until = item["end"]

    pieces = []
    last = 0

    for item in filtered:
        start, end = item["start"], item["end"]
        label = item["label"]

        pieces.append(text[last:start])
        pieces.append(f" {label} ")
        last = end

    pieces.append(text[last:])
    new_text = "".join(pieces)

    new_text = re.sub(r"\s+", " ", new_text).strip()
    return new_text
# ==============================
# CONFIG
# ==============================
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SFT_MODEL_PATH = "outputs/sft/final"
RL_MODEL_PATH = "outputs/rl/final"
CLS_MODEL_NAME = "outputs/best_model"

INPUT_COL = "sentiment_text"
TARGET_COL = "intensified_text"

MAX_INPUT_LEN = 128
MAX_NEW_TOKENS = 256
GEN_BATCH_SIZE = 16
CLS_BATCH_SIZE = 32

DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 1.0

OUTPUT_DIR = "eval_outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LM_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# ==============================
# Helpers
# ==============================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_label_name(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    return x


def map_true_label_to_full_id(label_name: str, label2id_dict: dict):
    norm = normalize_label_name(label_name)
    for k, v in label2id_dict.items():
        if normalize_label_name(k) == norm:
            return v
    raise ValueError(f"Label '{label_name}' not found in classifier mapping: {label2id_dict}")


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


# ==============================
# Prompting
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
    parts = full_output.split("<|assistant|>\n")
    if len(parts) > 1:
        return parts[-1].split("</s>")[0].strip()
    return full_output.strip()


# ==============================
# Generation / scoring
# ==============================
def generate_texts(model, tokenizer, inputs, labels, batch_size=8, desc="Generating"):
    generations = []

    for i in tqdm.tqdm(range(0, len(inputs), batch_size), desc=desc):
        batch_inputs = inputs[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        batch_prompts = [
            build_prompt_eval(tokenizer, txt, lbl)
            for txt, lbl in zip(batch_inputs, batch_labels)
        ]

        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
            return_tensors="pt",
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if DO_SAMPLE:
            gen_kwargs["temperature"] = TEMPERATURE
            gen_kwargs["top_p"] = TOP_P

        with torch.no_grad():
            output_ids = model.generate(**enc, **gen_kwargs)

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

        for full_text in decoded:
            clean_resp = extract_response_eval(full_text)
            generations.append(clean_resp if clean_resp else ".")

    return generations


def classifier_predict_full(model, tokenizer, texts, batch_size=32, desc="Classifier scoring"):
    pred_ids = []

    for i in tqdm.tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i:i + batch_size]

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


def classifier_predict_with_probs(model, tokenizer, texts, batch_size=32, desc="Classifier scoring", y_true=None):
    """Return predictions, per-class probabilities, logits, and per-item loss (if y_true provided)."""
    pred_ids = []
    probs_all = []
    logits_all = []
    losses = [] if y_true is not None else None

    for i in tqdm.tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

        pred_ids.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())
        logits_all.extend(logits.cpu().tolist())

        if y_true is not None:
            batch_true = torch.tensor(y_true[i:i + batch_size], dtype=torch.long).to(device)
            with torch.no_grad():
                loss_vals = F.cross_entropy(logits, batch_true, reduction='none')
            losses.extend(loss_vals.cpu().tolist())

    return pred_ids, probs_all, logits_all, losses


# ==============================
# Metrics
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


def compute_rouge_per_item(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_list, rouge2_list, rougeL_list = [], [], []

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1_list.append(scores["rouge1"].fmeasure)
        rouge2_list.append(scores["rouge2"].fmeasure)
        rougeL_list.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": rouge1_list,
        "rouge2": rouge2_list,
        "rougeL": rougeL_list,
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


def compute_bertscore_per_item(references, hypotheses):
    P, R, F1 = bertscore_score(
        hypotheses,
        references,
        lang="en",
        verbose=False,
        device=device.type,
    )

    return {
        "bertscore_precision": [p.item() for p in P],
        "bertscore_recall": [r.item() for r in R],
        "bertscore_f1": [f.item() for f in F1],
    }


def compute_bleu_per_item(references, hypotheses):
    smoothie = SmoothingFunction().method4
    from nltk.translate.bleu_score import sentence_bleu

    bleu_list = []
    for ref, hyp in zip(references, hypotheses):
        try:
            val = sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie)
        except Exception:
            val = 0.0
        bleu_list.append(val)
    return bleu_list

def compute_length_stats(inputs, generated_texts):
    length_ratios = []

    for src, gen in zip(inputs, generated_texts):
        src_len = max(len(src.split()), 1)
        gen_len = len(gen.split())
        length_ratios.append(gen_len / src_len)

    avg_length_ratio = sum(length_ratios) / len(length_ratios)
    return avg_length_ratio


def compute_length_ratio_per_item(inputs, generated_texts):
    ratios = []
    for src, gen in zip(inputs, generated_texts):
        src_len = max(len(src.split()), 1)
        gen_len = len(gen.split())
        ratios.append(gen_len / src_len)
    return ratios


def save_ngram_freq(model_name, texts, n=1):
    # texts: list[str]
    counter = Counter()
    for t in texts:
        tokens = t.split()
        if len(tokens) < n:
            continue
        ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        counter.update(ngrams)

    items = counter.most_common()
    out_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_{n}gram_freq.csv")
    df = pd.DataFrame(items, columns=["ngram", "count"])
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def save_per_item_metrics_csv(model_name, inputs, references, labels, generated_texts, per_item_metrics):
    out_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_per_item_metrics.csv")
    df = pd.DataFrame({
        "input_text": inputs,
        "reference_text": references,
        "generated_text": generated_texts,
        "sentiment": labels,
    })

    for k, v in per_item_metrics.items():
        df[k] = v

    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


# ==============================
# Model loading
# ==============================
def load_generation_tokenizer(adapter_path: str = None):
    """
    Load a tokenizer. If `adapter_path` points to a directory with a tokenizer, prefer that tokenizer
    (so vocab size matches adapter checkpoints). Falls back to the base model tokenizer.
    """
    tokenizer = None

    if adapter_path and os.path.isdir(adapter_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        except Exception:
            tokenizer = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=LM_DTYPE,
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model


def load_adapter_model(adapter_path, tokenizer):
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=LM_DTYPE,
    ).to(device)
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


# ==============================
# Save helpers
# ==============================
def save_model_outputs_csv(model_name, inputs, references, labels, generated_texts):
    out_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_generations.csv")
    df = pd.DataFrame({
        "input_text": inputs,
        "base_text": inputs,
        "reference_text": references,
        "generated_text": generated_texts,
        "sentiment": labels,
    })
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def save_model_metrics_txt(model_name, metrics_text):
    out_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_metrics.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    return out_path


# ==============================
# Evaluation
# ==============================
def evaluate_model(
    model,
    tokenizer,
    model_name,
    inputs,
    labels,
    references,
    sst5_texts,
    cls_model,
    cls_tokenizer,
    y_true,
    neg_idx,
    neu_idx,
    pos_idx,
    id2label,
):
    print(f"\n{'=' * 20} {model_name} {'=' * 20}")

    generated_texts = generate_texts(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        labels=labels,
        batch_size=GEN_BATCH_SIZE,
        desc=f"Generating ({model_name})",
    )

    fixed_generated_texts = [extract_emojis_with_placeholders(x) for x in generated_texts]

    # Compare generated texts against the original input/base text (not the reference)
    bleu = compute_bleu(inputs, generated_texts)
    rouge = compute_rouge(inputs, generated_texts)
    exact_match = compute_exact_match(inputs, generated_texts)
    bertscore = compute_bertscore(inputs, generated_texts)

    # --- per-item metrics (BLEU/ROUGE/BERT between input and generated)
    bleu_list = compute_bleu_per_item(inputs, generated_texts)
    rouge_per = compute_rouge_per_item(inputs, generated_texts)
    bert_per = compute_bertscore_per_item(inputs, generated_texts)
    exact_match_list = [int(normalize_text(inp) == normalize_text(hyp)) for inp, hyp in zip(inputs, generated_texts)]
    length_ratio_list = compute_length_ratio_per_item(inputs, generated_texts)

    per_item_metrics = {
        "bleu": bleu_list,
        "rouge1": rouge_per["rouge1"],
        "rouge2": rouge_per["rouge2"],
        "rougeL": rouge_per["rougeL"],
        "bertscore_precision": bert_per["bertscore_precision"],
        "bertscore_recall": bert_per["bertscore_recall"],
        "bertscore_f1": bert_per["bertscore_f1"],
        "exact_match": exact_match_list,
        "length_ratio": length_ratio_list,
    }

    distinct_1 = distinct_n(generated_texts, n=1)
    distinct_2 = distinct_n(generated_texts, n=2)
    avg_length_ratio = compute_length_stats(inputs, generated_texts)

    y_pred_sent, probs_all, logits_all, losses = classifier_predict_with_probs(
        cls_model,
        cls_tokenizer,
        fixed_generated_texts,
        batch_size=CLS_BATCH_SIZE,
        desc=f"Classifier scoring ({model_name})",
        y_true=y_true,
    )

    all_labels = sorted(id2label.keys())
    sent_acc = accuracy_score(y_true, y_pred_sent)
    sent_macro_f1 = f1_score(y_true, y_pred_sent, average="macro")

    posneg_to_neu = sum(
        1 for t, p in zip(y_true, y_pred_sent)
        if t in [neg_idx, pos_idx] and p == neu_idx
    )
    total_posneg = sum(1 for t in y_true if t in [neg_idx, pos_idx])
    neu_leak_rate = posneg_to_neu / total_posneg if total_posneg > 0 else 0.0

    report_text = classification_report(
        y_true,
        y_pred_sent,
        labels=all_labels,
        target_names=[id2label[i] for i in all_labels],
        digits=4,
    )

    # confusion matrix and predicted counts
    conf = confusion_matrix(y_true, y_pred_sent, labels=all_labels)
    pred_counts = {id2label[i]: y_pred_sent.count(i) for i in all_labels}

    # attach classifier predictions, probabilities and losses to per-item metrics then save per-item CSVs
    per_item_metrics["predicted_sentiment_id"] = y_pred_sent
    per_item_metrics["predicted_sentiment"] = [id2label[i] for i in y_pred_sent]
    # per-class probabilities
    for idx in all_labels:
        col = f"prob_{normalize_label_name(id2label[idx]).replace(' ', '_')}"
        per_item_metrics[col] = [p[idx] for p in probs_all]
    # max prob and per-item loss
    per_item_metrics["predicted_prob_max"] = [max(p) for p in probs_all]
    if losses is not None:
        per_item_metrics["per_item_loss"] = losses

    per_item_csv = save_per_item_metrics_csv(model_name, inputs, references, labels, generated_texts, per_item_metrics)
    uni_csv = save_ngram_freq(model_name, generated_texts, n=1)
    bi_csv = save_ngram_freq(model_name, generated_texts, n=2)

    print(f"Saved per-item metrics to: {per_item_csv}")
    print(f"Saved unigram frequencies to: {uni_csv}")
    print(f"Saved bigram frequencies to: {bi_csv}")

    metrics_text = (
        f"==================== {model_name} ====================\n\n"
        f"===== {model_name} Generative Task Evaluation =====\n"
        f"BLEU:                     {bleu:.4f}\n"
        f"ROUGE-1 F1:               {rouge['rouge1']:.4f}\n"
        f"ROUGE-2 F1:               {rouge['rouge2']:.4f}\n"
        f"ROUGE-L F1:               {rouge['rougeL']:.4f}\n"
        f"BERTScore Precision:      {bertscore['bertscore_precision']:.4f}\n"
        f"BERTScore Recall:         {bertscore['bertscore_recall']:.4f}\n"
        f"BERTScore F1:             {bertscore['bertscore_f1']:.4f}\n"
        f"Exact Match:              {exact_match:.4f}\n"
        f"Distinct-1:               {distinct_1:.4f}\n"
        f"Distinct-2:               {distinct_2:.4f}\n"
        f"Avg Length Ratio:         {avg_length_ratio:.4f}\n"
        f"===== {model_name} Sentiment on Generated Text =====\n"
        f"Sentiment Accuracy:       {sent_acc:.4f}\n"
        f"Sentiment Macro-F1:       {sent_macro_f1:.4f}\n"
        f"Neutral Leakage Rate:     {neu_leak_rate:.4f}\n\n"
        f"Classification Report ({model_name}):\n"
        f"{report_text}\n\n"
        f"Confusion Matrix (rows=true, cols=pred) with labels {all_labels}:\n{conf}\n\n"
        f"Predicted label counts:\n{pred_counts}\n\n"
    )

    print(metrics_text)

    return {
        "model": model_name,
        "generated_texts": generated_texts,
        "metrics_text": metrics_text,
    }


def evaluate_texts(
    model_name,
    inputs,
    labels,
    references,
    generated_texts,
    sst5_texts,
    cls_model,
    cls_tokenizer,
    y_true,
    neg_idx,
    neu_idx,
    pos_idx,
    id2label,
):
    """
    Evaluate already-produced texts (no generation step).
    """
    print(f"\n{'=' * 20} {model_name} {'=' * 20}")

    fixed_generated_texts = [extract_emojis_with_placeholders(x) for x in generated_texts]

    # Compare generated texts against the original input/base text (not the reference)
    bleu = compute_bleu(inputs, generated_texts)
    rouge = compute_rouge(inputs, generated_texts)
    exact_match = compute_exact_match(inputs, generated_texts)
    bertscore = compute_bertscore(inputs, generated_texts)

    # --- per-item metrics (BLEU/ROUGE/BERT between input and generated)
    bleu_list = compute_bleu_per_item(inputs, generated_texts)
    rouge_per = compute_rouge_per_item(inputs, generated_texts)
    bert_per = compute_bertscore_per_item(inputs, generated_texts)
    exact_match_list = [int(normalize_text(inp) == normalize_text(hyp)) for inp, hyp in zip(inputs, generated_texts)]
    length_ratio_list = compute_length_ratio_per_item(inputs, generated_texts)

    per_item_metrics = {
        "bleu": bleu_list,
        "rouge1": rouge_per["rouge1"],
        "rouge2": rouge_per["rouge2"],
        "rougeL": rouge_per["rougeL"],
        "bertscore_precision": bert_per["bertscore_precision"],
        "bertscore_recall": bert_per["bertscore_recall"],
        "bertscore_f1": bert_per["bertscore_f1"],
        "exact_match": exact_match_list,
        "length_ratio": length_ratio_list,
    }

    distinct_1 = distinct_n(generated_texts, n=1)
    distinct_2 = distinct_n(generated_texts, n=2)
    avg_length_ratio = compute_length_stats(inputs, generated_texts)

    y_pred_sent, probs_all, logits_all, losses = classifier_predict_with_probs(
        cls_model,
        cls_tokenizer,
        fixed_generated_texts,
        batch_size=CLS_BATCH_SIZE,
        desc=f"Classifier scoring ({model_name})",
        y_true=y_true,
    )

    all_labels = sorted(id2label.keys())
    sent_acc = accuracy_score(y_true, y_pred_sent)
    sent_macro_f1 = f1_score(y_true, y_pred_sent, average="macro")

    posneg_to_neu = sum(
        1 for t, p in zip(y_true, y_pred_sent)
        if t in [neg_idx, pos_idx] and p == neu_idx
    )
    total_posneg = sum(1 for t in y_true if t in [neg_idx, pos_idx])
    neu_leak_rate = posneg_to_neu / total_posneg if total_posneg > 0 else 0.0

    report_text = classification_report(
        y_true,
        y_pred_sent,
        labels=all_labels,
        target_names=[id2label[i] for i in all_labels],
        digits=4,
    )

    # confusion matrix and predicted counts
    conf = confusion_matrix(y_true, y_pred_sent, labels=all_labels)
    pred_counts = {id2label[i]: y_pred_sent.count(i) for i in all_labels}

    # attach classifier predictions, probabilities and losses to per-item metrics then save per-item CSVs
    per_item_metrics["predicted_sentiment_id"] = y_pred_sent
    per_item_metrics["predicted_sentiment"] = [id2label[i] for i in y_pred_sent]
    for idx in all_labels:
        col = f"prob_{normalize_label_name(id2label[idx]).replace(' ', '_')}"
        per_item_metrics[col] = [p[idx] for p in probs_all]
    per_item_metrics["predicted_prob_max"] = [max(p) for p in probs_all]
    if losses is not None:
        per_item_metrics["per_item_loss"] = losses

    per_item_csv = save_per_item_metrics_csv(model_name, inputs, references, labels, generated_texts, per_item_metrics)
    uni_csv = save_ngram_freq(model_name, generated_texts, n=1)
    bi_csv = save_ngram_freq(model_name, generated_texts, n=2)

    print(f"Saved per-item metrics to: {per_item_csv}")
    print(f"Saved unigram frequencies to: {uni_csv}")
    print(f"Saved bigram frequencies to: {bi_csv}")

    metrics_text = (
        f"==================== {model_name} ====================\n\n"
        f"===== {model_name} Generative Task Evaluation =====\n"
        f"BLEU:                     {bleu:.4f}\n"
        f"ROUGE-1 F1:               {rouge['rouge1']:.4f}\n"
        f"ROUGE-2 F1:               {rouge['rouge2']:.4f}\n"
        f"ROUGE-L F1:               {rouge['rougeL']:.4f}\n"
        f"BERTScore Precision:      {bertscore['bertscore_precision']:.4f}\n"
        f"BERTScore Recall:         {bertscore['bertscore_recall']:.4f}\n"
        f"BERTScore F1:             {bertscore['bertscore_f1']:.4f}\n"
        f"Exact Match:              {exact_match:.4f}\n"
        f"Distinct-1:               {distinct_1:.4f}\n"
        f"Distinct-2:               {distinct_2:.4f}\n"
        f"Avg Length Ratio:         {avg_length_ratio:.4f}\n"
        f"===== {model_name} Sentiment on Generated Text =====\n"
        f"Sentiment Accuracy:       {sent_acc:.4f}\n"
        f"Sentiment Macro-F1:       {sent_macro_f1:.4f}\n"
        f"Neutral Leakage Rate:     {neu_leak_rate:.4f}\n\n"
        f"Classification Report ({model_name}):\n"
        f"{report_text}\n"
    )

    print(metrics_text)

    return {
        "model": model_name,
        "generated_texts": generated_texts,
        "metrics_text": metrics_text,
    }


# ==============================
# Main
# ==============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # No generation models loaded — only evaluating input (base) and reference texts.

    cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
    cls_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME).to(device)
    cls_model.eval()

    id2label = {int(k): v for k, v in cls_model.config.id2label.items()}
    label2id = {v.lower(): int(k) for k, v in cls_model.config.id2label.items()}
    neg_idx, neu_idx, pos_idx = find_sentiment_indices(id2label)

    df = pd.read_csv("../data/test.csv").dropna(subset=[INPUT_COL, TARGET_COL])

    tmp = df[INPUT_COL].progress_apply(extract_clean_text)
    df[["clean_input", "label_name"]] = pd.DataFrame(tmp.tolist(), index=df.index)
    df = df.dropna(subset=["clean_input", TARGET_COL, "label_name"]).reset_index(drop=True)

    inputs = df["clean_input"].tolist()
    references = df[TARGET_COL].tolist()
    labels = df["label_name"].tolist()
    y_true = [map_true_label_to_full_id(l, label2id) for l in labels]

    sst5 = load_dataset("SetFit/sst5", split="test")
    sst5_texts = sst5["text"]

    # Evaluate base text (original inputs) and reference text metrics first
    base_result = evaluate_texts(
        model_name="INPUT",
        inputs=inputs,
        labels=labels,
        references=references,
        generated_texts=inputs,
        sst5_texts=sst5_texts,
        cls_model=cls_model,
        cls_tokenizer=cls_tokenizer,
        y_true=y_true,
        neg_idx=neg_idx,
        neu_idx=neu_idx,
        pos_idx=pos_idx,
        id2label=id2label,
    )

    csv_path = save_model_outputs_csv(
        model_name="INPUT",
        inputs=inputs,
        references=references,
        labels=labels,
        generated_texts=base_result["generated_texts"],
    )
    txt_path = save_model_metrics_txt(
        model_name="INPUT",
        metrics_text=base_result["metrics_text"],
    )
    print(f"Saved INPUT (original inputs) generations to: {csv_path}")
    print(f"Saved INPUT metrics to: {txt_path}")

    ref_result = evaluate_texts(
        model_name="REFERENCE_TEXT",
        inputs=inputs,
        labels=labels,
        references=references,
        generated_texts=references,
        sst5_texts=sst5_texts,
        cls_model=cls_model,
        cls_tokenizer=cls_tokenizer,
        y_true=y_true,
        neg_idx=neg_idx,
        neu_idx=neu_idx,
        pos_idx=pos_idx,
        id2label=id2label,
    )

    csv_path = save_model_outputs_csv(
        model_name="REFERENCE_TEXT",
        inputs=inputs,
        references=references,
        labels=labels,
        generated_texts=ref_result["generated_texts"],
    )
    txt_path = save_model_metrics_txt(
        model_name="REFERENCE",
        metrics_text=ref_result["metrics_text"],
    )
    print(f"Saved REFERENCE generations to: {csv_path}")
    print(f"Saved REFERENCE metrics to: {txt_path}")

    # --- Evaluate Base (unadapted) model ---
    try:
        tokenizer = load_generation_tokenizer()
        print("Loading Base model...")
        base_model = load_base_model(tokenizer)
        base_model_result = evaluate_model(
            model=base_model,
            tokenizer=tokenizer,
            model_name="BASE_MODEL",
            inputs=inputs,
            labels=labels,
            references=references,
            sst5_texts=sst5_texts,
            cls_model=cls_model,
            cls_tokenizer=cls_tokenizer,
            y_true=y_true,
            neg_idx=neg_idx,
            neu_idx=neu_idx,
            pos_idx=pos_idx,
            id2label=id2label,
        )

        csv_path = save_model_outputs_csv(
            model_name="BASE_MODEL",
            inputs=inputs,
            references=references,
            labels=labels,
            generated_texts=base_model_result["generated_texts"],
        )
        txt_path = save_model_metrics_txt(
            model_name="BASE_MODEL",
            metrics_text=base_model_result["metrics_text"],
        )
        print(f"Saved BASE_MODEL generations to: {csv_path}")
        print(f"Saved BASE_MODEL metrics to: {txt_path}")

        del base_model
        clear_memory()
    except Exception as e:
        print(f"Failed to evaluate Base model: {e}")

    # --- Evaluate SFT model (adapter) ---
    try:
        # prefer tokenizer saved with the adapter so vocab sizes match
        tokenizer = load_generation_tokenizer(adapter_path=SFT_MODEL_PATH)
        print("Loading SFT adapter model...")
        sft_model = load_adapter_model(SFT_MODEL_PATH, tokenizer)
        sft_result = evaluate_model(
            model=sft_model,
            tokenizer=tokenizer,
            model_name="SFT",
            inputs=inputs,
            labels=labels,
            references=references,
            sst5_texts=sst5_texts,
            cls_model=cls_model,
            cls_tokenizer=cls_tokenizer,
            y_true=y_true,
            neg_idx=neg_idx,
            neu_idx=neu_idx,
            pos_idx=pos_idx,
            id2label=id2label,
        )

        csv_path = save_model_outputs_csv(
            model_name="SFT",
            inputs=inputs,
            references=references,
            labels=labels,
            generated_texts=sft_result["generated_texts"],
        )
        txt_path = save_model_metrics_txt(
            model_name="SFT",
            metrics_text=sft_result["metrics_text"],
        )
        print(f"Saved SFT generations to: {csv_path}")
        print(f"Saved SFT metrics to: {txt_path}")

        del sft_model
        clear_memory()
    except Exception as e:
        print(f"Failed to evaluate SFT model: {e}")

    # --- Evaluate RL model (adapter) ---
    try:
        # prefer tokenizer saved with the adapter so vocab sizes match
        tokenizer = load_generation_tokenizer(adapter_path=RL_MODEL_PATH)
        print("Loading RL adapter model...")
        rl_model = load_adapter_model(RL_MODEL_PATH, tokenizer)
        rl_result = evaluate_model(
            model=rl_model,
            tokenizer=tokenizer,
            model_name="RL",
            inputs=inputs,
            labels=labels,
            references=references,
            sst5_texts=sst5_texts,
            cls_model=cls_model,
            cls_tokenizer=cls_tokenizer,
            y_true=y_true,
            neg_idx=neg_idx,
            neu_idx=neu_idx,
            pos_idx=pos_idx,
            id2label=id2label,
        )

        csv_path = save_model_outputs_csv(
            model_name="RL",
            inputs=inputs,
            references=references,
            labels=labels,
            generated_texts=rl_result["generated_texts"],
        )
        txt_path = save_model_metrics_txt(
            model_name="RL",
            metrics_text=rl_result["metrics_text"],
        )
        print(f"Saved RL generations to: {csv_path}")
        print(f"Saved RL metrics to: {txt_path}")

        del rl_model
        clear_memory()
    except Exception as e:
        print(f"Failed to evaluate RL model: {e}")

    del cls_model
    clear_memory()

    print("\nDone.")


if __name__ == "__main__":
    main()