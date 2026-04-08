import gc
import os
import re
import torch
import pandas as pd
import tqdm
from datasets import load_dataset

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
from emot import emot

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

def compute_length_stats(inputs, generated_texts):
    length_ratios = []

    for src, gen in zip(inputs, generated_texts):
        src_len = max(len(src.split()), 1)
        gen_len = len(gen.split())
        length_ratios.append(gen_len / src_len)

    avg_length_ratio = sum(length_ratios) / len(length_ratios)
    return avg_length_ratio


# ==============================
# Model loading
# ==============================
def load_generation_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
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

    bleu = compute_bleu(references, generated_texts)
    rouge = compute_rouge(references, generated_texts)
    exact_match = compute_exact_match(references, generated_texts)
    bertscore = compute_bertscore(references, generated_texts)

    distinct_1 = distinct_n(generated_texts, n=1)
    distinct_2 = distinct_n(generated_texts, n=2)
    avg_length_ratio = compute_length_stats(inputs, generated_texts)

    y_pred_sent = classifier_predict_full(
        cls_model,
        cls_tokenizer,
        fixed_generated_texts,
        batch_size=CLS_BATCH_SIZE,
        desc=f"Classifier scoring ({model_name})",
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

    gen_tokenizer = load_generation_tokenizer()

    cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
    cls_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME).to(device)
    cls_model.eval()

    id2label = {int(k): v for k, v in cls_model.config.id2label.items()}
    label2id = {v.lower(): int(k) for k, v in cls_model.config.id2label.items()}
    neg_idx, neu_idx, pos_idx = find_sentiment_indices(id2label)

    df = pd.read_csv("data/test.csv").dropna(subset=[INPUT_COL, TARGET_COL])

    tmp = df[INPUT_COL].progress_apply(extract_clean_text)
    df[["clean_input", "label_name"]] = pd.DataFrame(tmp.tolist(), index=df.index)
    df = df.dropna(subset=["clean_input", TARGET_COL, "label_name"]).reset_index(drop=True)

    inputs = df["clean_input"].tolist()
    references = df[TARGET_COL].tolist()
    labels = df["label_name"].tolist()
    y_true = [map_true_label_to_full_id(l, label2id) for l in labels]

    sst5 = load_dataset("SetFit/sst5", split="test")
    sst5_texts = sst5["text"]

    model_specs = [
        ("BASE", None),
        ("SFT", SFT_MODEL_PATH),
        ("RL", RL_MODEL_PATH),
    ]

    for model_name, adapter_path in model_specs:
        print(f"\nLoading {model_name} model...")

        if adapter_path is None:
            model = load_base_model(gen_tokenizer)
        else:
            model = load_adapter_model(adapter_path, gen_tokenizer)

        result = evaluate_model(
            model=model,
            tokenizer=gen_tokenizer,
            model_name=model_name,
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
            model_name=model_name,
            inputs=inputs,
            references=references,
            labels=labels,
            generated_texts=result["generated_texts"],
        )
        txt_path = save_model_metrics_txt(
            model_name=model_name,
            metrics_text=result["metrics_text"],
        )

        print(f"Saved {model_name} generations to: {csv_path}")
        print(f"Saved {model_name} metrics to: {txt_path}")

        del model
        clear_memory()

    del cls_model
    clear_memory()

    print("\nDone.")


if __name__ == "__main__":
    main()