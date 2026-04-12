import pandas as pd
import re
from emot import emot
from spellchecker import SpellChecker

input_df = pd.read_csv('original_train.csv', encoding='latin1')
final_df = input_df[['text', 'sentiment']]

df = final_df.where(pd.notnull(final_df), None)


sentiment_counts = df['sentiment'].value_counts()
sentiment_pct = df['sentiment'].value_counts(normalize=True) * 100
print("Sentiment Counts")
print(sentiment_counts)

df['text_len'] = df['text'].str.len()
print()
print("Text Lengths")
print(df['text_len'].describe())
print("Lengths < 30:", len(df[df["text_len"] < 30]))





emo = emot()

extra_emoticon_pattern = re.compile(
    r'(?:(?:[><][\-_][><])|(?:[-_.]{2,})|(?:T_+T)|(?:\^_+\^)|(?:>_+<)|(?:>_+>)|(?:<_+<)|(?:<3)|(?:</3)|(?:[:=;][\-]?[)(DPpOo/\\]+))'
)

def is_bad_emoticon_context(text: str, start: int, end: int) -> bool:
    """
    Filter false-positive emoticons like:
    - 36.7 %)
    - 50%)
    - 12:)
    """
    window = text[max(0, start - 6): min(len(text), end + 6)]

    # examples: 36.7 %), 50%), 20 :)
    if re.search(r"\d+(?:\.\d+)?\s*%\)", window):
        return True

    if re.search(r"\d+(?:\.\d+)?\s*:\)", window):
        return True

    if re.search(r"\d+(?:\.\d+)?\s*:-\)", window):
        return True

    if len(set(text[start:end])) <= 1:
        return True

    # immediate left-char heuristic
    left = text[start - 1] if start > 0 else ""
    if left.isdigit() or left == "%":
        return True

    return False

def extract_emojis_with_placeholders(text):
    global emoji_cnt
    if text is None:
        return "", {}

    text = str(text)
    found = []

    # --- detect unicode emojis with emot ---
    emoji_info = emo.emoji(text)
    if emoji_info and "value" in emoji_info and "location" in emoji_info:
        for val, loc in zip(emoji_info["value"], emoji_info["location"]):
            found.append({
                "start": loc[0],
                "end": loc[1],
                "raw": val,
            })

    # --- detect emoticons with emot ---
    emoticon_info = emo.emoticons(text)
    if emoticon_info and "value" in emoticon_info and "location" in emoticon_info:
        for val, loc in zip(emoticon_info["value"], emoticon_info["location"]):
            if is_bad_emoticon_context(text, loc[0], loc[1]):
                continue

            found.append({
                "start": loc[0],
                "end": loc[1],
                "raw": val,
            })

    # --- detect extra regex emoticons ---
    for match in extra_emoticon_pattern.finditer(text):
        start, end = match.start(), match.end()

        if is_bad_emoticon_context(text, start, end):
            continue

        found.append({
            "start": start,
            "end": end,
            "raw": match.group(),
        })

    # sort by span, then longer match first
    found.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    # --- remove overlaps / duplicates ---
    filtered = []
    occupied_until = -1
    seen_spans = set()

    for item in found:
        span = (item["start"], item["end"])
        if span in seen_spans:
            continue

        if item["start"] < occupied_until:
            continue

        filtered.append(item)
        seen_spans.add(span)
        occupied_until = item["end"]

    placeholders = []
    pieces = []
    last = 0

    for i, item in enumerate(filtered):
        start, end = item["start"], item["end"]

        placeholders.append(item["raw"])

        pieces.append(text[last:start])
        last = end

    pieces.append(text[last:])
    new_text = "".join(pieces)
    new_text = re.sub(r"\s+", " ", new_text).strip()

    return new_text, placeholders


def pad_punctuation(text):
    text = re.sub(r'([^\w\s\']+)', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

spell = SpellChecker()

def perform_corpus_analysis(df, text_column='text'):
    # --- Patterns ---
    url_pattern = r'https?\s*:\s*/\s*/\s*\S+|www[\.,]\S+|www\.\S+'
    user_pattern = r'(?<!\w)(?:@\s*[A-Za-z0-9_]+|_+[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_*)'
    tag_pattern = r'#\S+'
    repeat_punct_pattern = r'([^\w\s*])\1{2,}'

    results = {
        "URLs": {"matches": 0, "rows": 0, "unique_set": set()},
        "Users": {"matches": 0, "rows": 0, "unique_set": set()},
        "Hashtags": {"matches": 0, "rows": 0, "unique_set": set()},
        "Repeated Punct": {"matches": 0, "rows": 0, "unique_set": set()},
        "Emoticons": {"matches": 0, "rows": 0, "unique_set": set()},
        "Uppercase": {"matches": 0, "rows": 0, "unique_set": set()},
        "Edit-1 Typos": {"matches": 0, "rows": 0, "unique_set": set()},
        "Typos": {"matches": 0, "rows": 0, "unique_set": set()}
    }

    total_words = 0

    for raw_text in df[text_column].astype(str):
        text = raw_text.replace("`", "'")

        masked_text = re.sub(url_pattern, " URLTOKEN ", text)
        masked_text = re.sub(user_pattern, " USERTOKEN ", masked_text)
        masked_text = re.sub(tag_pattern, " TAGTOKEN ", masked_text)

        words = masked_text.split()
        total_words += len(words)

        for cat in ["URLs", "Users", "Hashtags"]:
            pat = url_pattern if cat == "URLs" else (user_pattern if cat == "Users" else tag_pattern)
            found = re.findall(pat, text)
            if found:
                results[cat]["matches"] += len(found)
                results[cat]["rows"] += 1
                results[cat]["unique_set"].update(found)

        repeats = [m.group() for m in re.finditer(repeat_punct_pattern, masked_text)]
        if repeats:
            results["Repeated Punct"]["matches"] += len(repeats)
            results["Repeated Punct"]["rows"] += 1
            results["Repeated Punct"]["unique_set"].update(repeats)

        masked_text, emoticons_found = extract_emojis_with_placeholders(masked_text)

        if emoticons_found:
            results["Emoticons"]["matches"] += len(emoticons_found)
            results["Emoticons"]["rows"] += 1
            results["Emoticons"]["unique_set"].update(emoticons_found)

        clean_for_typos = masked_text.replace("URLTOKEN", "").replace("USERTOKEN", "").replace("TAGTOKEN", "")
        all_potential = re.findall(r"\b[A-Za-z']+\b", clean_for_typos)

        row_has_edit1 = False
        row_has_typo = False

        for w in all_potential:
            if len(w) <= 1:
                continue

            is_misspelled = w.lower() in spell.unknown([w.lower()])

            if w.isupper() and not is_misspelled:
                results["Uppercase"]["unique_set"].add(w)
                results["Uppercase"]["matches"] += 1
            elif is_misspelled:
                dist1_candidates = spell.edit_distance_1(w.lower())
                valid_dist1 = spell.known(dist1_candidates)

                if valid_dist1:
                    target_cat = "Edit-1 Typos"
                    row_has_edit1 = True
                    correction = spell.correction(w.lower())
                    display_word = f"{w}->{correction}"
                    results[target_cat]["unique_set"].add(display_word)
                    results[target_cat]["matches"] += 1

                target_cat = "Typos"
                row_has_typo = True
                results[target_cat]["unique_set"].add(w.lower())
                results[target_cat]["matches"] += 1

        if row_has_edit1: results["Edit-1 Typos"]["rows"] += 1
        if row_has_typo: results["Typos"]["rows"] += 1

        if any(w.isupper() and (w.lower() not in spell.unknown([w.lower()])) for w in all_potential if len(w)>1):
            results["Uppercase"]["rows"] += 1


    total_rows = len(df)
    header = f"{'Category':<15} | {'Matches':<8} | {'Unique':<8} | {'Match/Word%':<11} | {'Row %':<7}"
    print()
    print(header)
    print("-" * len(header))

    for cat, data in results.items():
        unique_count = len(data["unique_set"])
        match_ratio = (data["matches"] / total_words) * 100 if total_words > 0 else 0
        row_ratio = (data["rows"] / total_rows) * 100 if total_rows > 0 else 0
        print(f"{cat:<15} | {data['matches']:<8} | {unique_count:<8} | {match_ratio:>10.2f}% | {row_ratio:>6.2f}%")


    header = f"{'Category':<15} | {'Examples':<55}"
    print()
    print(header)
    print("-" * len(header))

    for cat, data in results.items():
        ex_list = list(data["unique_set"])
        examples = ex_list[0]
        i = 1
        while True:
            if len(examples) + len(ex_list[i]) + 2 > 55:
                break
            examples += ', ' + ex_list[i]
            i+=1
        print(f"{cat:<15} | {examples:<55}")

perform_corpus_analysis(df, 'text')