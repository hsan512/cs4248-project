import re
from emot import emot
import ftfy

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

    placeholders = {}
    pieces = []
    last = 0

    for i, item in enumerate(filtered):
        start, end = item["start"], item["end"]
        key = f"EMO_TOKEN_{i}"

        placeholders[key] = item["raw"]

        pieces.append(text[last:start])
        pieces.append(f" {key} ")
        last = end

    pieces.append(text[last:])
    new_text = "".join(pieces)
    new_text = re.sub(r"\s+", " ", new_text).strip()

    return new_text, placeholders


def restore_emojis(text, emo_map, delete = False):
    for key, raw in emo_map.items():
        if delete:
            text = text.replace(key, "")
        else:
            text = text.replace(key, raw)
    return text

def extract_urls_usernames_symbols(text):
    """
    Replace URLs, usernames, emojis, and emoticons with unique placeholders.
    Returns:
        new_text: text with placeholders
        placeholders: dict for restoration
    """
    if text is None:
        return "", {}

    text = str(text)
    urls = []
    users = []
    hashtags = []

    # ---------- extract URLs ----------
    url_pattern = r'https?\s*:\s*/\s*/\s*\S+|www[\.,]\S+|www\.\S+'

    def replace_url(match):
        original = match.group(0)
        key = "<URL>"
        urls.append(original)
        return f" {key} "

    text = re.sub(url_pattern, replace_url, text)

    # ---------- extract emoticons ----------
    text, emo = extract_emojis_with_placeholders (text)

    # ---------- extract usernames ----------
    user_pattern = r'(?<!\w)(?:@\s*[A-Za-z0-9_]+|_+[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_*)'

    def replace_user(match):
        original = match.group(0)
        key = "<USER>"
        users.append(original)
        return f" {key} "

    text = re.sub(user_pattern, replace_user, text)

    tag_pattern = r'#\S+'

    def replace_hashtag(match):
        original = match.group(0)
        key = "<TAG>"
        users.append(original)
        return f" {key} "

    text = re.sub(tag_pattern, replace_hashtag, text)

    return text, urls, users, hashtags, emo



def restore_placeholders(text, urls, users, hashtags):
    for url in urls:
        text = text.replace("<URL>", url, 1)

    for user in users:
        text = text.replace("<USER>", user, 1)

    for tag in hashtags:
        text = text.replace("<TAG>", tag, 1)

    return text

def normalise_casing(text, tag = False, allLower = False):
    tokens = []
    for token in text.split():
        if token.isupper() and not allLower:
            if tag:
                tokens.append(token.lower() + "_upper")
            else:
                tokens.append(token)
        else:
            tokens.append(token.lower())
    return ' '.join(tokens)

def dedup_punctuation(text):
    return re.sub(r'([^\w\s])\1{2,}', r'\1\1', text)

def clean_text(text):
    # text = ftfy.fix_text(str(text))
    text = text.strip()

    # preserve urls, usernames, emojis/emoticons
    text, urls, users, hashtags, emo = extract_urls_usernames_symbols(text)

    # Keep original casing or... (Comment all below)
    text = normalise_casing(text) # lower case everything except all upper
    # text = normalise_casing(text, True) # lower case everything but upper add _upper tag
    # text = normalise_casing(text, True, True) # lower case everything

    # other cleaning
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r'\(\s*\)', ' ', text)
    text = re.sub(r'^[().,\[\]]+', '', text)
    text = text.strip()

    text = dedup_punctuation(text)

    # restore the emojis
    if len(emo) != 0:
        text = restore_emojis(text, emo)
        # text = restore_emojis(text, emo, True) # This deletes emojis instead

    return text, urls, users, hashtags



def preprocess_pipeline(text):
    cleaned_text, urls, users, hashtags = clean_text(text)

    return cleaned_text, urls, users, hashtags
