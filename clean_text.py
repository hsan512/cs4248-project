import re
from emot import emot
import ftfy

emo = emot()

extra_emoticon_pattern = re.compile(
    r'(?:(?:[><][\-_][><])|(?:[-_.]{2,})|(?:T_+T)|(?:\^_+\^)|(?:>_+<)|(?:>_+>)|(?:<_+<)|(?:<3)|(?:</3)|(?:[:=;][\-]?[)(DPpOo/\\]+))'
)

def extract_emojis_with_placeholders(text):

    if text is None:
        return "", {}

    text = str(text)

    found = []

    # --- detect emojis with emot ---
    emoji_info = emo.emoji(text)
    if emoji_info and "value" in emoji_info and "location" in emoji_info:
        for sym, loc in zip(emoji_info["value"], emoji_info["location"]):
            found.append((loc[0], loc[1], sym))

    # --- detect emoticons with emot ---
    emoticon_info = emo.emoticons(text)
    if emoticon_info and "value" in emoticon_info and "location" in emoticon_info:
        for sym, loc in zip(emoticon_info["value"], emoticon_info["location"]):
            found.append((loc[0], loc[1], sym))

    # --- detect missing emoticons with regex ---
    for match in extra_emoticon_pattern.finditer(text):
        found.append((match.start(), match.end(), match.group()))

    # sort by location
    found.sort(key=lambda x: x[0])

    placeholders = {}
    pieces = []
    last = 0

    for i, (start, end, sym) in enumerate(found):
        key = f"EMO_TOKEN_{i}"
        placeholders[key] = sym

        pieces.append(text[last:start])
        pieces.append(f" {key} ")

        last = end

    pieces.append(text[last:])
    new_text = "".join(pieces)

    return new_text, placeholders

def restore_emojis(text, emo_map):
    for key, sym in emo_map.items():
        text = text.replace(key, sym)
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

    tag_pattern = r'#\d+'

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


def clean_text(text):
    text = ftfy.fix_text(str(text))
    text = text.strip()

    # preserve urls, usernames, emojis/emoticons
    text, urls, users, hashtags, emo = extract_urls_usernames_symbols(text)

    # other cleaning
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r'\(\s*\)', ' ', text)
    text = re.sub(r'^[().,\[\]]+', '', text)
    text = text.strip()

    # restore the emojis
    if len(emo) != 0:
        text = restore_emojis(text, emo)

    return text, urls, users, hashtags


def preprocess_pipeline(text):
    cleaned_text, urls, users, hashtags = clean_text(text)
    return cleaned_text, urls, users, hashtags