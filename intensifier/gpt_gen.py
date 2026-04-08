import os
import json
import random
import asyncio
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AsyncOpenAI, DefaultAioHttpClient

load_dotenv()

# =========================
# Config
# =========================
INPUT_CSV = "test.csv"
OUTPUT_CSV = "test_.csv"
CHECKPOINT_CSV = "test_.checkpoint.csv"
FAILED_JSONL = "test_.failed.jsonl"

TEXT_COL = "sentiment_text"
OUT_COL = "intensified_text"

MODEL = "gpt-4o"
TEMPERATURE = 0.9
MAX_TOKENS = 120

CONCURRENCY = 8
CHECKPOINT_EVERY = 25
MAX_RETRIES = 8
BASE_SLEEP = 2.0
MAX_SLEEP = 60.0

SYSTEM_PROMPT = """
# You are rewriting tweets into strongly intensified versions.

# STRICT RULES:
# - Do NOT add [POS] or [NEG] token at the start
# - Intensify according to the sentiment token in the input, where [POS] means positive and [NEG] means negative
# - Keep the original meaning and context
# - Keep tweet style (informal, emotional, slightly messy)

# EMOTIONS:
# - You may use ASCII emoticons like :) :( :D :/ :') >:(
# - DO NOT use emojis
# - ONLY use ASCII emoticons

# CONSTRAINTS:
# - Preserve masked swear words like ****
# - Do NOT add explanations
# - Output ONLY the rewritten text

# IMPORTANT:
# - If you accidentally generate emojis, replace them with ASCII emoticons or remove them
# """.strip()

# =========================
# API client
# =========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Put it in your .env or environment.")

# AsyncOpenAI is officially supported; aiohttp backend is also supported for better async concurrency. :contentReference[oaicite:2]{index=2}
client = AsyncOpenAI(
    api_key=api_key,
    http_client=DefaultAioHttpClient(),
    timeout=90.0,      # SDK supports configurable timeouts. :contentReference[oaicite:3]{index=3}
    max_retries=2,     # SDK already retries some transient failures by default. :contentReference[oaicite:4]{index=4}
)

# =========================
# Checkpoint helpers
# =========================
def load_dataframe() -> pd.DataFrame:
    if Path(CHECKPOINT_CSV).exists():
        print(f"Resuming from checkpoint: {CHECKPOINT_CSV}")
        df = pd.read_csv(CHECKPOINT_CSV)
    else:
        df = pd.read_csv(INPUT_CSV)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found. Found: {list(df.columns)}")

    if OUT_COL not in df.columns:
        df[OUT_COL] = ""

    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[OUT_COL] = df[OUT_COL].fillna("").astype(str)
    return df

def save_csv_atomic(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def save_failed_row(idx: int, text: str, error: str) -> None:
    row = {"idx": idx, "text": text, "error": error}
    with open(FAILED_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def is_done(val: str) -> bool:
    return isinstance(val, str) and val.strip() != ""

# =========================
# Error helpers
# =========================
def is_billing_inactive(exc: Exception) -> bool:
    msg = str(exc)
    return "billing_not_active" in msg or "Your account is not active" in msg

def backoff_sleep(attempt: int) -> float:
    base = min(MAX_SLEEP, BASE_SLEEP * (2 ** (attempt - 1)))
    jitter = random.uniform(0, 1.0)
    return base + jitter

# =========================
# Generation
# =========================
async def generate_once(text: str) -> str:
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    out = resp.choices[0].message.content.strip()
    return out

async def generate_with_retry(text: str) -> str:
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await generate_once(text)
        except Exception as exc:
            last_exc = exc

            if is_billing_inactive(exc):
                raise

            sleep_s = backoff_sleep(attempt)
            print(f"\nRetry {attempt}/{MAX_RETRIES} failed: {exc}")
            print(f"Sleeping {sleep_s:.1f}s before retry...")
            await asyncio.sleep(sleep_s)

    raise last_exc

# =========================
# Worker
# =========================
async def worker(name: int, queue: asyncio.Queue, df: pd.DataFrame, lock: asyncio.Lock, pbar: tqdm):
    processed_since_ckpt = 0

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        idx = item
        text = df.at[idx, TEXT_COL]

        try:
            out = await generate_with_retry(text)
        except Exception as exc:
            if is_billing_inactive(exc):
                # save checkpoint immediately and re-raise
                async with lock:
                    save_csv_atomic(df, CHECKPOINT_CSV)
                queue.task_done()
                raise

            out = text
            save_failed_row(idx, text, str(exc))

        async with lock:
            df.at[idx, OUT_COL] = out
            pbar.update(1)
            processed_since_ckpt += 1

            if processed_since_ckpt >= CHECKPOINT_EVERY:
                save_csv_atomic(df, CHECKPOINT_CSV)
                processed_since_ckpt = 0

        queue.task_done()

# =========================
# Main
# =========================
async def main():
    df = load_dataframe()

    pending_idx = [i for i, v in enumerate(df[OUT_COL].tolist()) if not is_done(v)]
    print(f"Rows total: {len(df)}")
    print(f"Rows remaining: {len(pending_idx)}")

    if not pending_idx:
        save_csv_atomic(df, OUTPUT_CSV)
        print(f"Nothing to do. Final file already saved to {OUTPUT_CSV}")
        return

    queue = asyncio.Queue()
    for idx in pending_idx:
        await queue.put(idx)

    for _ in range(CONCURRENCY):
        await queue.put(None)

    lock = asyncio.Lock()

    with tqdm(total=len(pending_idx), desc="Generating", unit="row") as pbar:
        tasks = [asyncio.create_task(worker(i, queue, df, lock, pbar)) for i in range(CONCURRENCY)]

        try:
            await queue.join()
        except Exception:
            # save progress if something blows up
            async with lock:
                save_csv_atomic(df, CHECKPOINT_CSV)
            for t in tasks:
                t.cancel()
            raise

        # propagate worker exceptions if any
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                save_csv_atomic(df, CHECKPOINT_CSV)
                raise r

    save_csv_atomic(df, OUTPUT_CSV)
    print(f"Saved final output to {OUTPUT_CSV}")

    if Path(CHECKPOINT_CSV).exists():
        os.remove(CHECKPOINT_CSV)
        print(f"Removed checkpoint {CHECKPOINT_CSV}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())