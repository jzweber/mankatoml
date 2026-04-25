"""
Generate synthetic bearish Amazon ($AMZN) social media posts using LM Studio.
Posts are informal, bombastic, and energetic — matching real StockTwits style.
A random sample of existing bearish posts is included in each system prompt for variability.
Messages are restarted each batch to avoid hitting the 4098 token limit.
Output is a CSV matching the phase 2 processed_data format.
"""

import os
import json
import gc
import random
import time
import pandas as pd
from openai import OpenAI

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
AMZN_CSV = r"D:\OneDrive - MNSCU\OneDrive - Minnesota State\Documents\Machine Learning\Project\phase 2\processed_data\AMZN_combined.csv"
EXAMPLES_CSV = r"D:\OneDrive - MNSCU\OneDrive - Minnesota State\Documents\Machine Learning\Project\phase 5\bearish_examples_amzn.csv"
OUTPUT_CSV = r"D:\OneDrive - MNSCU\OneDrive - Minnesota State\Documents\Machine Learning\Project\phase 5\generated_bearish_amzn.csv"

EXAMPLE_POOL_SIZE = 10000  # rows to extract from existing data

TARGET_ROWS = 139812
BATCH_SIZE = 25        # keep batches small so we stay well under 4098 tokens
FLUSH_EVERY = 100
GC_EVERY_BATCHES = 5
MAX_CONSECUTIVE_FAILS = 10
EXAMPLE_COUNT = 8      # how many real posts to include as examples per batch

# ─── STEP 1: EXTRACT 10,000 BEARISH ROWS INTO A SEPARATE CSV ────────────────────
os.makedirs(os.path.dirname(EXAMPLES_CSV), exist_ok=True)

if os.path.exists(EXAMPLES_CSV):
    print(f"Examples CSV already exists: {EXAMPLES_CSV}")
    print("Loading existing examples file (delete it to re-extract).")
    examples_df = pd.read_csv(EXAMPLES_CSV)
else:
    print("Loading full AMZN dataset to extract bearish examples...")
    df = pd.read_csv(AMZN_CSV, usecols=["body", "entities"])
    bearish_df = df[df["entities"].str.contains("Bearish", na=False)].copy()
    # Clean up: remove NaN, strip whitespace, drop very short posts
    bearish_df = bearish_df[bearish_df["body"].notna()]
    bearish_df["body"] = bearish_df["body"].astype(str).str.strip()
    bearish_df = bearish_df[bearish_df["body"].str.len() > 20]

    # Sample 10,000 random bearish rows
    examples_df = bearish_df.sample(n=min(EXAMPLE_POOL_SIZE, len(bearish_df)), random_state=42)
    examples_df.to_csv(EXAMPLES_CSV, index=False)
    print(f"Extracted {len(examples_df)} bearish rows -> {EXAMPLES_CSV}")

    # Free the big dataframe
    del df, bearish_df
    gc.collect()

# Load the example pool for sampling during generation
existing_posts = examples_df["body"].tolist()
print(f"Example pool ready: {len(existing_posts)} posts.")
del examples_df
gc.collect()

# ─── LM STUDIO CLIENT ──────────────────────────────────────────────────────────
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# ─── BEARISH ENTITY STRING (matches existing CSV format) ────────────────────────
BEARISH_ENTITY = "{'sentiment': {'basic': 'Bearish'}}"


def build_system_prompt(examples: list[str]) -> str:
    """Build a fresh system prompt with random example posts for diversity."""
    example_block = "\n".join(f"  - \"{ex[:200]}\"" for ex in examples)

    return f"""You are generating synthetic social media posts about Amazon stock ($AMZN) with a BEARISH sentiment. These posts should sound like they come from real traders on StockTwits or Twitter/X.

STYLE RULES:
- Be informal, bombastic, energetic, chaotic — like a real trader ranting on social media
- Use slang, abbreviations, ALL CAPS for emphasis, emojis occasionally, exclamation marks
- Mix up tone: some posts are panicked, some are sarcastic, some are smug "told you so" vibes
- DO NOT be formal or professional. No one talks like a news anchor on StockTwits
- VARY the length — some posts are 5 words, some are 2-3 sentences
- NOT every post needs to say "bearish" — many should express negativity WITHOUT using that word
  - e.g. "this is going to dump hard", "shorting this trash", "who is buying this garbage lmaooo"
- Some posts (~30%) should include the $AMZN ticker tag, but NOT all of them
- A few posts can mention other tickers too ($SPY, $AAPL, $TSLA, etc.) alongside $AMZN
- Some posts can reference Amazon by name instead of the ticker
- Reference real-world events vaguely (earnings, competition, AWS, regulation, Bezos, etc.)

Here are some REAL examples of bearish Amazon posts for reference — use them for tone and style, but DO NOT copy them:
{example_block}

OUTPUT FORMAT:
Return ONLY a valid JSON array of strings. Each string is one social media post. No markdown. No commentary.
Example: ["post 1 text here", "another post text", "yet another one"]"""


def parse_posts(raw_output: str) -> list[str]:
    """Extract posts from the model response."""
    start = raw_output.find("[")
    end = raw_output.rfind("]") + 1
    if start == -1 or end == 0:
        return []

    cleaned = raw_output[start:end]
    posts = json.loads(cleaned)

    # Validate: must be a list of strings
    if not isinstance(posts, list):
        return []

    valid = []
    for p in posts:
        if isinstance(p, str) and len(p.strip()) > 10:
            valid.append(p.strip())
    return valid


# ─── RESUMABLE SETUP ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Check if we already have rows from a previous run
existing_row_count = 0
if os.path.exists(OUTPUT_CSV):
    existing_row_count = sum(1 for _ in open(OUTPUT_CSV, encoding="utf-8")) - 1  # subtract header
    if existing_row_count < 0:
        existing_row_count = 0
    print(f"Found existing output with {existing_row_count} rows. Will RESUME from here.")
else:
    # Write the CSV header for a brand new file
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        header_df = pd.DataFrame(columns=[
            "Unnamed: 0", "id", "body", "created_at", "user", "source",
            "symbols", "mentioned_users", "entities", "links", "likes",
            "conversation", "reshares", "reshare_message", "owned_symbols"
        ])
        header_df.to_csv(f, index=False)
    print("Created new output CSV with header.")

valid_count = existing_row_count
remaining_total = TARGET_ROWS - valid_count

if remaining_total <= 0:
    print(f"\nAlready have {valid_count}/{TARGET_ROWS} rows. Nothing to generate!")
else:
    # ─── MAIN GENERATION LOOP ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" GENERATING {remaining_total} MORE BEARISH AMAZON POSTS")
    print(f" ({valid_count} already done, {TARGET_ROWS} total target)")
    print(f"{'='*60}\n")

    failed_batches = 0
    batch_counter = 0
    next_id = valid_count  # for unique IDs

    while valid_count < TARGET_ROWS:
        remaining = TARGET_ROWS - valid_count
        request_amount = min(BATCH_SIZE, remaining)

        # Sample random existing posts as examples (fresh each batch for variety)
        sample_examples = random.sample(existing_posts, min(EXAMPLE_COUNT, len(existing_posts)))

        # Build a FRESH system prompt each time (this is the "restart" to avoid token buildup)
        system_prompt = build_system_prompt(sample_examples)
        user_prompt = f"Generate exactly {request_amount} unique bearish Amazon social media posts. Output the raw JSON array now."

        print(f"[{valid_count}/{TARGET_ROWS}] Requesting batch of {request_amount}...")

        try:
            response = client.chat.completions.create(
                model="qwen2.5-coder-7b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.95,
                max_tokens=3500,  # leave headroom under 4098
                timeout=120       # 2 min timeout to avoid hanging forever
            )

            raw_output = response.choices[0].message.content.strip()
            posts = parse_posts(raw_output)

            if not posts:
                print("  -> ERROR: No valid posts extracted. Discarding batch.")
                failed_batches += 1
            else:
                # Append this batch directly to the CSV file
                batch_rows = []
                for post in posts:
                    if valid_count >= TARGET_ROWS:
                        break
                    batch_rows.append({
                        "Unnamed: 0": next_id,
                        "id": f"synth_{next_id}",
                        "body": post,
                        "created_at": pd.Timestamp.now().isoformat(),
                        "user": "synthetic",
                        "source": "lm_studio_generated",
                        "symbols": "$AMZN",
                        "mentioned_users": "",
                        "entities": BEARISH_ENTITY,
                        "links": "",
                        "likes": 0,
                        "conversation": "",
                        "reshares": "",
                        "reshare_message": "",
                        "owned_symbols": ""
                    })
                    valid_count += 1
                    next_id += 1

                # Append to CSV (no header, since it already exists)
                batch_df = pd.DataFrame(batch_rows)
                batch_df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

                print(f"  -> Batch yielded {len(batch_rows)} valid posts. Total: {valid_count}/{TARGET_ROWS}")
                failed_batches = 0
                batch_counter += 1

                del batch_rows, batch_df

            # Clean up per-batch objects
            del raw_output, response
            if 'posts' in dir():
                del posts

            if batch_counter % GC_EVERY_BATCHES == 0:
                gc.collect()

        except json.JSONDecodeError:
            print("  -> ERROR: JSON parsing failed. Discarding batch.")
            failed_batches += 1
        except Exception as e:
            print(f"  -> ERROR: {e}")
            failed_batches += 1
            print("  -> Waiting 5s before retry...")
            time.sleep(5)  # longer pause on connection/unexpected errors

        if failed_batches >= MAX_CONSECUTIVE_FAILS:
            print(f"\nCRITICAL: {MAX_CONSECUTIVE_FAILS} consecutive failures. Stopping early.")
            print(f"Saved {valid_count} posts so far. Re-run the script to resume.")
            break

    print(f"\nDone! {valid_count}/{TARGET_ROWS} total bearish posts in:")
    print(f"  {OUTPUT_CSV}")

gc.collect()
