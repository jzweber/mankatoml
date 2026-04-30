"""
Generate synthetic bearish Amazon ($AMZN) social media posts using LM Studio.
Posts are informal, bombastic, and energetic — matching real StockTwits style.
A random sample of existing bearish posts is included in each system prompt for variability.
Messages are restarted each batch to avoid hitting the 4098 token limit.
Output is a CSV matching the phase 2 processed_data format.
"""

import os
import json
import re
import gc
import random
import time
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
AMZN_CSV = r"D:\OneDrive - MNSCU\OneDrive - Minnesota State\Documents\Machine Learning\Project\phase 2\processed_data\AMZN_combined.csv"
EXAMPLES_CSV = r"D:\OneDrive - MNSCU\OneDrive - Minnesota State\Documents\Machine Learning\Project\phase 5\bearish_examples_amzn.csv"
OUTPUT_DIR = r"D:\OneDrive - MNSCU\OneDrive - Minnesota State\Documents\Machine Learning\Project\phase 5"

# ─── LM STUDIO CLIENT ──────────────────────────────────────────────────────────
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

BUZZWORDS_FILE = os.path.join(OUTPUT_DIR, "temp_stock_buzzwords.json")

def generate_ai_buzzwords(ticker, company, count=100, min_required=50, max_retries=3):
    """Uses the local model to generate a large unique list of products/buzzwords."""
    system_prompt = "You are a helpful assistant. Output ONLY a valid JSON array of strings."
    user_prompt = (
        f"Generate exactly {count} unique strings representing physical products, "
        f"software services, business segments, or negative financial buzzwords for {company} ({ticker}).\n"
        f"CRITICAL RULES:\n"
        f"1. DO NOT include the words '{company}' or '{ticker}' anywhere in your strings! "
        f"(e.g., write 'margins' instead of '{company} margins', write 'sales' instead of '{ticker} sales').\n"
        f"2. DO NOT output the words 'stock' or 'shares' alone, as they are redundant.\n"
        f"Return ONLY the JSON array, no formatting, no markdown."
    )
    
    for attempt in range(max_retries):
        print(f"Generating ~{count} clean unique products/buzzwords for {company} ({ticker}) [Attempt {attempt+1}/{max_retries}]...")
        try:
            response = client.chat.completions.create(
                model="qwen2.5-coder-7b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7 + (0.1 * attempt),
                max_tokens=2000
            )
            content = response.choices[0].message.content.strip()
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end != 0:
                parsed = json.loads(content[start:end])
                
                unique_dict = {}
                for x in parsed:
                    s = str(x).strip()
                    
                    # Regex to automatically strip out the company name, ticker, and symbols 
                    # so they don't echo repetitively against {TICKER} and {COMPANY} injections
                    s = re.sub(rf"(?i)\b{re.escape(company)}\b", "", s).strip()
                    s = re.sub(rf"(?i)\b{re.escape(ticker)}\b", "", s).strip()
                    s = s.replace("$", "").replace("'", "").strip()
                    
                    # Remove any extra space left behind from regex deletions
                    s = re.sub(r"\s+", " ", s)
                    
                    s_lower = s.lower()
                    # Filter out short fragments or generic/redundant terms
                    if len(s) > 2 and s_lower not in ["stock", "shares", "company", "inc", "corp"] and s_lower not in unique_dict:
                        unique_dict[s_lower] = s

                unique = list(unique_dict.values())
                print(f" -> Generated {len(unique)} clean unique items for {ticker}.")
                
                if len(unique) >= min_required:
                    return unique
                else:
                    print(f" -> Insufficient unique items after cleaning ({len(unique)} < {min_required}). Retrying...")
                    time.sleep(2)
            else:
                print(" -> JSON array could not be found in response. Retrying...")
                time.sleep(2)
        except Exception as e:
            print(f" -> Error generating buzzwords: {e}")
            time.sleep(2)
            
    # Fallback if it fails completely
    print(f" -> Failed to generate {min_required} buzzwords after {max_retries} attempts.")
    return [f"{company} products", "sales", "margins", "growth trajectory"]

def load_or_generate_buzzwords():
    """Loads buzzwords from a temporary JSON file or generates them if missing."""
    if os.path.exists(BUZZWORDS_FILE):
        print(f"Loading buzzwords from {BUZZWORDS_FILE} ...")
        with open(BUZZWORDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
            
    print("Temporary buzzwords file not found. Generating now...")
    buzzwords_dict = {
        "AAPL": generate_ai_buzzwords("AAPL", "Apple"),
        "FB": generate_ai_buzzwords("FB", "Meta"),
        "NVDA": generate_ai_buzzwords("NVDA", "Nvidia"),
        "TSLA": generate_ai_buzzwords("TSLA", "Tesla")
    }
    
    with open(BUZZWORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(buzzwords_dict, f, indent=4)
        
    return buzzwords_dict

# Generate / Load the JSON file
load_or_generate_buzzwords()

def get_random_product(ticker):
    """Pulls a random generated buzzword/product from the temp JSON file."""
    with open(BUZZWORDS_FILE, "r", encoding="utf-8") as f:
        buzzwords_dict = json.load(f)
    return random.choice(buzzwords_dict.get(ticker, [f"{ticker} item"]))

# Pre-generate lists once at startup
STOCK_PROFILES = {
    "AAPL": {"{TICKER}": "$AAPL", "{COMPANY}": "Apple", "{CEO}": "Cook", "{PRODUCT}": lambda: get_random_product("AAPL")},
    "FB":   {"{TICKER}": "$FB", "{COMPANY}": "Meta", "{CEO}": "Zuck", "{PRODUCT}": lambda: get_random_product("FB")},
    "NVDA": {"{TICKER}": "$NVDA", "{COMPANY}": "Nvidia", "{CEO}": "Jensen", "{PRODUCT}": lambda: get_random_product("NVDA")},
    "TSLA": {"{TICKER}": "$TSLA", "{COMPANY}": "Tesla", "{CEO}": "Elon", "{PRODUCT}": lambda: get_random_product("TSLA")}
}

# Add random scenarios to force the LLM into wildly different subsets of "bearish" sentiment
SCENARIOS = [
    "Panic and doom: Acting like the stock is going straight to zero, total capitulation.",
    "Sarcastic and mocking: Laughing at the 'bulls' and 'bagholders' who are buying the dip.",
    "Management hate: Ranting specifically about how poorly the CEO and executives are running things.",
    "Technical analysis / Charts: Dropping informal chart logic like 'broke support', 'head and shoulders', 'moving averages are dead'.",
    "Earnings disappointment: Complaining that growth is entirely dead and the upcoming guidance will be a complete disaster.",
    "Short seller confidence: Smug 'told you so' attitude, bragging about short gains or massive put option positions.",
    "Macro/Market drag: Emphasizing that inflation, interest rates, or the broader market is dragging this garbage stock down."
]

OUTPUT_CSVS = {target: os.path.join(OUTPUT_DIR, f"generated_bearish_{target.lower()}.csv") for target in STOCK_PROFILES.keys()}

EXAMPLE_POOL_SIZE = 10000  # rows to extract from existing data

# INDIVIDUAL TARGETS FOR EACH STOCK
TARGET_COUNTS = {
    "AAPL": 255500,
    "FB": 255500,
    "NVDA": 255500,
    "TSLA": 255500
}

BATCH_SIZE = 25        # Generating small 15-post bursts processes exponentially faster locally
FLUSH_EVERY = 120
GC_EVERY_BATCHES = 5
MAX_CONSECUTIVE_FAILS = 15
EXAMPLE_COUNT = 1      # Reduced from 4: Huge time savings on prompt parsing
MAX_WORKERS = 1        # Single-threaded linear performance

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
# Client is instantiated above for the buzzword generator

# ─── BEARISH ENTITY STRING (matches existing CSV format) ────────────────────────
BEARISH_ENTITY = "{'sentiment': {'basic': 'Bearish'}}"


def build_system_prompt(examples: list[str], scenario: str) -> str:
    """Build a fresh system prompt instructing the model to output raw templates."""
    # Strip AMZN specifics out of the examples to show how templates look
    scrubbed = []
    for ex in examples:
        # Case insensitive replacements and other common variations
        txt = ex.replace("$AMZN", "{TICKER}").replace("Amazon", "{COMPANY}").replace("amzn", "{TICKER}")
        txt = txt.replace("AMZN", "{TICKER}").replace("amazon", "{COMPANY}").replace("Bezos", "{CEO}")
        txt = txt.replace("bezos", "{CEO}").replace("AWS", "{PRODUCT}").replace("aws", "{PRODUCT}")
        scrubbed.append(txt)
    
    example_block = "\n".join(f"  - \"{ex[:200]}\"" for ex in scrubbed)

    return f"""You are generating synthetic social media posts with a BEARISH sentiment. These posts must sound like real traders ranting on StockTwits or Twitter/X.

>>> TONE FOCUS FOR THIS BATCH: {scenario} <<<
Ensure ALL posts in this JSON array lean heavily into this specific sub-genre of bearishness!

CRITICAL INSTRUCTION - USE PLACEHOLDERS:
Instead of specific names, you MUST use the following exact placeholders: 
- {{TICKER}} (e.g. $AAPL)
- {{COMPANY}} (e.g. Apple)
- {{CEO}} (e.g. Tim Cook)
- {{PRODUCT}} (e.g. the iPhone)

Example: "I can't believe {{CEO}} is doing this! Shorting {{TICKER}} because {{COMPANY}} is dead. {{PRODUCT}} is a disaster."
STYLE RULES:
- Be informal, bombastic, energetic, chaotic — like a real trader ranting on social media
- Use slang, abbreviations, ALL CAPS occasionally, emojis occasionally, exclamation marks
- Mix up tone: Panthers, sarcasm, "told you so"
- VARY the length — some posts are 5 words, some are 2-3 sentences.
- NOT every post needs {{PRODUCT}} or {{CEO}}, use them naturally.
- NOT every post needs to say "bearish" — many should express negativity WITHOUT using that word
- e.g. "this is going to dump hard", "shorting this trash", "who is buying this garbage lmaooo"
- A few posts can mention other tickers too alongside the main one, but they should still be negative about the main {{TICKER}}
Here are REAL examples of bearish posts (Notice how they use the {{PLACEHOLDERS}}):
{example_block}

OUTPUT FORMAT:
Return ONLY a valid JSON array of strings. Each string is a template post. No markdown. No commentary."""


def parse_posts(raw_output: str) -> list[str]:
    """Extract posts from the model response, salvaging incomplete JSON using regex."""
    start = raw_output.find("[")
    end = raw_output.rfind("]") + 1
    if start != -1 and end != 0:
        cleaned = raw_output[start:end]
        try:
            posts = json.loads(cleaned)
            if isinstance(posts, list):
                return [p.strip() for p in posts if isinstance(p, str) and len(p.strip()) > 10]
        except json.JSONDecodeError:
            pass # Fallback to regex below
            
    # Fallback: Extract everything in quotes that looks like a post
    # This prevents wasting 1-2 minutes of generation if the model forgets a single comma!
    matches = re.findall(r'"([^"]{10,})"', raw_output)
    
    valid = []
    for m in matches:
        m = m.strip()
        # Ensure it's not a generic json key 
        if m.lower() not in ["role", "content", "system", "user"]:
            valid.append(m)
    
    return valid


# ─── RESUMABLE SETUP ────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

valid_counts = {}

for target_name, out_csv in OUTPUT_CSVS.items():
    if not os.path.exists(out_csv):
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            header_df = pd.DataFrame(columns=[
                "Unnamed: 0", "id", "body", "created_at", "user", "source",
                "symbols", "mentioned_users", "entities", "links", "likes",
                "conversation", "reshares", "reshare_message", "owned_symbols"
            ])
            header_df.to_csv(f, index=False)
        print(f"Created new output CSV for {target_name}.")
        valid_counts[target_name] = 0
    else:
        existing_row_count = sum(1 for _ in open(out_csv, encoding="utf-8")) - 1
        valid_counts[target_name] = max(0, existing_row_count)

next_ids = {k: v for k, v in valid_counts.items()}

def get_max_remaining():
    """Returns the maximum number of rows still needed across all active targets."""
    return max((TARGET_COUNTS[t] - valid_counts[t] for t in STOCK_PROFILES), default=0)

if get_max_remaining() <= 0:
    print(f"\nAlready reached all target rows for all stocks. Nothing to generate!")
else:
    # ─── MAIN GENERATION LOOP ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" GENERATING MORE BEARISH POST TEMPLATES")
    for t in TARGET_COUNTS:
        print(f"  {t}: {valid_counts[t]}/{TARGET_COUNTS[t]} generated")
    print(f"{'='*60}\n")

    failed_batches = 0
    csv_lock = threading.Lock()
    
    def generate_batch(batch_size_req, existing_posts_pool):
        sample_examples = random.sample(existing_posts_pool, min(EXAMPLE_COUNT, len(existing_posts_pool)))
        assigned_scenario = random.choice(SCENARIOS)
        print(f"Requesting '{assigned_scenario[:30]}...' batch size {batch_size_req}")
        system_prompt = build_system_prompt(sample_examples, assigned_scenario)
        user_prompt = f"Generate exactly {batch_size_req} unique synthetic bearish social media posts using {{TICKER}}, {{COMPANY}}, {{CEO}}, and {{PRODUCT}} placeholders. Output the raw JSON array now."
        
        response = client.chat.completions.create(
            model="qwen2.5-coder-7b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.95,
            max_tokens=600,
            timeout=120
        )
        raw_output = response.choices[0].message.content.strip()
        return parse_posts(raw_output)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while get_max_remaining() > 0:
            futures = []
            for _ in range(MAX_WORKERS):
                remaining = get_max_remaining()
                if remaining <= 0: break
                request_amount = min(BATCH_SIZE, remaining)
                futures.append(executor.submit(generate_batch, request_amount, existing_posts))
            
            if not futures: break
                
            for future in as_completed(futures):
                try:
                    template_posts = future.result()
                    if not template_posts:
                        failed_batches += 1
                        continue
                    
                    with csv_lock:
                        if get_max_remaining() <= 0: break
                        
                        batch_stats_summary = []
                        # Apply templates across ALL active dictionaries and save to their respective CSVs simultaneously
                        for target_name, target_csv in OUTPUT_CSVS.items():
                            if valid_counts[target_name] >= TARGET_COUNTS[target_name]:
                                continue # This stock has already reached its goal
                                
                            profile = STOCK_PROFILES[target_name]
                            batch_rows = []
                            for post_template in template_posts:
                                if valid_counts[target_name] + len(batch_rows) >= TARGET_COUNTS[target_name]:
                                    break # Stop appending if limit reached mid-batch
                                    
                                # Apply the translation dictionary!
                                populated_post = post_template
                                for key, replacement in profile.items():
                                    if callable(replacement):
                                        while key in populated_post:
                                            repl_str = replacement()
                                            populated_post = populated_post.replace(key, repl_str, 1)
                                    else:
                                        populated_post = populated_post.replace(key, replacement)
                                
                                batch_rows.append({
                                    "Unnamed: 0": next_ids[target_name],
                                    "id": f"synth_{next_ids[target_name]}_{target_name.lower()}",
                                    "body": populated_post,
                                    "created_at": pd.Timestamp.now().isoformat(),
                                    "user": "synthetic",
                                    "source": "lm_studio_generated",
                                    "symbols": profile["{TICKER}"],
                                    "mentioned_users": "",
                                    "entities": BEARISH_ENTITY,
                                    "links": "",
                                    "likes": 0,
                                    "conversation": "",
                                    "reshares": "",
                                    "reshare_message": "",
                                    "owned_symbols": ""
                                })
                                next_ids[target_name] += 1
                                
                            if batch_rows:
                                batch_df = pd.DataFrame(batch_rows)
                                batch_df.to_csv(target_csv, mode="a", header=False, index=False)
                                valid_counts[target_name] += len(batch_rows)
                                batch_stats_summary.append(f"{target_name}: {valid_counts[target_name]}/{TARGET_COUNTS[target_name]}")
                        
                        # Increment stats once per successful batch of templates
                        print(f"  -> Batch templates complete: [ " + " | ".join(batch_stats_summary) + " ]")
                        failed_batches = 0

                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
                    failed_batches += 1
                except Exception as e:
                    print(f"Exception during generation: {e}")
                    failed_batches += 1
                    time.sleep(5)

            gc.collect()
            if failed_batches >= MAX_CONSECUTIVE_FAILS: break

    print(f"\nDone! Overall Multi-Stock generation complete for {len(STOCK_PROFILES)} datasets.")

gc.collect()
