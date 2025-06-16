"""conversation_memory_align_modified.py

Usage examples:
    python conversation_memory_align_modified.py \
        --workbook template_Study1_KM_SY.xlsx \
        --conv_sheet "Conversation idea units" \
        --mem_sheet "Memory 1 Idea Units" \
        --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
        --conv_window 1 --mem_window 2

The script ranks every *memory* idea-unit window against *conversation* idea-unit windows
using a HuggingFace Cross-Encoder.  It then writes a CSV that includes **both** the manual
coding (ground-truth conversation idea-unit number) and the automatically ranked top match
so you can directly compare auto vs. manual coding.

CSV fields written:
    memory unit #              – ID from memory sheet
    memory sentence            – memory window text used as query
    manual conversation unit # – ground-truth column from memory sheet
    auto conversation unit #   – top ranked conversation idea-unit ID
    score                      – Cross-Encoder similarity score
    is_match                   – 1 if auto == manual else 0
    auto conversation window   – conversation text window returned by the model

If the memory sheet lacks a column literally named "matching idea unit from conversation",
use --manual_col to point to the correct one.
"""

import argparse, os, sys
import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
from difflib import get_close_matches

# ---------- helpers ----------

def normalize_column_name(name):
    """Normalize column name: lowercase, strip spaces, handle common variations."""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    # Remove extra whitespace between words
    name = re.sub(r'\s+', ' ', name)
    return name

def handle_plural_singular(name):
    """Generate both plural and singular variations of a column name."""
    variations = [name]
    
    # Handle common plural/singular patterns
    if name.endswith('s') and len(name) > 1:
        # Try removing 's' for singular
        variations.append(name[:-1])
    else:
        # Try adding 's' for plural
        variations.append(name + 's')
    
    # Handle 'unit' vs 'units' specifically
    if 'unit ' in name:
        variations.append(name.replace('unit ', 'units '))
    if 'units ' in name:
        variations.append(name.replace('units ', 'unit '))
    
    return variations

def find_best_column_match(target_col, available_cols, threshold=0.6):
    """
    Find the best matching column name from available columns.
    
    Args:
        target_col: The target column name to match
        available_cols: List of available column names
        threshold: Minimum similarity threshold (0-1)
    
    Returns:
        Best matching column name or None if no good match found
    """
    if not target_col or not available_cols:
        return None
    
    # Normalize the target column
    target_normalized = normalize_column_name(target_col)
    
    # Generate variations of the target (plural/singular)
    target_variations = handle_plural_singular(target_normalized)
    
    # Normalize all available columns
    available_normalized = [(normalize_column_name(col), col) for col in available_cols]
    available_norm_names = [norm for norm, orig in available_normalized]
    
    # First, try exact matches with any variation
    for variation in target_variations:
        for norm_name, orig_name in available_normalized:
            if variation == norm_name:
                return orig_name
    
    # If no exact match, use fuzzy matching
    best_matches = []
    for variation in target_variations:
        matches = get_close_matches(variation, available_norm_names, n=3, cutoff=threshold)
        for match in matches:
            # Find the original column name
            for norm_name, orig_name in available_normalized:
                if norm_name == match:
                    best_matches.append((orig_name, match))
                    break
    
    if best_matches:
        # Return the first best match
        return best_matches[0][0]
    
    return None

def get_column_safe(df, target_col, sheet_name=""):
    """
    Safely get a column from dataframe with fuzzy matching.
    Raises a clear error if column cannot be found.
    """
    if target_col in df.columns:
        return df[target_col]
    
    best_match = find_best_column_match(target_col, df.columns.tolist())
    
    if best_match:
        print(f"Column '{target_col}' not found in {sheet_name}. Using '{best_match}' instead.")
        return df[best_match]
    else:
        available_cols = ", ".join(df.columns.tolist())
        raise KeyError(f"Column '{target_col}' not found in {sheet_name}. "
                      f"Available columns: {available_cols}")

def build_windows(units, left=1, right=1):
    """Return two parallel lists:
         windows – list[str] where each item is a concatenated context window
         centers – list[int] index of the centre idea-unit in original *units* list"""
    pad = [""] * left + units + [""] * right
    windows, centers = [], []
    for i in range(left, len(pad) - right):
        win = " ".join(pad[i - left : i + right + 1]).strip()
        windows.append(win)
        centers.append(i - left)  # position w.r.t. original list
    return windows, centers

def compute_overlap(text1, text2):
    """Compute overlap count and Jaccard similarity between texts after filtering out stopwords."""
    tokens1 = set(re.findall(r"\w+", text1.lower())) - ENGLISH_STOP_WORDS
    tokens2 = set(re.findall(r"\w+", text2.lower())) - ENGLISH_STOP_WORDS
    overlap = tokens1 & tokens2
    overlap_count = len(overlap)
    union = tokens1 | tokens2
    jaccard = float(overlap_count) / len(union) if union else 0.0
    return overlap_count, jaccard

def find_best_sheet_match(target_sheet, available_sheets, threshold=0.6):
    """
    Find the best matching sheet name from available sheets.
    
    Args:
        target_sheet: The target sheet name to match
        available_sheets: List of available sheet names
        threshold: Minimum similarity threshold (0-1)
    
    Returns:
        Best matching sheet name or None if no good match found
    """
    if not target_sheet or not available_sheets:
        return None
    
    # Normalize the target sheet
    target_normalized = normalize_column_name(target_sheet)
    
    # Generate variations of the target (plural/singular)
    target_variations = handle_plural_singular(target_normalized)
    
    # Normalize all available sheets
    available_normalized = [(normalize_column_name(sheet), sheet) for sheet in available_sheets]
    available_norm_names = [norm for norm, orig in available_normalized]
    
    # First, try exact matches with any variation
    for variation in target_variations:
        for norm_name, orig_name in available_normalized:
            if variation == norm_name:
                return orig_name
    
    # If no exact match, use fuzzy matching
    best_matches = []
    for variation in target_variations:
        matches = get_close_matches(variation, available_norm_names, n=3, cutoff=threshold)
        for match in matches:
            # Find the original sheet name
            for norm_name, orig_name in available_normalized:
                if norm_name == match:
                    best_matches.append((orig_name, match))
                    break
    
    if best_matches:
        # Return the first best match
        return best_matches[0][0]
    
    return None

# ---------- CLI ----------

parser = argparse.ArgumentParser()
parser.add_argument("--workbook", default="/Users/teo/Desktop/Yoon/idea_matching/data/template_Study3_ED.xlsx",
                    help="Path to XLSX workbook that holds conversation & memory sheets")
parser.add_argument("--conv_sheet", default="Conversation idea units ",
                    help="Sheet name containing conversation idea-units")
parser.add_argument("--mem_sheet",  default="Memory 1 Idea Units ",
                    help="Sheet name containing memory idea-units")
parser.add_argument("--model", default="cross-encoder/stsb-distilroberta-base",
                    help="Any Cross-Encoder model on HuggingFace hub")
parser.add_argument("--conv_window", type=int, default=0,
                    help="# idea-units of context either side of centre for conversation windows")
parser.add_argument("--mem_window",  type=int, default=0,
                    help="# idea-units of context either side of centre for memory windows")
parser.add_argument("--manual_col", default="matching idea unit",
                    help="Column in memory sheet that stores the manually coded conversation unit ID")
parser.add_argument("--top_k", type=int, default=3,
                    help="Number of top matched conversation units to consider for ranking and is_match computation")

args = parser.parse_args()

# ---------- load workbook ----------

try:
    # First, get all available sheet names
    excel_file = pd.ExcelFile(args.workbook)
    available_sheets = excel_file.sheet_names
    
    # Find best matching sheet names
    conv_sheet_match = find_best_sheet_match(args.conv_sheet, available_sheets)
    mem_sheet_match = find_best_sheet_match(args.mem_sheet, available_sheets)
    
    if not conv_sheet_match:
        available_sheet_list = ", ".join(str(sheet) for sheet in available_sheets)
        sys.exit(f"Conversation sheet '{args.conv_sheet}' not found. Available sheets: {available_sheet_list}")
    
    if not mem_sheet_match:
        available_sheet_list = ", ".join(str(sheet) for sheet in available_sheets)
        sys.exit(f"Memory sheet '{args.mem_sheet}' not found. Available sheets: {available_sheet_list}")
    
    # Inform user if we're using different sheet names
    if conv_sheet_match != args.conv_sheet:
        print(f"Conversation sheet '{args.conv_sheet}' not found. Using '{conv_sheet_match}' instead.")
    
    if mem_sheet_match != args.mem_sheet:
        print(f"Memory sheet '{args.mem_sheet}' not found. Using '{mem_sheet_match}' instead.")
    
    # Load the sheets using the matched names
    conv_df = pd.read_excel(args.workbook, sheet_name=conv_sheet_match)
    mem_df  = pd.read_excel(args.workbook, sheet_name=mem_sheet_match)
    
except FileNotFoundError:
    sys.exit(f"Workbook file not found: {args.workbook}")
except Exception as e:
    sys.exit(f"Error reading workbook: {e}")

# ---------- build windows with robust column access ----------

conv_units = get_column_safe(conv_df, "Transcript", f"conversation sheet '{args.conv_sheet}'").astype(str).str.strip().tolist()
conv_wins, conv_centers = build_windows(conv_units,
                                        left=args.conv_window,
                                        right=args.conv_window)

mem_units = get_column_safe(mem_df, "Transcript", f"memory sheet '{args.mem_sheet}'").astype(str).str.strip().tolist()
mem_wins, mem_centers = build_windows(mem_units,
                                      left=args.mem_window,
                                      right=args.mem_window)

# ---------- rank ----------

model = CrossEncoder(args.model)

rows = []
for idx, mem_win in tqdm(list(enumerate(mem_wins)),
                         total=len(mem_wins), desc="Ranking memory units"):
    mem_center = mem_centers[idx]
    mem_row    = mem_df.iloc[mem_center]

    # Rank conversation windows for this memory window
    ranks = model.rank(mem_win, conv_wins, top_k=args.top_k)
    auto_cuids = []
    auto_windows = []
    scores = []
    for r in ranks:
        idx_corp = int(r["corpus_id"])
        conv_idx = conv_centers[idx_corp]
        conv_id_col = get_column_safe(conv_df, "idea units #", f"conversation sheet '{args.conv_sheet}'")
        auto_cuids.append(conv_id_col.iloc[conv_idx])
        auto_windows.append(conv_wins[idx_corp])
        scores.append(r["score"])
    
    manual_cuid = get_column_safe(mem_row.to_frame().T, args.manual_col, f"memory sheet '{args.mem_sheet}'").iloc[0]
    
    # handle cases where manual_cuid may be a range like "5-7"
    manual_str = str(manual_cuid).strip()
    if '-' in manual_str:
        parts = manual_str.split('-')
        try:
            start = int(parts[0].strip()); end = int(parts[1].strip())
            manual_ids = [str(i) for i in range(start, end + 1)]
        except ValueError:
            manual_ids = [manual_str]
    else:
        manual_ids = [manual_str]
    # check if any auto_cuid falls within the manual IDs
    is_match = int(any(str(ac) in manual_ids for ac in auto_cuids))

    # compute overlap metrics between memory and auto conversation window
    overlap_count, overlap_jaccard = compute_overlap(mem_win, auto_windows[0])

    mem_id_col = get_column_safe(mem_row.to_frame().T, "idea unit #", f"memory sheet '{args.mem_sheet}'")
    
    rows.append({
        "memory unit #":            mem_id_col.iloc[0],
        "memory sentence":          mem_win,
        "manual conversation unit #": manual_cuid,
        "auto conversation unit #":  auto_cuids[0],
        "score":                    scores[0],
        "is_match":                 is_match,
        "auto conversation window": auto_windows[0],
        "overlap_count":            overlap_count,
        "overlap_jaccard":          overlap_jaccard
    })

# ---------- save ----------

base_name = os.path.splitext(os.path.basename(args.workbook))[0]
model_tag = args.model.replace("/", "_")
outfile = f"result_{base_name}_{model_tag}_conv{args.conv_window}_mem{args.mem_window}.csv"

# save results
df = pd.DataFrame(rows)
df.to_csv(outfile, index=False)
print(f"Saved {outfile}")

# summary statistics
total = len(df)
matches = df['is_match'].sum()
accuracy = matches / total * 100 if total else 0.0
print("\nSummary statistics:")
print(f"Model: {args.model}")
print(f"Total memory windows: {total}")
print(f"Top-{args.top_k} accuracy: {matches}/{total} ({accuracy:.2f}%)")
print(f"Score: mean {df['score'].mean():.4f}, median {df['score'].median():.4f}, min {df['score'].min():.4f}, max {df['score'].max():.4f}")

# overall overlapping words analysis
global_counter = Counter()
for _, row in df.iterrows():
    tokens1 = set(re.findall(r"\w+", row["memory sentence"].lower())) - ENGLISH_STOP_WORDS
    tokens2 = set(re.findall(r"\w+", row["auto conversation window"].lower())) - ENGLISH_STOP_WORDS
    overlap = tokens1 & tokens2
    global_counter.update(overlap)
print("\nOverall overlapping words analysis:")
print(f"Unique overlapping words: {len(global_counter)}")
print("Top 20 overlapping words:")
for word, cnt in global_counter.most_common(20):
    print(f"  {word}: {cnt}")

# overall overlap percentages
# compute global raw token sets
all_tokens_mem = set()
all_tokens_auto = set()
for _, row in df.iterrows():
    tokens_mem_raw = set(re.findall(r"\w+", row["memory sentence"].lower()))
    all_tokens_mem.update(tokens_mem_raw)
    tokens_auto_raw = set(re.findall(r"\w+", row["auto conversation window"].lower()))
    all_tokens_auto.update(tokens_auto_raw)
global_overlap_all = all_tokens_mem & all_tokens_auto
union_all = all_tokens_mem | all_tokens_auto
perc_overlap_all = len(global_overlap_all) / len(union_all) * 100 if union_all else 0.0
# compute important words overlap
all_tokens_mem_imp = all_tokens_mem - ENGLISH_STOP_WORDS
all_tokens_auto_imp = all_tokens_auto - ENGLISH_STOP_WORDS
global_overlap_imp = all_tokens_mem_imp & all_tokens_auto_imp
union_imp = all_tokens_mem_imp | all_tokens_auto_imp
perc_overlap_imp = len(global_overlap_imp) / len(union_imp) * 100 if union_imp else 0.0
print(f"\nPercentage of overlapping words (all tokens): {perc_overlap_all:.2f}%")
print(f"Percentage of overlapping words (important tokens): {perc_overlap_imp:.2f}%")
