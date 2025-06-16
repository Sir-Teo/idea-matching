#!/usr/bin/env python3
"""
Batch process all corrected files in data/cleaned_corrections directory.
These files have been pre-processed and are in a different format than the original Excel files.

Usage:
    python batch_process_corrected.py
"""

import pandas as pd
import os
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers.cross_encoder import CrossEncoder

# Import functions from match.py
from match import (
    normalize_column_name, 
    handle_plural_singular, 
    find_best_column_match, 
    get_column_safe, 
    build_windows, 
    compute_overlap
)

def find_matching_files(base_dir="data/cleaned_corrections"):
    """
    Find all matching conversation and memory files in the directory.
    Returns a list of dictionaries with file information.
    """
    files = glob.glob(os.path.join(base_dir, "*.xlsx"))
    
    # Group files by study
    study_groups = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Extract study info from filename
        # Pattern: cleaned_Study1_KM_SY_conversation_corrected.xlsx
        # or: cleaned_Study1_KM_SY_memory-1_corrected.xlsx
        parts = filename.replace("cleaned_", "").replace("_corrected.xlsx", "").split("_")
        
        if len(parts) >= 2:
            # Find the study identifier (e.g., "Study1", "study11")
            study_id = None
            for i, part in enumerate(parts):
                if part.lower().startswith("study"):
                    study_id = part
                    # Include any additional identifiers after study
                    remaining_parts = parts[i+1:]
                    if remaining_parts:
                        # Keep non-file-type parts as part of study ID
                        additional_id = []
                        for rpart in remaining_parts:
                            if rpart not in ["conversation", "memory-1", "memory-2"]:
                                additional_id.append(rpart)
                        if additional_id:
                            study_id = study_id + "_" + "_".join(additional_id)
                    break
            
            if study_id:
                if study_id not in study_groups:
                    study_groups[study_id] = {"conversation": None, "memory-1": None, "memory-2": None}
                
                if "conversation" in filename:
                    study_groups[study_id]["conversation"] = file_path
                elif "memory-1" in filename:
                    study_groups[study_id]["memory-1"] = file_path
                elif "memory-2" in filename:
                    study_groups[study_id]["memory-2"] = file_path
    
    # Convert to list of complete sets
    complete_sets = []
    for study_id, files in study_groups.items():
        if files["conversation"] and (files["memory-1"] or files["memory-2"]):
            complete_sets.append({
                "study_id": study_id,
                "conversation": files["conversation"],
                "memory-1": files["memory-1"],
                "memory-2": files["memory-2"]
            })
    
    return complete_sets

def run_match_analysis_corrected(conv_file, mem_file, model_name, conv_window=0, mem_window=0, top_k=3):
    """
    Run match analysis on corrected files format.
    Returns stats dictionary and results DataFrame.
    """
    try:
        # Load the files
        conv_df = pd.read_excel(conv_file)
        mem_df = pd.read_excel(mem_file)
        
        # For corrected files, the structure is different
        # Columns: ['Subject Pair', 'Original Turn', 'Idea Unit #', 'Transcript']
        
        # Build windows using the Transcript column
        conv_units = conv_df["Transcript"].astype(str).str.strip().tolist()
        conv_wins, conv_centers = build_windows(conv_units, 
                                               left=conv_window, 
                                               right=conv_window)

        mem_units = mem_df["Transcript"].astype(str).str.strip().tolist()
        mem_wins, mem_centers = build_windows(mem_units, 
                                             left=mem_window, 
                                             right=mem_window)

        # Load model
        cross_encoder = CrossEncoder(model_name)

        # Process each memory window
        rows = []
        for idx, mem_win in enumerate(mem_wins):
            mem_center = mem_centers[idx]
            mem_row = mem_df.iloc[mem_center]

            # Rank conversation windows for this memory window
            ranks = cross_encoder.rank(mem_win, conv_wins, top_k=top_k)
            auto_cuids = []
            auto_windows = []
            scores = []
            
            for r in ranks:
                idx_corp = int(r["corpus_id"])
                conv_idx = conv_centers[idx_corp]
                auto_cuids.append(conv_df.iloc[conv_idx]["Idea Unit #"])
                auto_windows.append(conv_wins[idx_corp])
                scores.append(r["score"])
            
            # For corrected files, we don't have manual coding to compare against
            # So we'll set manual_cuid to None and is_match to None
            manual_cuid = None
            is_match = None

            # compute overlap metrics between memory and auto conversation window
            overlap_count, overlap_jaccard = compute_overlap(mem_win, auto_windows[0])
            
            rows.append({
                "memory unit #": mem_row["Idea Unit #"],
                "memory sentence": mem_win,
                "manual conversation unit #": manual_cuid,
                "auto conversation unit #": auto_cuids[0],
                "score": scores[0],
                "is_match": is_match,
                "auto conversation window": auto_windows[0],
                "overlap_count": overlap_count,
                "overlap_jaccard": overlap_jaccard
            })

        # Create results DataFrame
        results_df = pd.DataFrame(rows)
        
        # Calculate summary statistics
        total = len(results_df)
        
        # Calculate overlapping words statistics
        global_counter = Counter()
        for _, row in results_df.iterrows():
            tokens1 = set(re.findall(r"\w+", row["memory sentence"].lower())) - ENGLISH_STOP_WORDS
            tokens2 = set(re.findall(r"\w+", row["auto conversation window"].lower())) - ENGLISH_STOP_WORDS
            overlap = tokens1 & tokens2
            global_counter.update(overlap)
        
        # Calculate global overlap percentages
        all_tokens_mem = set()
        all_tokens_auto = set()
        for _, row in results_df.iterrows():
            tokens_mem_raw = set(re.findall(r"\w+", row["memory sentence"].lower()))
            all_tokens_mem.update(tokens_mem_raw)
            tokens_auto_raw = set(re.findall(r"\w+", row["auto conversation window"].lower()))
            all_tokens_auto.update(tokens_auto_raw)
        
        global_overlap_all = all_tokens_mem & all_tokens_auto
        union_all = all_tokens_mem | all_tokens_auto
        perc_overlap_all = len(global_overlap_all) / len(union_all) * 100 if union_all else 0.0
        
        all_tokens_mem_imp = all_tokens_mem - ENGLISH_STOP_WORDS
        all_tokens_auto_imp = all_tokens_auto - ENGLISH_STOP_WORDS
        global_overlap_imp = all_tokens_mem_imp & all_tokens_auto_imp
        union_imp = all_tokens_mem_imp | all_tokens_auto_imp
        perc_overlap_imp = len(global_overlap_imp) / len(union_imp) * 100 if union_imp else 0.0
        
        # Compile stats
        stats = {
            'model': model_name,
            'total_windows': total,
            'score_mean': results_df['score'].mean(),
            'score_median': results_df['score'].median(),
            'score_min': results_df['score'].min(),
            'score_max': results_df['score'].max(),
            'unique_overlapping_words': len(global_counter),
            'overlap_percent_all': perc_overlap_all,
            'overlap_percent_important': perc_overlap_imp
        }
        
        return stats, results_df
        
    except Exception as e:
        print(f"Error processing {conv_file} and {mem_file}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Batch process corrected files for idea matching")
    parser.add_argument("--model", default="cross-encoder/stsb-distilroberta-base",
                       help="HuggingFace Cross-Encoder model")
    parser.add_argument("--conv_window", type=int, default=0,
                       help="Context window size for conversation units")
    parser.add_argument("--mem_window", type=int, default=0,
                       help="Context window size for memory units")
    parser.add_argument("--top_k", type=int, default=3,
                       help="Number of top matches to consider")
    parser.add_argument("--data_dir", default="data/cleaned_corrections",
                       help="Directory containing corrected files")
    parser.add_argument("--output_dir", default="results_corrected",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all matching file sets
    file_sets = find_matching_files(args.data_dir)
    
    if not file_sets:
        print(f"No matching file sets found in {args.data_dir}")
        return
    
    print(f"Found {len(file_sets)} study sets to process")
    
    # Store all results
    all_results = []
    all_detailed_results = []
    
    # Process each file set
    total_runs = sum(1 for fs in file_sets for mem_type in ["memory-1", "memory-2"] if fs[mem_type])
    
    with tqdm(total=total_runs, desc="Processing files") as pbar:
        for file_set in file_sets:
            study_id = file_set["study_id"]
            conv_file = file_set["conversation"]
            
            for mem_type in ["memory-1", "memory-2"]:
                mem_file = file_set[mem_type]
                if not mem_file:
                    continue
                
                pbar.set_description(f"Processing {study_id} {mem_type}")
                
                stats, detailed_df = run_match_analysis_corrected(
                    conv_file, mem_file, args.model,
                    conv_window=args.conv_window,
                    mem_window=args.mem_window,
                    top_k=args.top_k
                )
                
                if stats is not None:
                    stats['study_id'] = study_id
                    stats['memory_type'] = mem_type
                    stats['conv_file'] = os.path.basename(conv_file)
                    stats['mem_file'] = os.path.basename(mem_file)
                    all_results.append(stats)
                    
                    if detailed_df is not None:
                        detailed_df['study_id'] = study_id
                        detailed_df['memory_type'] = mem_type
                        all_detailed_results.append(detailed_df)
                        
                        # Save individual CSV
                        model_tag = args.model.replace("/", "_")
                        csv_filename = f"result_{study_id}_{mem_type}_{model_tag}_conv{args.conv_window}_mem{args.mem_window}.csv"
                        csv_path = os.path.join(args.output_dir, csv_filename)
                        detailed_df.drop(['study_id', 'memory_type'], axis=1).to_csv(csv_path, index=False)
                        
                        print(f"Saved: {csv_filename}")
                else:
                    print(f"Failed to process {study_id} {mem_type}")
                
                pbar.update(1)
    
    if not all_results:
        print("No results to save")
        return
    
    # Save summary results
    results_df = pd.DataFrame(all_results)
    summary_path = os.path.join(args.output_dir, "batch_processing_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    
    # Combine and save all detailed results
    if all_detailed_results:
        combined_detailed_df = pd.concat(all_detailed_results, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "all_detailed_results.csv")
        combined_detailed_df.to_csv(combined_path, index=False)
        print(f"Saved combined results: {combined_path}")
    
    print(f"\nProcessed {len(all_results)} file pairs")
    print(f"Average score: {results_df['score_mean'].mean():.4f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 