#!/usr/bin/env python3
"""
Script to compare two cross-encoder models on all Excel files in the data directory.
Runs match.py functions directly for both models and creates comparison plots.

Models compared:
- cross-encoder/stsb-distilroberta-base (default)
- cross-encoder/ms-marco-MiniLM-L-12-v2

Usage:
    python compare_models.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers.cross_encoder import CrossEncoder

# Import functions from match.py
from match import (
    normalize_column_name, 
    handle_plural_singular, 
    find_best_column_match, 
    get_column_safe, 
    build_windows, 
    compute_overlap,
    find_best_sheet_match
)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def run_match_analysis(xlsx_file, model, conv_window=0, mem_window=0, top_k=3, manual_col="matching idea unit"):
    """
    Run the match analysis directly without subprocess.
    Returns stats dictionary and results DataFrame.
    """
    try:
        # Load workbook
        excel_file = pd.ExcelFile(xlsx_file)
        available_sheets = excel_file.sheet_names
        
        # Find best matching sheet names (using defaults from match.py)
        conv_sheet_match = find_best_sheet_match("Conversation idea units", available_sheets)
        mem_sheet_match = find_best_sheet_match("Memory 1 Idea Units", available_sheets)
        
        if not conv_sheet_match or not mem_sheet_match:
            print(f"Required sheets not found in {xlsx_file}")
            return None, None
        
        # Load the sheets using the matched names
        conv_df = pd.read_excel(xlsx_file, sheet_name=conv_sheet_match)
        mem_df = pd.read_excel(xlsx_file, sheet_name=mem_sheet_match)
        
        # Build windows
        conv_units = get_column_safe(conv_df, "Transcript", f"conversation sheet").astype(str).str.strip().tolist()
        conv_wins, conv_centers = build_windows(conv_units, left=conv_window, right=conv_window)

        mem_units = get_column_safe(mem_df, "Transcript", f"memory sheet").astype(str).str.strip().tolist()
        mem_wins, mem_centers = build_windows(mem_units, left=mem_window, right=mem_window)

        # Load model
        cross_encoder = CrossEncoder(model)

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
                conv_id_col = get_column_safe(conv_df, "idea units #", f"conversation sheet")
                auto_cuids.append(conv_id_col.iloc[conv_idx])
                auto_windows.append(conv_wins[idx_corp])
                scores.append(r["score"])
            
            manual_cuid = get_column_safe(mem_row.to_frame().T, manual_col, f"memory sheet").iloc[0]
            
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

            mem_id_col = get_column_safe(mem_row.to_frame().T, "idea unit #", f"memory sheet")
            
            rows.append({
                "memory unit #": mem_id_col.iloc[0],
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
        matches = results_df['is_match'].sum()
        accuracy = matches / total * 100 if total else 0.0
        
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
            'model': model,
            'total_windows': total,
            'matches': matches,
            'total': total,
            'accuracy_percent': accuracy,
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
        print(f"Error processing {xlsx_file} with {model}: {e}")
        return None, None

def run_comparison():
    """Main function to run comparison between models."""
    
    # Define models to compare
    models = [
        'cross-encoder/stsb-distilroberta-base',
        'cross-encoder/ms-marco-MiniLM-L-12-v2'
    ]
    
    # Find all Excel files in data directory
    data_dir = 'data'
    xlsx_files = glob.glob(os.path.join(data_dir, '*.xlsx'))
    
    if not xlsx_files:
        print(f"No Excel files found in {data_dir} directory")
        return
    
    print(f"Found {len(xlsx_files)} Excel files to process")
    
    # Store results
    all_results = []
    all_detailed_results = []
    
    # Run analysis for each file and model combination
    total_runs = len(xlsx_files) * len(models)
    with tqdm(total=total_runs, desc="Running comparisons") as pbar:
        for xlsx_file in xlsx_files:
            file_name = os.path.basename(xlsx_file)
            for model in models:
                pbar.set_description(f"Processing {file_name} with {model.split('/')[-1]}")
                
                stats, detailed_df = run_match_analysis(xlsx_file, model)
                
                if stats is not None:
                    stats['file'] = file_name
                    stats['xlsx_path'] = xlsx_file
                    all_results.append(stats)
                    
                    if detailed_df is not None:
                        detailed_df['model'] = model
                        detailed_df['source_file'] = file_name
                        all_detailed_results.append(detailed_df)
                        
                        # Save individual CSV for this run
                        base_name = os.path.splitext(file_name)[0]
                        model_tag = model.replace("/", "_")
                        csv_filename = f"result_{base_name}_{model_tag}_conv0_mem0.csv"
                        detailed_df.drop(['model', 'source_file'], axis=1).to_csv(csv_filename, index=False)
                else:
                    print(f"Failed to process {file_name} with {model}")
                
                pbar.update(1)
    
    if not all_results:
        print("No results to compare")
        return
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Combine all detailed results
    if all_detailed_results:
        combined_detailed_df = pd.concat(all_detailed_results, ignore_index=True)
    else:
        combined_detailed_df = None
    
    # Create comparison plots
    create_comparison_plots(results_df, combined_detailed_df)
    
    # Save results summary
    results_df.to_csv('model_comparison_summary.csv', index=False)
    print("\nSaved model_comparison_summary.csv")
    
    return results_df

def create_comparison_plots(results_df, combined_detailed_df=None):
    """Create comprehensive comparison plots between models as separate images."""
    
    models = results_df['model'].unique()
    model_labels = [model.split('/')[-1] for model in models]
    
    print("\nCreating comparison plots...")
    
    # 1. Accuracy comparison by file
    plt.figure(figsize=(14, 8))
    pivot_acc = results_df.pivot(index='file', columns='model', values='accuracy_percent')
    pivot_acc.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Accuracy by File', fontsize=16, fontweight='bold')
    plt.xlabel('Files', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(model_labels, loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('01_accuracy_by_file.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Average accuracy comparison
    plt.figure(figsize=(8, 6))
    avg_acc = results_df.groupby('model')['accuracy_percent'].mean()
    bars = plt.bar(model_labels, avg_acc.values, color=['skyblue', 'lightcoral'])
    plt.title('Average Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Average Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_acc.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('02_average_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Score mean comparison by file
    plt.figure(figsize=(14, 8))
    pivot_score = results_df.pivot(index='file', columns='model', values='score_mean')
    pivot_score.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Mean Score by File', fontsize=16, fontweight='bold')
    plt.xlabel('Files', fontsize=12)
    plt.ylabel('Mean Score', fontsize=12)
    plt.legend(model_labels, loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('03_mean_score_by_file.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Score distribution comparison
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        plt.hist(model_data['score_mean'], alpha=0.7, label=model_labels[i], bins=15)
    plt.title('Score Mean Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Mean Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('04_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Overlap percentage (all tokens) comparison
    plt.figure(figsize=(14, 8))
    pivot_overlap = results_df.pivot(index='file', columns='model', values='overlap_percent_all')
    pivot_overlap.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Overlap Percentage (All Tokens) by File', fontsize=16, fontweight='bold')
    plt.xlabel('Files', fontsize=12)
    plt.ylabel('Overlap (%)', fontsize=12)
    plt.legend(model_labels, loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('05_overlap_all_tokens_by_file.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Overlap percentage (important tokens) comparison
    plt.figure(figsize=(14, 8))
    pivot_overlap_imp = results_df.pivot(index='file', columns='model', values='overlap_percent_important')
    pivot_overlap_imp.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Overlap Percentage (Important Tokens) by File', fontsize=16, fontweight='bold')
    plt.xlabel('Files', fontsize=12)
    plt.ylabel('Overlap (%)', fontsize=12)
    plt.legend(model_labels, loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('06_overlap_important_tokens_by_file.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Score range (max - min) comparison
    plt.figure(figsize=(14, 8))
    results_df['score_range'] = results_df['score_max'] - results_df['score_min']
    pivot_range = results_df.pivot(index='file', columns='model', values='score_range')
    pivot_range.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Score Range (Max - Min) by File', fontsize=16, fontweight='bold')
    plt.xlabel('Files', fontsize=12)
    plt.ylabel('Score Range', fontsize=12)
    plt.legend(model_labels, loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('07_score_range_by_file.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Scatter plot: Accuracy vs Mean Score
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        plt.scatter(model_data['score_mean'], model_data['accuracy_percent'], 
                   color=colors[i], label=model_labels[i], alpha=0.7, s=50)
    plt.title('Accuracy vs Mean Score', fontsize=16, fontweight='bold')
    plt.xlabel('Mean Score', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('08_accuracy_vs_mean_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Unique overlapping words comparison
    plt.figure(figsize=(14, 8))
    pivot_unique = results_df.pivot(index='file', columns='model', values='unique_overlapping_words')
    pivot_unique.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Unique Overlapping Words by File', fontsize=16, fontweight='bold')
    plt.xlabel('Files', fontsize=12)
    plt.ylabel('Unique Words', fontsize=12)
    plt.legend(model_labels, loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('09_unique_overlapping_words_by_file.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Average metrics comparison (bar chart)
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy_percent', 'score_mean', 'overlap_percent_all', 'overlap_percent_important']
    metric_labels = ['Accuracy (%)', 'Mean Score', 'Overlap All (%)', 'Overlap Imp (%)']
    
    model1_data = results_df[results_df['model'] == models[0]][metrics].mean()
    model2_data = results_df[results_df['model'] == models[1]][metrics].mean()
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, model1_data.values, width, label=model_labels[0], alpha=0.8)
    bars2 = plt.bar(x + width/2, model2_data.values, width, label=model_labels[1], alpha=0.8)
    
    plt.title('Average Metrics Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.xticks(x, metric_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('10_average_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 11. Box plot of accuracies
    plt.figure(figsize=(8, 6))
    accuracy_data = [results_df[results_df['model'] == model]['accuracy_percent'].values 
                     for model in models]
    box_plot = plt.boxplot(accuracy_data)
    plt.title('Accuracy Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(range(1, len(model_labels) + 1), model_labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('11_accuracy_distribution_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 12. Performance difference plot
    if len(models) == 2:
        plt.figure(figsize=(14, 8))
        # Calculate performance difference (model2 - model1)
        pivot_acc = results_df.pivot(index='file', columns='model', values='accuracy_percent')
        diff = pivot_acc.iloc[:, 1] - pivot_acc.iloc[:, 0]  # model2 - model1
        
        colors = ['green' if x > 0 else 'red' for x in diff]
        bars = plt.bar(range(len(diff)), diff.values, color=colors, alpha=0.7)
        plt.title(f'Accuracy Difference\n({model_labels[1]} - {model_labels[0]})', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Files', fontsize=12)
        plt.ylabel('Accuracy Difference (%)', fontsize=12)
        plt.xticks(range(len(diff)), diff.index, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, diff.values):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.5 if val > 0 else -0.5), 
                    f'{val:.1f}', ha='center', 
                    va='bottom' if val > 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('12_accuracy_difference.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Main comparison plots saved as separate images (01-12).")
    
    # Create detailed CSV analysis plots if detailed data exists
    if combined_detailed_df is not None:
        create_detailed_csv_analysis(combined_detailed_df)

def create_detailed_csv_analysis(combined_detailed_df):
    """Create detailed analysis from the combined detailed results as separate images."""
    
    print("\nCreating detailed CSV analysis plots...")
    
    models = combined_detailed_df['model'].unique()
    model_labels = [model.split('/')[-1] for model in models]
    colors = ['blue', 'red']
    
    # 1. Score distribution
    plt.figure(figsize=(10, 6))
    score_data = [combined_detailed_df[combined_detailed_df['model'] == model]['score'].values 
                  for model in models]
    plt.hist(score_data, bins=30, alpha=0.7, label=model_labels, color=colors)
    plt.title('Score Distribution (All Individual Matches)', fontsize=16, fontweight='bold')
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('detailed_01_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Is_match comparison
    plt.figure(figsize=(8, 6))
    match_rates = [combined_detailed_df[combined_detailed_df['model'] == model]['is_match'].mean() * 100 
                   for model in models]
    bars = plt.bar(model_labels, match_rates, color=colors, alpha=0.7)
    plt.title('Overall Match Rate', fontsize=16, fontweight='bold')
    plt.ylabel('Match Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, match_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detailed_02_overall_match_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Overlap count distribution
    plt.figure(figsize=(8, 6))
    overlap_data = [combined_detailed_df[combined_detailed_df['model'] == model]['overlap_count'].values 
                    for model in models]
    box_plot = plt.boxplot(overlap_data)
    plt.title('Overlap Count Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('Overlap Count', fontsize=12)
    plt.xticks(range(1, len(model_labels) + 1), model_labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('detailed_03_overlap_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Overlap Jaccard distribution
    plt.figure(figsize=(8, 6))
    jaccard_data = [combined_detailed_df[combined_detailed_df['model'] == model]['overlap_jaccard'].values 
                    for model in models]
    box_plot = plt.boxplot(jaccard_data)
    plt.title('Jaccard Similarity Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('Jaccard Similarity', fontsize=12)
    plt.xticks(range(1, len(model_labels) + 1), model_labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('detailed_04_jaccard_similarity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Score vs Match scatter
    plt.figure(figsize=(12, 8))
    for i, model in enumerate(models):
        data = combined_detailed_df[combined_detailed_df['model'] == model]
        matches = data[data['is_match'] == 1]
        non_matches = data[data['is_match'] == 0]
        
        plt.scatter(matches['score'], [i + 0.1] * len(matches), 
                   color=colors[i], alpha=0.6, label=f'{model_labels[i]} (Match)', s=20)
        plt.scatter(non_matches['score'], [i - 0.1] * len(non_matches), 
                   color=colors[i], alpha=0.3, marker='x', 
                   label=f'{model_labels[i]} (No Match)', s=20)
    
    plt.title('Score Distribution by Match Status', fontsize=16, fontweight='bold')
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.yticks([0, 1], model_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('detailed_05_score_by_match_status.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Performance by file
    plt.figure(figsize=(16, 8))
    match_rates_by_file = {}
    for model in models:
        data = combined_detailed_df[combined_detailed_df['model'] == model]
        match_rates_by_file[model] = data.groupby('source_file')['is_match'].mean() * 100
    
    files = list(set().union(*[list(match_rates_by_file[model].index) for model in models]))
    x = np.arange(len(files))
    width = 0.35
    
    for i, model in enumerate(models):
        rates = [match_rates_by_file[model].get(f, 0) for f in files]
        plt.bar(x + i * width, rates, width, label=model_labels[i], 
               color=colors[i], alpha=0.7)
    
    plt.title('Match Rate by File', fontsize=16, fontweight='bold')
    plt.xlabel('Files', fontsize=12)
    plt.ylabel('Match Rate (%)', fontsize=12)
    plt.xticks(x + width / 2, [f[:15] + '...' if len(f) > 15 else f for f in files], 
               rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('detailed_06_match_rate_by_file.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Detailed analysis plots saved as separate images (detailed_01-06).")

def print_summary_statistics(results_df):
    """Print summary statistics comparing the models."""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    models = results_df['model'].unique()
    
    for model in models:
        model_data = results_df[results_df['model'] == model]
        print(f"\nModel: {model}")
        print(f"Files processed: {len(model_data)}")
        print(f"Average accuracy: {model_data['accuracy_percent'].mean():.2f}% ± {model_data['accuracy_percent'].std():.2f}%")
        print(f"Average score: {model_data['score_mean'].mean():.4f} ± {model_data['score_mean'].std():.4f}")
        print(f"Average overlap (all): {model_data['overlap_percent_all'].mean():.2f}%")
        print(f"Average overlap (important): {model_data['overlap_percent_important'].mean():.2f}%")
    
    if len(models) == 2:
        model1_data = results_df[results_df['model'] == models[0]]
        model2_data = results_df[results_df['model'] == models[1]]
        
        acc_diff = model2_data['accuracy_percent'].mean() - model1_data['accuracy_percent'].mean()
        score_diff = model2_data['score_mean'].mean() - model1_data['score_mean'].mean()
        
        print(f"\n{models[1].split('/')[-1]} vs {models[0].split('/')[-1]}:")
        print(f"Accuracy difference: {acc_diff:+.2f}%")
        print(f"Score difference: {score_diff:+.4f}")
        
        # Statistical significance test (paired t-test)
        from scipy import stats
        acc_t_stat, acc_p_value = stats.ttest_rel(model2_data['accuracy_percent'], 
                                                  model1_data['accuracy_percent'])
        print(f"Accuracy difference p-value: {acc_p_value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare cross-encoder models on idea matching task")
    parser.add_argument("--skip-run", action="store_true", 
                       help="Skip running analysis and use existing CSV files")
    args = parser.parse_args()
    
    if not args.skip_run:
        results_df = run_comparison()
    else:
        # Try to load existing results
        if os.path.exists('model_comparison_summary.csv'):
            results_df = pd.read_csv('model_comparison_summary.csv')
            print("Loaded existing results from model_comparison_summary.csv")
            
            # Try to load existing detailed CSV files and create plots
            csv_files = glob.glob('result_*.csv')
            if csv_files:
                detailed_dfs = []
                for csv_file in csv_files:
                    if 'stsb-distilroberta-base' in csv_file:
                        model = 'cross-encoder/stsb-distilroberta-base'
                    elif 'ms-marco-MiniLM-L-12-v2' in csv_file:
                        model = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
                    else:
                        continue
                    
                    df = pd.read_csv(csv_file)
                    df['model'] = model
                    parts = csv_file.split('_')
                    if len(parts) >= 3:
                        xlsx_file = '_'.join(parts[1:-3]) + '.xlsx'
                        df['source_file'] = xlsx_file
                    detailed_dfs.append(df)
                
                if detailed_dfs:
                    combined_detailed_df = pd.concat(detailed_dfs, ignore_index=True)
                    create_comparison_plots(results_df, combined_detailed_df)
                else:
                    create_comparison_plots(results_df)
            else:
                create_comparison_plots(results_df)
        else:
            print("No existing results found. Run without --skip-run to generate new results.")
    
    if 'results_df' in locals():
        print_summary_statistics(results_df) 