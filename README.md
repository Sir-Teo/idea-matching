# Idea Matching Project

A Python script for automatically matching memory idea units to conversation idea units using Cross-Encoder models from HuggingFace.

## Description

This project implements an automated approach to align memory idea-units with conversation idea-units using state-of-the-art Cross-Encoder models. The script ranks every memory idea-unit window against conversation idea-unit windows and compares the automatic rankings with manual coding to evaluate performance.

## Features

- **Flexible Window Contexts**: Configure context windows around idea units for both conversation and memory data
- **Cross-Encoder Ranking**: Uses HuggingFace Cross-Encoder models for semantic similarity scoring
- **Robust Column Matching**: Automatically handles variations in column names (plural/singular, spacing)
- **Comprehensive Analysis**: Provides overlap analysis, accuracy metrics, and detailed statistics
- **Excel Integration**: Reads directly from Excel workbooks with multiple sheets
- **Batch Processing**: Process multiple corrected files automatically with `batch_process_corrected.py`
- **Model Comparison**: Compare different Cross-Encoder models with `compare_models.py`

## Requirements

```bash
pip install pandas sentence-transformers tqdm scikit-learn openpyxl
```

## Usage

### Single File Processing

```bash
python match.py \
    --workbook data/your_workbook.xlsx \
    --conv_sheet "Conversation idea units" \
    --mem_sheet "Memory 1 Idea Units" \
    --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --conv_window 1 \
    --mem_window 2
```

### Batch Processing for Corrected Files

For processing multiple corrected files in `data/cleaned_corrections/`:

```bash
python batch_process_corrected.py \
    --model cross-encoder/stsb-distilroberta-base \
    --conv_window 0 \
    --mem_window 0 \
    --data_dir data/cleaned_corrections \
    --output_dir results_corrected
```

### Model Comparison

To compare multiple Cross-Encoder models across all files:

```bash
python compare_models.py
```

### Parameters

#### Single File Processing (`match.py`)
- `--workbook`: Path to Excel workbook containing conversation and memory sheets
- `--conv_sheet`: Sheet name with conversation idea-units
- `--mem_sheet`: Sheet name with memory idea-units  
- `--model`: HuggingFace Cross-Encoder model (default: `cross-encoder/stsb-distilroberta-base`)
- `--conv_window`: Context window size for conversation units (default: 0)
- `--mem_window`: Context window size for memory units (default: 0)
- `--manual_col`: Column name for manual coding (default: "matching idea unit")
- `--top_k`: Number of top matches to consider (default: 3)

#### Batch Processing (`batch_process_corrected.py`)
- `--model`: HuggingFace Cross-Encoder model (default: `cross-encoder/stsb-distilroberta-base`)
- `--conv_window`: Context window size for conversation units (default: 0)
- `--mem_window`: Context window size for memory units (default: 0)
- `--top_k`: Number of top matches to consider (default: 3)
- `--data_dir`: Directory containing corrected files (default: `data/cleaned_corrections`)
- `--output_dir`: Directory to save results (default: `results_corrected`)

### Required Excel Columns

#### Original Format (for `match.py`)

**Conversation Sheet:**
- `Transcript`: The conversation text/idea units
- `idea units #`: Unique identifier for each conversation unit

**Memory Sheet:**
- `Transcript`: The memory text/idea units  
- `idea unit #`: Unique identifier for each memory unit
- `matching idea unit` (or specify with `--manual_col`): Manual coding reference

#### Corrected Format (for `batch_process_corrected.py`)

**Both Conversation and Memory Files:**
- `Subject Pair`: Subject identifier
- `Original Turn`: Turn number from original conversation
- `Idea Unit #`: Unique identifier for each idea unit
- `Transcript`: The processed idea unit text

## Output

The script generates a CSV file with the following columns:

- `memory unit #`: ID from memory sheet
- `memory sentence`: Memory window text used as query
- `manual conversation unit #`: Ground-truth conversation unit ID
- `auto conversation unit #`: Top-ranked conversation unit ID  
- `score`: Cross-Encoder similarity score
- `is_match`: 1 if auto matches manual, 0 otherwise
- `auto conversation window`: Conversation text window from model
- `overlap_count`: Number of overlapping words (excluding stop words)
- `overlap_jaccard`: Jaccard similarity coefficient

## Models

The script supports any Cross-Encoder model from HuggingFace. Recommended models:

- `cross-encoder/ms-marco-MiniLM-L-12-v2` (efficient, good performance)
- `cross-encoder/stsb-distilroberta-base` (default, balanced)
- `cross-encoder/stsb-roberta-large` (highest accuracy, slower)

## Analysis Features

- **Accuracy Metrics**: Top-k accuracy calculation
- **Score Statistics**: Mean, median, min, max similarity scores
- **Word Overlap Analysis**: Most frequent overlapping words across all comparisons
- **Jaccard Similarity**: Token-level similarity between memory and conversation windows

## Project Structure

```
idea_matching/
├── match.py                     # Single file processing script
├── batch_process_corrected.py   # Batch processing for corrected files
├── compare_models.py           # Model comparison script
├── generate_ideas_pipeline.ipynb # Jupyter notebook for idea extraction
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Data directory (excluded from git)
│   ├── cleaned_corrections/    # Corrected/preprocessed files
│   └── *.xlsx                 # Original Excel files
└── results_corrected/         # Results directory (excluded from git)
    ├── batch_processing_summary.csv
    ├── all_detailed_results.csv
    └── result_*.csv           # Individual result files
```

## Data Privacy

This repository includes a comprehensive `.gitignore` file that excludes all data files to protect sensitive information. Data files are kept local and not tracked in version control.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here] 