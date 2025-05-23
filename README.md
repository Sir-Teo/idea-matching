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

## Requirements

```bash
pip install pandas sentence-transformers tqdm scikit-learn openpyxl
```

## Usage

### Basic Usage

```bash
python match.py \
    --workbook data/your_workbook.xlsx \
    --conv_sheet "Conversation idea units" \
    --mem_sheet "Memory 1 Idea Units" \
    --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --conv_window 1 \
    --mem_window 2
```

### Parameters

- `--workbook`: Path to Excel workbook containing conversation and memory sheets
- `--conv_sheet`: Sheet name with conversation idea-units
- `--mem_sheet`: Sheet name with memory idea-units  
- `--model`: HuggingFace Cross-Encoder model (default: `cross-encoder/stsb-distilroberta-base`)
- `--conv_window`: Context window size for conversation units (default: 0)
- `--mem_window`: Context window size for memory units (default: 0)
- `--manual_col`: Column name for manual coding (default: "matching idea unit")
- `--top_k`: Number of top matches to consider (default: 3)

### Required Excel Columns

**Conversation Sheet:**
- `Transcript`: The conversation text/idea units
- `idea units #`: Unique identifier for each conversation unit

**Memory Sheet:**
- `Transcript`: The memory text/idea units  
- `idea unit #`: Unique identifier for each memory unit
- `matching idea unit` (or specify with `--manual_col`): Manual coding reference

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

## Data Privacy

This repository includes a comprehensive `.gitignore` file that excludes all data files to protect sensitive information. Data files are kept local and not tracked in version control.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here] 