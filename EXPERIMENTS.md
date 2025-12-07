# Binoculars Experiments

This guide explains how to run the Binoculars zero-shot detection pipeline on the `master_table.jsonl` dataset.

## 1. Setup

Ensure you have the required dependencies installed. The pipeline relies on `transformers`, `torch`, `numpy`, `pandas`, and `scikit-learn`.

```bash
pip install -r requirements.txt
pip install tabulate  # Required for reporting
```

## 2. Quick Start Examples

Here are common scenarios you might want to run.

### Scenario A: Quick Sanity Check
Run on just 5 examples to verify everything works (model loading, scoring, saving).
```bash
python run_zero_shot_master_table.py \
  --data_path data/master_table.jsonl \
  --output_path results/test_run.csv \
  --limit 5
```

### Scenario B: Full Run (Standard Metrics)
Run on the entire dataset, computing Binoculars Score, PPL, and X-PPL.
```bash
python run_zero_shot_master_table.py \
  --data_path data/master_table.jsonl \
  --output_path results/full_standard.csv
```

### Scenario C: Full Run with Robust Metrics (Slower)
Run on the entire dataset, but also compute **KL Divergence** and **Jensen-Shannon Divergence**. This is useful for deep analysis of distributional shift.
```bash
python run_zero_shot_master_table.py \
  --data_path data/master_table.jsonl \
  --output_path results/full_robust.csv \
  --compute_token_metrics
```

### Scenario D: Run Specific Experiments Only
If you only want to evaluate specific generation strategies (e.g., just the greedy decoding column).
```bash
python run_zero_shot_master_table.py \
  --data_path data/master_table.jsonl \
  --output_path results/greedy_only.csv \
  --configs machine_greedy
```

### Scenario E: Merge Results from Multiple Runs
If you ran experiments separately (e.g., `results/greedy.csv` and `results/beam.csv`), merge them into one report.
```bash
python merge_results.py \
  --input_pattern "results/*.csv" \
  --output_dir results/merged
```

---

## 3. Detailed Usage

The main entry point is `run_zero_shot_master_table.py`.

### Arguments
*   `--data_path`: Path to the input JSONL file (required).
*   `--output_path`: Path to save the summary CSV (required).
*   `--limit`: (Optional) Number of examples to process. Useful for testing.
*   `--configs`: (Optional) List of specific AI columns to process. Default is all.
*   `--compute_token_metrics`: (Optional) Flag to enable KL/JSD calculation.
*   `--batch_size`: (Optional) Batch size for inference. Default 16.

## 4. Output Format

The script generates two files for every run:

1.  **Summary CSV** (`output_path.csv`):
    *   Aggregated metrics for each configuration.
    *   Metrics: `roc_auc`, `mean_human_score`, `mean_ai_score`, `tpr_at_0.01%_fpr`, `tpr_at_1.0%_fpr`.

2.  **Details CSV** (`output_path_details.csv`):
    *   Row-level results for every example.
    *   Columns: `id`, `config`, `label`, `score`, `ppl`, `x_ppl`, `kl` (optional), `jsd` (optional).

## 5. Merging Results

The `merge_results.py` script aggregates all results in a directory into a single CSV and a Markdown report.

This will produce:
*   `merged_results.csv`: A single CSV with all summary metrics.
*   `experiment_report.md`: A formatted report comparing configurations, including a "Top Performers" table based on TPR@0.01% FPR.
