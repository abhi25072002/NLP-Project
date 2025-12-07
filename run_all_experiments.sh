#!/bin/bash

# Exit on error
set -e

# Directory setup
DATA_PATH="data/master_table.jsonl"
OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"

echo "Starting Binoculars Experiments..."

# List of all configurations
CONFIGS=(
    "machine_default"
    "machine_T0.7_p0.8"
    "machine_T0.7_p0.9"
    "machine_T0.7_p0.95"
    "machine_T0.7_p0.99"
    "machine_greedy"
    "machine_beam_search"
    "machine_top_k_50"
    "machine_T0.1_p0.95"
    "machine_T0.3_p0.95"
    "machine_T0.5_p0.95"
    "machine_T1.0_p0.95"
    "machine_T1.2_p0.95"
    "machine_T1.5_p0.95"
    "machine_default_BT"
    "human_BT"
)

# Loop through each config and run the pipeline
for config in "${CONFIGS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running experiment for config: $config"
    echo "----------------------------------------------------------------"
    
    python run_zero_shot_master_table.py \
        --data_path "$DATA_PATH" \
        --output_path "$OUTPUT_DIR/$config.csv" \
        --configs "$config" \
        --compute_token_metrics \
        --limit 5

        
    echo "Finished $config"
done

echo "----------------------------------------------------------------"
echo "Merging all results..."
echo "----------------------------------------------------------------"

python merge_results.py \
    --input_pattern "$OUTPUT_DIR/*.csv" \
    --output_dir "$OUTPUT_DIR/merged"

echo "All experiments completed successfully!"
echo "Report available at $OUTPUT_DIR/merged/experiment_report.md"

