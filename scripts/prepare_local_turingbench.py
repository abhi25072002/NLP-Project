#!/usr/bin/env python3
"""
prepare_local_turingbench.py

This script handles the extraction and consolidation of the local TuringBench zip file.
It is needed because the Hugging Face dataset script is no longer supported.

Steps:
1. Unzip `data/turingbench.zip` (or user provided path).
2. Walk through the extracted directories (20 generators).
3. Read all `train.csv`, `test.csv`, `valid.csv`.
4. Filter for rows where label is 'human'.
5. Consolidate into a single `data/human_turingbench.csv`.
"""

import argparse
import logging
import os
import zipfile
import pandas as pd
import glob
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare local TuringBench dataset")
    parser.add_argument("--zip_path", type=str, default="data/turingbench.zip", help="Path to turingbench.zip")
    parser.add_argument("--output_dir", type=str, default="data/raw_turingbench", help="Directory to unzip to")
    parser.add_argument("--final_output", type=str, default="data/human_turingbench.csv", help="Final consolidated CSV")
    args = parser.parse_args()

    # Load config
    config_path = "configs/data_config.yaml"
    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            # Override defaults with config if not provided via args
            if args.zip_path == "data/turingbench.zip":
                args.zip_path = config.get("local", {}).get("zip_path", args.zip_path)
            if args.output_dir == "data/raw_turingbench":
                args.output_dir = config.get("local", {}).get("extract_dir", args.output_dir)
            if args.final_output == "data/human_turingbench.csv":
                args.final_output = config.get("local", {}).get("consolidated_csv", args.final_output)
    
    if not os.path.exists(args.zip_path):
        logger.error(f"Zip file not found at {args.zip_path}. Please upload it.")
        return

    # 1. Unzip
    logger.info(f"Unzipping {args.zip_path} to {args.output_dir}...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    with zipfile.ZipFile(args.zip_path, 'r') as zip_ref:
        zip_ref.extractall(args.output_dir)
        
    # 2. Walk and Collect
    logger.info("Scanning for CSV files...")
    # The structure is likely: output_dir/TuringBench/<Generator>/<split>.csv or similar
    # We'll just look for all csvs recursively
    csv_files = glob.glob(os.path.join(args.output_dir, "**", "*.csv"), recursive=True)
    
    logger.info(f"Found {len(csv_files)} CSV files.")
    
    all_human_data = []
    
    for csv_file in csv_files:
        try:
            # Try reading with pandas
            # Based on user input, columns might be 'Generation' and 'label'
            # Separator might be comma or tab. Let's try comma first, then tab if needed.
            # User sample looked like tab or space separated, but filename is .csv
            
            # Attempt to sniff separator
            df = pd.read_csv(csv_file)
            
            # Check columns
            # User sample: "Generation", "label"
            if "Generation" not in df.columns or "label" not in df.columns:
                # Try tab separator
                df = pd.read_csv(csv_file, sep='\t')
            
            if "Generation" in df.columns and "label" in df.columns:
                # Filter for human
                # User sample showed 'human' in label column
                human_rows = df[df['label'] == 'human']
                
                if not human_rows.empty:
                    # Keep only necessary columns and rename for consistency if we want
                    # But let's keep original for now and handle mapping in loader
                    all_human_data.append(human_rows[['Generation', 'label']])
            else:
                logger.warning(f"Skipping {csv_file}: Could not find 'Generation' and 'label' columns.")
                
        except Exception as e:
            logger.warning(f"Error reading {csv_file}: {e}")

    if not all_human_data:
        logger.error("No human data found! Check the CSV format.")
        return

    # 3. Consolidate
    logger.info("Consolidating data...")
    final_df = pd.concat(all_human_data, ignore_index=True)
    
    # Remove duplicates if any (human data might be repeated across splits if not careful, but usually distinct)
    final_df.drop_duplicates(subset=['Generation'], inplace=True)
    
    logger.info(f"Found {len(final_df)} unique human samples.")
    
    # 4. Save
    logger.info(f"Saving to {args.final_output}...")
    final_df.to_csv(args.final_output, index=False)
    logger.info("Done!")

    # Optional: Cleanup extracted files to save space?
    # shutil.rmtree(args.output_dir)

if __name__ == "__main__":
    main()
