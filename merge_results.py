import argparse
import pandas as pd
import os
import glob

def main():
    parser = argparse.ArgumentParser(description="Merge Binoculars Results")
    parser.add_argument("--input_pattern", type=str, required=True, help="Glob pattern for result CSVs (e.g. 'results/*.csv')")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged artifacts")
    
    args = parser.parse_args()
    
    files = glob.glob(args.input_pattern)
    if not files:
        print(f"No files found matching {args.input_pattern}")
        return
        
    print(f"Found {len(files)} result files.")
    
    dfs = []
    for f in files:
        # Skip detail files if pattern matches them
        if "_details.csv" in f:
            continue
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        print("No valid summary CSVs found.")
        return
        
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by config if possible
    if 'config' in merged_df.columns:
        merged_df = merged_df.sort_values('config')
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save merged CSV
    csv_path = os.path.join(args.output_dir, "merged_results.csv")
    merged_df.to_csv(csv_path, index=False)
    print(f"Saved merged results to {csv_path}")
    
    # Generate Markdown Report
    md_path = os.path.join(args.output_dir, "experiment_report.md")
    with open(md_path, 'w') as f:
        f.write("# Binoculars Experiment Report\n\n")
        f.write("## Summary Metrics\n\n")
        f.write(merged_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Analysis\n")
        f.write("### Top Performing Configs (by TPR@0.01% FPR)\n\n")
        
        if 'tpr_at_0.01%_fpr' in merged_df.columns:
            top_df = merged_df.sort_values('tpr_at_0.01%_fpr', ascending=False)
            f.write(top_df[['config', 'tpr_at_0.01%_fpr', 'roc_auc']].to_markdown(index=False))
        else:
            f.write("TPR@0.01% FPR metric not found.\n")
            
    print(f"Saved report to {md_path}")

if __name__ == "__main__":
    main()
