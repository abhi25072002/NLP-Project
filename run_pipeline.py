import argparse
import os
import pandas as pd
from src.data_manager import DataManager
from src.detectors import BinocularsDetector, MockDetector
from src.evaluator import Evaluator
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run Binoculars Ablation Pipeline")
    parser.add_argument("--data_path", type=str, default="data/master_table (1).jsonl", help="Path to master table JSONL")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--mock", action="store_true", help="Use MockDetector for testing pipeline logic")
    parser.add_argument("--experiments", type=str, nargs='+', default=["default"], 
                        help="List of experiments to run. Options: default, knobs, bt, all. "
                             "Or specify specific machine keys like 'machine_T0.7_p0.9'")
    
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    dm = DataManager(args.data_path)
    dm.assign_splits()
    
    # 2. Initialize Detector
    if args.mock:
        print("Using MockDetector...")
        detector = MockDetector()
    else:
        print("Using BinocularsDetector (this may take time to load models)...")
        detector = BinocularsDetector()
        
    # 3. Define Experiments
    # Map 'experiment names' to machine keys
    # We can inspect the dataframe to find available keys, but for now let's define some standard ones based on the file content I saw earlier.
    # From file view: machine_default, machine_T0.7_p0.8, machine_T0.7_p0.9, machine_greedy, etc.
    
    available_keys = [col for col in dm.df.columns if col.startswith("machine_")]
    print(f"Available machine keys in data: {available_keys}")
    
    experiments_to_run = []
    
    if "all" in args.experiments:
        experiments_to_run = available_keys
    elif "knobs" in args.experiments:
        experiments_to_run = [k for k in available_keys if "machine_T" in k or "greedy" in k or "beam" in k]
    elif "default" in args.experiments:
        experiments_to_run = ["machine_default"]
    else:
        # Assume user passed specific keys
        experiments_to_run = args.experiments

    # Add BT experiments if requested
    if "bt" in args.experiments:
        # Check for BT keys
        if "machine_default_BT" in dm.df.columns:
            experiments_to_run.append("machine_default_BT")
            
    print(f"Running experiments: {experiments_to_run}")
    
    # 4. Run Loop
    results = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for machine_key in experiments_to_run:
        if machine_key not in dm.df.columns:
            print(f"Warning: Key {machine_key} not found in dataframe. Skipping.")
            continue
            
        print(f"\n--- Running Experiment: Human vs {machine_key} ---")
        
        # Get Test Data
        try:
            texts, labels = dm.get_test_set(split_name='test', machine_key=machine_key)
        except ValueError as e:
            print(e)
            continue
            
        print(f"Test set size: {len(texts)}")
        
        # Run Detection
        # Process in batches to avoid OOM if list is huge, though Binoculars might handle it.
        # Let's do a simple batching here just in case.
        batch_size = 32
        scores = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
            batch_texts = texts[i:i+batch_size]
            batch_scores = detector.predict_batch(batch_texts)
            scores.extend(batch_scores)
            
        # Evaluate
        metrics = Evaluator.compute_metrics(labels, scores)
        metrics['Experiment'] = machine_key
        print(f"Results for {machine_key}: {metrics}")
        results.append(metrics)
        
    # 5. Save Results
    if results:
        results_df = pd.DataFrame(results)
        output_file = os.path.join(args.output_dir, "results.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\nAll results saved to {output_file}")
        print(results_df)
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
