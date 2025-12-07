import argparse
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from src.binoculars import Binoculars

def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_ai_columns(data):
    if not data:
        return []
    row = data[0]
    exclude = {'id', 'split', 'prompt', 'human_text'}
    return [k for k in row.keys() if k not in exclude]

def compute_scores_batched(bino, texts, batch_size=32, compute_token_metrics=False):
    results = {"score": [], "ppl": [], "x_ppl": []}
    if compute_token_metrics:
        results["kl"] = []
        results["jsd"] = []
        
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        batch = texts[i:i+batch_size]
        # Return components now
        batch_out = bino.compute_score(batch, return_components=True, compute_token_metrics=compute_token_metrics)
        results["score"].extend(batch_out["score"])
        results["ppl"].extend(batch_out["ppl"])
        results["x_ppl"].extend(batch_out["x_ppl"])
        
        if compute_token_metrics:
            results["kl"].extend(batch_out["kl"])
            results["jsd"].extend(batch_out["jsd"])
            
    return results

def calculate_metrics(human_scores, ai_scores, 
                      human_ppl=None, ai_ppl=None,
                      human_x_ppl=None, ai_x_ppl=None,
                      human_kl=None, ai_kl=None,
                      human_jsd=None, ai_jsd=None):
    # Binoculars: Lower score = AI.
    # For ROC AUC, we usually want Higher score = Positive (AI).
    # So we use -score as the prediction score for AI class.
    
    y_true = [0] * len(human_scores) + [1] * len(ai_scores)
    y_scores = human_scores + ai_scores
    
    # Invert scores for AUC calculation so that higher is more likely AI
    # Binoculars score: low -> AI. -score: high -> AI.
    y_scores_inverted = [-s for s in y_scores]
    
    auc = roc_auc_score(y_true, y_scores_inverted)
    
    # Binoculars default threshold is for the raw score.
    # If threshold is not provided, use the model's default.
    # But here we just want to report metrics.
    # Let's calculate F1 at the default threshold if possible, or just AUC.
    # The Binoculars class has a threshold attribute.
    
    metrics = {
        "roc_auc": auc,
        "mean_human_score": np.mean(human_scores),
        "mean_ai_score": np.mean(ai_scores),
        "std_human_score": np.std(human_scores),
        "std_ai_score": np.std(ai_scores)
    }

    # Calculate TPR at specific FPR thresholds
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores_inverted)
    
    for target_fpr in [0.0001, 0.001, 0.01]:  # 0.01%, 0.1%, 1%
        # Find the first index where FPR > target_fpr
        # We want the TPR at the threshold that gives FPR <= target_fpr
        # Since fpr is increasing, we look for the last index where fpr <= target_fpr
        idx = np.where(fpr <= target_fpr)[0][-1]
        metrics[f"tpr_at_{target_fpr*100}%_fpr"] = tpr[idx]

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run Binoculars Zero-Shot Detection on Master Table")
    parser.add_argument("--data_path", type=str, required=True, help="Path to master_table.jsonl")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save metrics summary (JSON or CSV)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--observer_model", type=str, default="tiiuae/falcon-7b", help="Observer model")
    parser.add_argument("--performer_model", type=str, default="tiiuae/falcon-7b-instruct", help="Performer model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples for testing")
    parser.add_argument("--compute_token_metrics", action="store_true", help="Compute KL and JSD metrics (slower)")
    parser.add_argument("--configs", type=str, nargs="+", default=None, help="Specific AI configs (columns) to run. If None, runs all.")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)
    if args.limit:
        data = data[:args.limit]
    print(f"Loaded {len(data)} examples.")
    
    ai_columns = get_ai_columns(data)
    if args.configs:
        # Validate configs
        missing = [c for c in args.configs if c not in ai_columns]
        if missing:
            raise ValueError(f"Configs {missing} not found in data. Available: {ai_columns}")
        ai_columns = args.configs
        
    print(f"Found AI columns: {ai_columns}")
    
    print("Initializing Binoculars...")
    bino = Binoculars(
        observer_name_or_path=args.observer_model,
        performer_name_or_path=args.performer_model,
        mode="low-fpr" # Default mode
    )
    
    # 1. Compute Human Scores
    print("Computing scores for Human text...")
    human_texts = [d['human_text'] for d in data]
    human_results = compute_scores_batched(bino, human_texts, batch_size=args.batch_size, compute_token_metrics=args.compute_token_metrics)
    human_scores = human_results["score"]
    
    results = []
    detailed_records = [] # Store per-example details
    
    # 2. Compute AI Scores for each config
    for col in ai_columns:
        print(f"Computing scores for AI config: {col}...")
        ai_texts = [d[col] for d in data]
        ai_results = compute_scores_batched(bino, ai_texts, batch_size=args.batch_size, compute_token_metrics=args.compute_token_metrics)
        ai_scores = ai_results["score"]
        
        # Extract components for aggregation
        ai_ppl = ai_results["ppl"]
        ai_x_ppl = ai_results["x_ppl"]
        ai_kl = ai_results.get("kl")
        ai_jsd = ai_results.get("jsd")
        
        human_ppl = human_results["ppl"]
        human_x_ppl = human_results["x_ppl"]
        human_kl = human_results.get("kl")
        human_jsd = human_results.get("jsd")
        
        metrics = calculate_metrics(human_scores, ai_scores,
                                    human_ppl=human_ppl, ai_ppl=ai_ppl,
                                    human_x_ppl=human_x_ppl, ai_x_ppl=ai_x_ppl,
                                    human_kl=human_kl, ai_kl=ai_kl,
                                    human_jsd=human_jsd, ai_jsd=ai_jsd)
        metrics['config'] = col
        results.append(metrics)
        
        # Collect detailed records for this config
        for i in range(len(ai_scores)):
            record = {
                "id": data[i].get("id", i),
                "config": col,
                "label": "AI",
                "score": ai_scores[i],
                "ppl": ai_results["ppl"][i],
                "x_ppl": ai_results["x_ppl"][i]
            }
            if args.compute_token_metrics:
                record["kl"] = ai_results["kl"][i]
                record["jsd"] = ai_results["jsd"][i]
            detailed_records.append(record)
            
    # Add human detailed records (only once)
    for i in range(len(human_scores)):
        record = {
            "id": data[i].get("id", i),
            "config": "human",
            "label": "Human",
            "score": human_scores[i],
            "ppl": human_results["ppl"][i],
            "x_ppl": human_results["x_ppl"][i]
        }
        if args.compute_token_metrics:
            record["kl"] = human_results["kl"][i]
            record["jsd"] = human_results["jsd"][i]
        detailed_records.append(record)
        
    # 3. Save Results
    df = pd.DataFrame(results)
    # Reorder columns to put config first
    cols = ['config'] + [c for c in df.columns if c != 'config']
    df = df[cols]
    
    print("Results:")
    print(df)
    
    if args.output_path.endswith('.json'):
        df.to_json(args.output_path, orient='records', indent=2)
    else:
        df.to_csv(args.output_path, index=False)
    
    # Save detailed components
    details_path = args.output_path.replace('.csv', '_details.csv').replace('.json', '_details.csv')
    pd.DataFrame(detailed_records).to_csv(details_path, index=False)
    print(f"Saved summary to {args.output_path}")
    print(f"Saved detailed components to {details_path}")

if __name__ == "__main__":
    main()
