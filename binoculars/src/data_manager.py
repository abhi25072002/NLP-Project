import json
import pandas as pd
import numpy as np
import hashlib
from typing import List, Tuple, Dict, Optional

class DataManager:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self.load_data()
        
    def load_data(self) -> pd.DataFrame:
        """Loads data from JSONL file."""
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return pd.DataFrame(data)

    def assign_splits(self):
        """
        Assigns 'test' to all rows for zero-shot evaluation.
        """
        self.df['split'] = 'test'
        print(f"Assigned all {len(self.df)} samples to 'test' split (Zero-Shot Evaluation).")

    def get_experiment_data(self, machine_key: str = 'machine_default', 
                          human_key: str = 'human_text') -> Tuple[List[str], List[int]]:
        """
        Constructs the data for a specific experiment scenario using ALL available data.
        Returns:
            texts: List of strings (mixed human and machine)
            labels: List of integers (0 for human, 1 for machine)
        """
        # Use the entire dataframe
        subset = self.df
        
        texts = []
        labels = []
        
        for _, row in subset.iterrows():
            # Add Human Example
            if human_key in row and row[human_key]:
                texts.append(row[human_key])
                labels.append(0)
            
            # Add Machine Example
            if machine_key in row and row[machine_key]:
                texts.append(row[machine_key])
                labels.append(1)
                
        return texts, labels

    def save_splits(self, output_path: str):
        """Saves the dataframe with splits to a new JSONL file."""
        self.df.to_json(output_path, orient='records', lines=True)
        print(f"Saved data with splits to {output_path}")
