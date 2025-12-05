# src/data/loader.py

def load_turingbench(split="train", cache_dir=None):
    """
    Load the TuringBench dataset.
    
    Args:
        split (str): Dataset split to load.
        cache_dir (str): Directory to cache the dataset.
        
    Returns:
        Dataset object.
    """
    from datasets import load_dataset
    # Default to the path found in config if not provided, but here we hardcode or use a default
    dataset_path = "turingbench/TuringBench" 
    return load_dataset(dataset_path, split=split, cache_dir=cache_dir)
