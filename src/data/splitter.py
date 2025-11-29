# src/data/splitter.py

def create_splits(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Create train/val/test splits ensuring prompt-level separation.
    
    Args:
        dataset: Input dataset.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.
        seed (int): Random seed.
        
    Returns:
        dict: Dictionary containing 'train', 'val', 'test' lists of IDs.
    """
    pass
