# src/models/tfidf.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

def train_tfidf_lr(texts, labels, config):
    """
    Train a TF-IDF + Logistic Regression model with Word and Char n-grams.
    
    Args:
        texts (list): List of training texts.
        labels (list): List of labels.
        config (dict): Model configuration containing keys like:
                       'word_ngram_range', 'char_ngram_range', 'max_features', 'C', 'solver'.
        
    Returns:
        pipeline: Trained sklearn pipeline.
    """
    # Extract config
    word_ngram_range = tuple(config.get('word_ngram_range', [1, 2]))
    char_ngram_range = tuple(config.get('char_ngram_range', [2, 4]))
    max_features = config.get('max_features', 100000)
    C = config.get('C', 1.0)
    solver = config.get('solver', 'liblinear')
    
    # Heuristic: disable max_features if None, otherwise split between word/char 
    # or just use as is for each if intent is per-vectorizer. 
    # Usually "max 100k features" means total input dimension.
    # We will allocate half to likely dense char n-grams and half to word n-grams.
    mf_split = max_features // 2 if max_features else None

    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=word_ngram_range,
        max_features=mf_split,
        min_df=2 # Exclude extremely rare words to save separate space
    )
    
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=char_ngram_range,
        max_features=mf_split,
        min_df=2
    )
    
    union = FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])
    
    # Create Pipeline
    pipeline = Pipeline([
        ('features', union),
        ('clf', LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)) 
    ])
    
    # Train
    pipeline.fit(texts, labels)
    
    return pipeline
