import os
import torch

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_roberta(train_dataset, val_dataset, config):
    """
    Fine-tune RoBERTa model.
    
    Args:
        train_dataset: Training dataset (HF Dataset).
        val_dataset: Validation dataset (HF Dataset).
        config (dict): Model configuration.
        
    Returns:
        trainer: HuggingFace Trainer object.
    """
    params = config.get("supervised", {}).get("roberta", {})
    training_config = config.get("training", {})
    
    model_name = params.get("model_name", "roberta-base")
    output_dir = os.path.join(training_config.get("output_dir", "checkpoints"), "roberta")
    
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=params.get("max_seq_length", 512))
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(params.get("learning_rate", 2e-5)),
        per_device_train_batch_size=params.get("batch_size", 16),
        per_device_eval_batch_size=params.get("batch_size", 16),
        num_train_epochs=params.get("epochs", 3),
        weight_decay=params.get("weight_decay", 0.01),
        warmup_steps=params.get("warmup_steps", 500),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        seed=training_config.get("seed", 42),
        fp16=torch.cuda.is_available(), # Use FP16 if CUDA is available
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save tokenizer too
    tokenizer.save_pretrained(output_dir)
    
    return trainer
