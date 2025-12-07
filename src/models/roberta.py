from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

def get_roberta_model(model_name="roberta-base", num_labels=2):
    """
    Initialize RoBERTa model for sequence classification.
    """
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model

def get_tokenizer(model_name="roberta-base"):
    """
    Initialize RoBERTa tokenizer.
    """
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    return tokenizer
