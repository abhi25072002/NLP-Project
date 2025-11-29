# src/generation/paraphrase.py
import torch
from transformers import MarianMTModel, MarianTokenizer
import logging

logger = logging.getLogger(__name__)

class BackTranslator:
    def __init__(self, device="cuda"):
        """
        Initialize the translation models (En->De and De->En).
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing BackTranslator on {self.device}...")
        
        # En -> De
        self.model_en_de_name = "Helsinki-NLP/opus-mt-en-de"
        self.tokenizer_en_de = MarianTokenizer.from_pretrained(self.model_en_de_name)
        self.model_en_de = MarianMTModel.from_pretrained(self.model_en_de_name).to(self.device)
        
        # De -> En
        self.model_de_en_name = "Helsinki-NLP/opus-mt-de-en"
        self.tokenizer_de_en = MarianTokenizer.from_pretrained(self.model_de_en_name)
        self.model_de_en = MarianMTModel.from_pretrained(self.model_de_en_name).to(self.device)
        
    def translate(self, texts, model, tokenizer):
        """
        Helper to translate a batch of texts using a specific model/tokenizer.
        """
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            translated = model.generate(**inputs)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    def backtranslate(self, texts):
        """
        Perform back-translation (En -> De -> En).
        
        Args:
            texts (list): List of English texts.
            
        Returns:
            list: List of back-translated English texts.
        """
        # Step 1: En -> De
        german_texts = self.translate(texts, self.model_en_de, self.tokenizer_en_de)
        
        # Step 2: De -> En
        english_texts = self.translate(german_texts, self.model_de_en, self.tokenizer_de_en)
        
        return english_texts
