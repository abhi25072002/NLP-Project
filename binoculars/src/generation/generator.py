# src/generation/generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class VariantGenerator:
    def __init__(self, model_name="gpt2-xl", device="cuda"):
        """
        Initialize the generator model.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading generator model: {model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def generate(self, prompts, max_new_tokens=256, max_length=512, **kwargs):
        """
        Generate text variants.
        
        Args:
            prompts (list): List of prompt strings.
            max_new_tokens (int): Maximum new tokens to generate.
            max_length (int): Maximum length for input truncation.
            **kwargs: Generation arguments (temperature, top_p, etc.).
            
        Returns:
            list: List of generated texts (including prompt).
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts
