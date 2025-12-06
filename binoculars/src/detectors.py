from typing import List
import torch

class BaseDetector:
    def predict(self, text: str) -> float:
        raise NotImplementedError
    
    def predict_batch(self, texts: List[str]) -> List[float]:
        raise NotImplementedError

class BinocularsDetector(BaseDetector):
    def __init__(self, observation_model_name="tiiuae/falcon-7b", instructor_model_name="tiiuae/falcon-7b-instruct", use_gpu=True):
        try:
            from binoculars import Binoculars
        except ImportError:
            raise ImportError("Binoculars library not found. Please install it via 'pip install git+https://github.com/ahans30/Binoculars.git'")
        
        print(f"Initializing Binoculars with {observation_model_name} and {instructor_model_name}...")
        self.bino = Binoculars(
            observer_name_or_path=observation_model_name,
            approver_name_or_path=instructor_model_name
        )
        
    def predict(self, text: str) -> float:
        return self.bino.compute_score(text)
    
    def predict_batch(self, texts: List[str]) -> List[float]:
        # Binoculars compute_score can handle lists if implemented, but let's check the library signature.
        # Based on the README chunk, it says: "user can also pass a list of str to compute_score"
        return self.bino.compute_score(texts)

class MockDetector(BaseDetector):
    """For testing pipeline without loading heavy models."""
    def predict(self, text: str) -> float:
        import random
        return random.random()
    
    def predict_batch(self, texts: List[str]) -> List[float]:
        import random
        return [random.random() for _ in texts]
