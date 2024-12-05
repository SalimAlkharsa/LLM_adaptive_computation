import logging
from datasets import load_dataset
from typing import Tuple, List, Dict

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dataset = None
        self.train_data = None
        self.test_data = None

    def load_data(self, max_train_samples: int = 100, max_test_samples: int = 1319) -> None:
        self.logger.info("Loading GSM8K dataset...")
        self.dataset = load_dataset("gsm8k", "socratic")
        
        # Get train/test splits with size limits
        self.train_data = self.dataset["train"].select(range(max_train_samples))
        self.test_data = self.dataset["test"].select(range(max_test_samples))  # Note: using test split now
        
        self.logger.info(f"Loaded {len(self.train_data)} train and {len(self.test_data)} test samples")

    def prepare_model_inputs(self, split: str = "train") -> List[Dict[str, str]]:
        """Format data for T5Model input"""
        data = self.train_data if split == "train" else self.test_data
        if data is None:
            raise ValueError("Please load data first using load_data()")
            
        formatted_data = []
        for item in data:
            formatted_data.append({
                "input": item["question"],  # Changed from problem to question
                "target": item["answer"]    # Changed from expected_answer to answer
            })
        return formatted_data

    def get_train_test_data(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Get formatted data for both training and testing"""
        return (
            self.prepare_model_inputs("train"),
            self.prepare_model_inputs("test")
        )
