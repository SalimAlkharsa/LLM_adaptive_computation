import logging
from model import LLaMAModel
from data_loader import DataLoader
from typing import Dict, List
import json
import argparse
import sys
from pathlib import Path

def setup_logging(log_file: str = None):
    """Configure logging to both file and console"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

class Evaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = LLaMAModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # Specify model explicitly
        self.data_loader = DataLoader()

    def evaluate_samples(self, num_samples: int = 5) -> None:
        """Evaluate and log detailed results for a small number of samples"""
        self.data_loader.load_data(max_test_samples=num_samples)
        _, test_data = self.data_loader.get_train_test_data()

        self.logger.info(f"\nEvaluating {num_samples} samples with detailed output:")
        for idx, example in enumerate(test_data):
            question = example["input"]
            expected = example["target"]
            predicted = self.model.generate_answer(
                question,
                max_new_tokens=256  # Shorter response length for testing
            )
            
            self.logger.info(f"\nSample {idx + 1}:")
            self.logger.info(f"Question: {question}")
            self.logger.info(f"Expected: {expected}")
            self.logger.info(f"Predicted: {predicted}")
            self.logger.info("-" * 80)  # Add separator for clarity


    def evaluate_metrics(self, num_samples: int = 5) -> Dict:
        """Evaluate model performance using multiple metrics"""
        self.data_loader.load_data(max_test_samples=num_samples)
        _, test_data = self.data_loader.get_train_test_data()

        self.logger.info(f"\nEvaluating {num_samples} samples for metrics...")
        metrics = ["rouge", "bleu"]
        results = self.model.evaluate(test_data, metrics=metrics)
        
        self.logger.info("\nEvaluation Results:")
        self.logger.info(json.dumps(results, indent=2))
        return results

def main():
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--log-file', type=str, default='logs/evaluation.log',
                      help='Path to log file (default: logs/evaluation.log)')
    parser.add_argument('--num-samples', type=int, default=5,
                      help='Number of samples to evaluate (default: 5)')
    args = parser.parse_args()
    
    # Setup logging before any other operations
    setup_logging(args.log_file)
    
    evaluator = Evaluator()
    evaluator.evaluate_samples(num_samples=args.num_samples)
    
if __name__ == "__main__":
    main()
