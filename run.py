from contextlib import contextmanager
import logging
import json
from pathlib import Path
from typing import Dict, List
import argparse
import sys
import modal
import os
import pickle

from src.data_loader import DataLoader
from src.model import LLaMAModel

from Supervised_learning_model.predict import predict, convert_to_json

# Environment detection to determine if running on Modal
IS_MODAL = os.environ.get("MODAL_ENVIRONMENT") == "modal"

EVALUATOR_PATH = "/root/evaluator.pkl"

def stub_init_model(augment_prompt=True):
    """Initialize model and evaluator during image building"""
    from src.data_loader import DataLoader
    from src.model import LLaMAModel
    
    # Initialize evaluator
    evaluator = Evaluator(augment_prompt)
    
    # Force model loading and caching during build
    evaluator.model.tokenizer
    evaluator.model.model
    evaluator.data_loader.load_data(max_test_samples=1)
    
    # Save the initialized evaluator to a persistent path
    with open(EVALUATOR_PATH, 'wb') as f:
        pickle.dump(evaluator, f)

    return evaluator

def load_evaluator():
    """Load the pre-initialized evaluator"""
    with open(EVALUATOR_PATH, 'rb') as f:
        return pickle.load(f)

# Create image with pre-initialized evaluator
image = (modal.Image.debian_slim()
         .pip_install("transformers", "datasets")
         .pip_install_from_requirements("requirements.txt")
         .copy_local_file("word2idx.json", remote_path="/root/word2idx.json")
         .copy_local_file("tag2idx.json", remote_path="/root/tag2idx.json")
         .copy_local_file("bilstm_model.pth", remote_path="/root/bilstm_model.pth")
         .run_function(stub_init_model))

# Update app with new image
app = modal.App(
    "common-run-py",
    image=image,
    mounts=[modal.Mount.from_local_dir("src", remote_path="/src"),
            modal.Mount.from_local_dir("Supervised_learning_model", remote_path="/Supervised_learning_model")]
)

# Logging configuration
def setup_logging(log_file: str = None):
    """Configure logging to both file and console"""
    log_path = "/tmp/evaluation.log" if IS_MODAL else log_file or "logs/evaluation.log"
    # Ensure logs directory exists locally
    if not IS_MODAL and log_file != "/tmp/evaluation.log":
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )

# Evaluation logic class
class Evaluator:
    def __init__(self, augment_prompt):
        self.logger = logging.getLogger(__name__)
        self.augment_prompt = augment_prompt
        self.model = LLaMAModel("TinyLlama/TinyLlama_v1.1_math_code")
        self.data_loader = DataLoader()

        if self.augment_prompt:
            self.logger.info("Prompt augmentation is enabled.")
            with open("word2idx.json", "r") as f:
                self.word2idx = json.load(f)
            with open("tag2idx.json", "r") as f:
                self.tag2idx = json.load(f)

    def evaluate_sample_by_index(self, index: int) -> Dict:
        """Evaluate and return results for a specific sample based on index"""
        self.data_loader.load_data(max_test_samples=index + 1)  # Load up to the specified index
        _, test_data = self.data_loader.get_train_test_data()

        if self.augment_prompt:
            augmented = predict(test_data[index]["input"], self.word2idx, self.tag2idx)
            augmented_json = convert_to_json(augmented)
            # Add the augmented context to the input
            test_data[index]["input"] += f" {augmented_json['entities']}"


        if index >= len(test_data):
            self.logger.error(f"Index {index} is out of range.")
            return {}

        example = test_data[index]
        question = example["input"]
        expected = example["target"]
        predicted = self.model.generate_answer(
            question,
            max_new_tokens=180
        )

        # Collect and return results for the specific sample
        result = {
            "index": index,
            "question": question,
            "expected": expected,
            "predicted": predicted,
        }
        self.logger.info(f"\nSample {index + 1}:")
        self.logger.info(f"Question: {question}")
        self.logger.info(f"Expected: {expected}")
        self.logger.info(f"Predicted: {predicted}")
        self.logger.info("-" * 80)

        return result

    def evaluate_metrics(self, num_samples: int = 5) -> Dict:
        """Evaluate model performance using multiple metrics"""
        self.data_loader.load_data(max_test_samples=num_samples)
        _, test_data = self.data_loader.get_train_test_data()

        self.logger.info(f"\nEvaluating {num_samples} samples for metrics...")
        metrics = ["rouge", "bleu"]
        # Assuming the model supports these metrics, otherwise adjust accordingly
        results = self.model.evaluate(test_data, metrics=metrics)

        self.logger.info("\nEvaluation Results:")
        self.logger.info(json.dumps(results, indent=2))
        return results

# Cloud evaluation function on Modal
@app.function()
def cloud_evaluation(index: int) -> Dict:
    """Function to run evaluation on Modal Cloud for a specific sample based on index"""
    setup_logging("/tmp/evaluation.log")
    # Load the pre-initialized evaluator from persistent storage
    evaluator = load_evaluator()
    return evaluator.evaluate_sample_by_index(index=index)

@contextmanager
def run():
    """Main entry point for both local and cloud runs"""
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--log-file", type=str, default="logs/evaluation.log",
                        help="Path to log file (default: logs/evaluation.log)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples to evaluate (default: 5)")
    parser.add_argument("--cloud", action="store_true",
                        help="Run evaluation on Modal Cloud (default: False)")
    parser.add_argument("--augment_prompt", action="store_true",
                        help="Augment the prompt with additional context (default: False)")
    args = parser.parse_args()

    # Automatically set --cloud to True when running on Modal
    if IS_MODAL:
        args.cloud = True
        print("Running on Modal Cloud.")

    # Setup logging
    setup_logging(args.log_file)

    all_results = []  # This will store all evaluation results

    if args.cloud:
        # Run on Modal Cloud in parallel using map
        with app.run():
            # Using map to apply the cloud_evaluation function to multiple indices concurrently
            results = list(cloud_evaluation.map(range(args.num_samples)))
            all_results.extend(results)  # Collect the results from cloud evaluation
    else:
        # Run locally, process samples sequentially by index
        evaluator = Evaluator(augment_prompt=args.augment_prompt)
        for idx in range(args.num_samples):
            result = evaluator.evaluate_sample_by_index(index=idx)
            all_results.append(result)

    # After collecting all results, log them
    logging.info("Aggregated Results:")
    logging.info(json.dumps(all_results, indent=2))

# Ensure the script runs with the correct entry point
if __name__ == "__main__":
    run()
