import re
from rouge_score import rouge_scorer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import torch
from collections import defaultdict
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility Functions
def extract_numerical_answer(text):
    """Extract the numerical answer from a text string."""
    if not text:
        return None
    # Find all numbers preceded by "####"
    numbers = re.findall(r'####\s*(\d+)', text)
    return int(numbers[0]) if numbers else None


def calculate_accuracy(predictions, ground_truth):
    """Calculate the accuracy of numerical answers."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(ground_truth)


def calculate_rouge_scores(predictions, references):
    """Calculate ROUGE scores for step-by-step solutions."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = defaultdict(list)

    for pred, ref in zip(predictions, references):
        score = scorer.score(pred, ref)
        for key, value in score.items():
            scores[key].append(value.fmeasure)
    
    # Average scores
    return {k: np.mean(v) for k, v in scores.items()}


def calculate_perplexity(predictions, model_name="gpt2"):
    """Calculate perplexity scores for the predicted solutions."""
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()

    perplexities = []
    for text in predictions:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss  # Average negative log-likelihood
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
    
    return perplexities


def calculate_redundancy(predictions):
    """Calculate redundancy by counting the number of unnecessary steps."""
    step_counts = [len(re.findall(r'Step \d+::', pred)) for pred in predictions]
    avg_steps = np.mean(step_counts)
    return avg_steps


def calculate_semantic_similarity(predictions, references):
    """Calculate semantic similarity using sentence embeddings."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    similarities = []

    for pred, ref in zip(predictions, references):
        pred_embedding = model.encode(pred, convert_to_tensor=True)
        ref_embedding = model.encode(ref, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(pred_embedding, ref_embedding).item()
        similarities.append(similarity)
    
    return np.mean(similarities)


def calculate_cohesion_readability(predictions):
    # This is actually the average number of words per sentence, I just didnt bother renaming the function
    cohesion_scores = []

    for pred in predictions:
        score = sum(len(sentence.split()) for sentence in pred.split('.')) / len(pred.split('.'))
        cohesion_scores.append(score)
    
    return np.mean(cohesion_scores)


# Evaluation Function
def evaluate(expected_texts, predicted_texts):
    """
    Evaluate the models' outputs based on multiple metrics.
    Args:
        parsed_logs (list of dict): Parsed log entries containing:
            - 'Expected': Ground truth solutions (full text).
            - 'Predicted': Model-predicted solutions (full text).
    """
    # Take the predicted from after "Solution:", remove stuff before that
    predicted_texts = [text.split("Solution:")[1].strip() for text in predicted_texts]

    # 2. ROUGE Scores
    rouge_scores = calculate_rouge_scores(predicted_texts, expected_texts)
    logger.info("ROUGE Scores:")
    for metric, score in rouge_scores.items():
        logger.info(f"{metric}: {score:.4f}")

    # 3. Perplexity
    logger.info("Calculating Perplexity...")
    perplexity_scores = calculate_perplexity(predicted_texts)
    avg_perplexity = np.mean(perplexity_scores)
    logger.info(f"Average Perplexity: {avg_perplexity:.4f}")

    # 4. Redundancy
    avg_steps = calculate_redundancy(predicted_texts)
    logger.info(f"Average Number of Steps: {avg_steps:.2f}")

    # 5. Semantic Similarity
    semantic_similarity = calculate_semantic_similarity(predicted_texts, expected_texts)
    logger.info(f"Semantic Similarity: {semantic_similarity:.4f}")

    # 6. Cohesion and Readability
    cohesion_readability = calculate_cohesion_readability(predicted_texts)
    logger.info(f"Cohesion and Readability: {cohesion_readability:.4f}")

    return {
        "rouge_scores": rouge_scores,
        "avg_perplexity": avg_perplexity,
        "avg_steps": avg_steps,
        "semantic_similarity": semantic_similarity,
        "cohesion_readability": cohesion_readability,
    }


def parse_log_file(file_path):
    """
    Parse the log file to extract Expected and Predicted entries.
    Args:
        file_path (str): Path to the log file.
    Returns:
        List[dict]: A list of dictionaries with 'Expected' and 'Predicted' keys.
    """
    parsed_logs = []

    with open(file_path, 'r') as f:
        content = f.read()
        samples = content.split('----------------------------------------')  # Split by delimiter
        
        for sample in samples:
            sample = sample.strip()
            
            # Extract Expected section
            expected_match = re.search(r'Expected:\n(.*?)\n####', sample, re.DOTALL)
            expected = expected_match.group(1).strip() if expected_match else None
            
            # Extract Predicted section
            predicted_match = re.search(r'Predicted:\n(.*?)\n(?:2024|\Z)', sample, re.DOTALL)
            predicted = predicted_match.group(1).strip() if predicted_match else None

            if expected and predicted:
                parsed_logs.append({
                    "Expected": expected,
                    "Predicted": predicted
                })
    
    return parsed_logs


# Example Main Function
def main():
    # log_path = 'PARSED_FINAL.log'  # Update with the path to the log file
    log_path = 'PARSED_BASELINE.log'  

    # Parse the log file
    logger.info(f"Parsing log file: {log_path}")
    parsed_logs = parse_log_file(log_path)
    logger.info(f"Parsed {len(parsed_logs)} samples from log.")

    # Extract expected texts and predicted texts from parsed logs
    expected_texts = [log['Expected'] for log in parsed_logs]
    predicted_texts = [log['Predicted'] for log in parsed_logs]

    # Evaluate the parsed data
    results = evaluate(expected_texts, predicted_texts)

    # Log results
    logger.info("Evaluation Complete.")
    logger.info(f"Final Metrics: {results}")

if __name__ == "__main__":
    main()
