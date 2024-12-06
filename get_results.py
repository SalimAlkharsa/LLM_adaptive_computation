import re
from rouge_score import rouge_scorer
import logging
from collections import defaultdict

def extract_numerical_answer(text):
    """Extract the numerical answer from a text string."""
    if not text:
        return None
    # Find all numbers in the text
    numbers = re.findall(r'####\s*(\d+)', text)
    return int(numbers[0]) if numbers else None

def calculate_accuracy(predictions, ground_truth):
    """Calculate the accuracy of predictions compared to ground truth."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    # Filter out invalid predictions (-1)
    valid_pairs = [(p, g) for p, g in zip(predictions, ground_truth) if p != -1]
    if not valid_pairs:
        return 0.0
        
    predictions_filtered, ground_truth_filtered = zip(*valid_pairs)
    correct = sum(1 for p, g in zip(predictions_filtered, ground_truth_filtered) if p == g)
    return correct / len(predictions_filtered)

def calculate_rouge_scores(predictions, references):
    """Calculate ROUGE scores for predictions against references."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = defaultdict(list)
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(pred, ref)
        for key, value in score.items():
            scores[key].append(value.fmeasure)
    
    # Average scores
    return {k: sum(v)/len(v) for k, v in scores.items()}

def parse_log_file(file_path):
    """Parse the log file to extract predictions and ground truth."""
    predictions = []
    ground_truth = []
    pred_texts = []
    truth_texts = []
    
    with open(file_path, 'r') as f:
        content = f.read()
        samples = content.split('Sample')
        
        for sample in samples[1:]:  # Skip first split as it's header
            # Extract ground truth
            truth_match = re.search(r'Expected:(.*?)####\s*(\d+)', sample, re.DOTALL)
            if truth_match:
                ground_truth.append(int(truth_match.group(2)))
                truth_texts.append(truth_match.group(1).strip())
            
            # Extract prediction
            pred_match = re.search(r'Predicted:(.*?)-{80}', sample, re.DOTALL)
            if pred_match:
                pred_text = pred_match.group(1).strip()
                pred_texts.append(pred_text)
                
                # Try to extract numerical answer from prediction
                num_answer = extract_numerical_answer(pred_text)
                predictions.append(num_answer if num_answer is not None else -1)
    
    return predictions, ground_truth, pred_texts, truth_texts

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parse log file
    file_path = '/Users/salimalkharsa/Desktop/Projects/LLM_adaptive_computation/logs/first_batch.log'
    predictions, ground_truth, pred_texts, truth_texts = parse_log_file(file_path)
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(pred_texts, truth_texts)
    logger.info("ROUGE Scores:")
    for metric, score in rouge_scores.items():
        logger.info(f"{metric}: {score:.4f}")
        
    # Log some statistics
    logger.info(f"Total samples processed: {len(ground_truth)}")
    logger.info(f"Number of valid predictions: {sum(1 for p in predictions if p != -1)}")

if __name__ == "__main__":
    main()
