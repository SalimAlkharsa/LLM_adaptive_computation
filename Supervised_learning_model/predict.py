import torch
from torch.nn.utils.rnn import pad_sequence
from model import BiLSTMTagger
import json

# Load vocab mappings
with open("word2idx.json", "r") as f:
    word2idx = json.load(f)

with open("tag2idx.json", "r") as f:
    tag2idx = json.load(f)

# Load model
model = BiLSTMTagger(len(word2idx), len(tag2idx))
model.load_state_dict(torch.load("bilstm_model.pth"))
model.eval()

def convert_to_json(tagged_output):
    structured_data = {
        "entities": [],
        "quantities": {},
        "relationships": [],
        "goal": None,
    }
    
    current_relationship = {}
    goal_tokens = []

    for token, tag in tagged_output:
        if tag == "Entity":
            structured_data["entities"].append(token)
        elif tag == "Quantity":
            # Add quantity to the current relationship or standalone
            current_relationship["quantity"] = int(token) if token.isdigit() else token
        elif tag == "Relationship":
            # Add relationship context (e.g., "more", "than")
            current_relationship["relation"] = current_relationship.get("relation", "") + " " + token
        elif tag == "Goal":
            goal_tokens.append(token)
        elif tag == "O":
            # Commit the current relationship if it's complete
            if current_relationship:
                structured_data["relationships"].append(current_relationship)
                current_relationship = {}

    # Add the goal if detected
    if goal_tokens:
        structured_data["goal"] = " ".join(goal_tokens)

    # Commit any leftover relationship
    if current_relationship:
        structured_data["relationships"].append(current_relationship)

    return structured_data


def preprocess_sentence(sentence, word2idx):
    tokens = sentence.split()
    indices = [word2idx.get(token, 0) for token in tokens]  # Handle OOV by mapping unknown words to index 0
    return torch.tensor(indices)

def predict(sentence, word2idx, tag2idx):
    token_indices = preprocess_sentence(sentence, word2idx)  # Get indices
    # Pad sequence to the expected batch shape for model
    token_indices = pad_sequence([token_indices], batch_first=True, padding_value=0)  # Correct dimension: [batch_size, seq_len]
    
    with torch.no_grad():
        entity_outputs, relationship_outputs = model(token_indices)
    entity_out = torch.argmax(entity_outputs, dim=-1).squeeze(0).tolist() 
    relationship_out = torch.argmax(relationship_outputs, dim=-1).squeeze(0).tolist()
    predictions = relationship_out
    idx2tag = {v: k for k, v in tag2idx.items()}
    return [(token, idx2tag[pred]) for token, pred in zip(sentence.split(), predictions)]

# Example usage
example_sentence = "Ravi has some coins. He has 2 more quarters than nickels and 4 more dimes than quarters. If he has 6 nickels, how much money does he have?"
tagged_output = predict(example_sentence, word2idx, tag2idx)
structured_data = convert_to_json(tagged_output)
print(json.dumps(structured_data, indent=2))
