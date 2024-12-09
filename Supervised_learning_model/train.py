import torch
import json
from torch.utils.data import DataLoader
from dataset import tokenize_and_label, build_vocab, WordProblemDataset
from model import BiLSTMTagger
from sklearn.model_selection import train_test_split
import torch.nn as nn


# Training loop
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for sentences, labels in train_loader:
        sentences, labels = sentences.to(device), labels.to(device)
        optimizer.zero_grad()
        entities_outputs, relationship_output = model(sentences)
        entities_outputs = entities_outputs.view(-1, entities_outputs.shape[-1])
        relationship_output = relationship_output.view(-1, relationship_output.shape[-1])
        labels = labels.view(-1)
        loss = criterion(entities_outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # # Monitor how the model is predicting
        # predictions = torch.argmax(outputs, dim=-1)
        # print("Predictions: ", predictions[:5])  # Print first 5 predictions
        # print("Labels: ", labels[:5])  # Compare with ground truth labels
    return total_loss / len(train_loader)

# Evaluation loop
def evaluate_model(model, test_loader, tag2idx, device):
    model.eval()
    correct_entities, total_entities = 0, 0
    correct_relationships, total_relationships = 0, 0

    with torch.no_grad():
        for sentences, labels in test_loader:
            sentences, labels = sentences.to(device), labels.to(device)

            # Model outputs: two separate logits tensors
            entities_outputs, relationship_outputs = model(sentences)

            # Predictions for entities and relationships
            entity_predictions = torch.argmax(entities_outputs, dim=-1)  # [batch_size, seq_len]
            relationship_predictions = torch.argmax(relationship_outputs, dim=-1)  # [batch_size, seq_len]

            # Separate masks for each label type
            mask_entities = labels != tag2idx["<PAD>"]  # Adjust if entity and relationship masks differ
            mask_relationships = mask_entities  # Assuming same mask; change if needed.

            # Evaluate entities
            masked_entity_predictions = entity_predictions[mask_entities]
            masked_entity_labels = labels[mask_entities]
            correct_entities += (masked_entity_predictions == masked_entity_labels).sum().item()
            total_entities += mask_entities.sum().item()

            # Evaluate relationships
            masked_relationship_predictions = relationship_predictions[mask_relationships]
            masked_relationship_labels = labels[mask_relationships]
            correct_relationships += (masked_relationship_predictions == masked_relationship_labels).sum().item()
            total_relationships += mask_relationships.sum().item()

    # Calculate separate accuracies
    entity_accuracy = correct_entities / total_entities if total_entities > 0 else 0
    relationship_accuracy = correct_relationships / total_relationships if total_relationships > 0 else 0

    print(f"Entity Accuracy: {entity_accuracy:.2%}")
    print(f"Relationship Accuracy: {relationship_accuracy:.2%}")

    # Return combined accuracy if desired
    combined_accuracy = (correct_entities + correct_relationships) / (total_entities + total_relationships)
    return combined_accuracy

if __name__ == "__main__":
    # Load and preprocess dataset
    # Load and preprocess dataset
    with open("questions_dataset.json", "r") as file:
        data = json.load(file)

    tokenized_sentences, labels = tokenize_and_label(data)
    word2idx, tag2idx = build_vocab(tokenized_sentences, labels)

    

    # Save vocab mappings to files for later use
    with open("word2idx.json", "w") as f:
        json.dump(word2idx, f)

    with open("tag2idx.json", "w") as f:
        json.dump(tag2idx, f)


    # Split train-test
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        tokenized_sentences, labels, test_size=0.2
    )

    # Determine max_len
    max_len = max(
        max(len(sentence) for sentence in train_sentences),
        max(len(sentence) for sentence in test_sentences),
    )

    train_dataset = WordProblemDataset(
        train_sentences, train_labels, word2idx=word2idx, tag2idx=tag2idx, max_len=max_len
    )
    test_dataset = WordProblemDataset(
        test_sentences, test_labels, word2idx=word2idx, tag2idx=tag2idx, max_len=max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
    # Initialize model, optimizer, and criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMTagger(len(word2idx), len(tag2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])
    
    # Train and evaluate
    for epoch in range(100):  # Adjust number of epochs as needed
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")
    
    accuracy = evaluate_model(model, test_loader, tag2idx, device)
    print(f"Test Accuracy: {accuracy:.2%}")
    
    # Save model
    torch.save(model.state_dict(), "bilstm_model.pth")