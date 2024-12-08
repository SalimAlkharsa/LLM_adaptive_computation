import json
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict


def tokenize_and_label(data):
    """
    Tokenize sentences and dynamically label tokens based on context.
    """
    tokenized_sentences = []
    labels = []

    for item in data:
        sentence = item["question"]
        entities = set(item["entities"])
        quantities = item["quantities"]  # This will contain numerical quantities or named quantities
        relationships = item["relationships"]  # Map expressions or math patterns here
        goal = item["goal"]

        # Tokenize the sentence by splitting words and punctuation
        tokens = re.findall(r'\w+|\d+|[^\w\s]', sentence)  # Split by words, numbers, and punctuation
        token_labels = []

        for token in tokens:
            if token in entities:
                token_labels.append("Entity")
            elif token.isdigit():  # Map raw numbers as quantities
                token_labels.append("Quantity")
            elif any(op in token for op in relationships.keys()):  # Detect math relationships
                token_labels.append("Relationship")
            elif goal.startswith(token):  # Match parts of the goal
                token_labels.append("Goal")
            else:
                token_labels.append("O")  # Default case

        tokenized_sentences.append(tokens)
        labels.append(token_labels)

    return tokenized_sentences, labels


class WordProblemDataset(Dataset):
    """
    PyTorch Dataset for word problems.
    """
    def __init__(self, sentences, labels, word2idx, tag2idx, max_len=None):
        self.sentences = [
            [word2idx[word] for word in sentence] for sentence in sentences
        ]
        self.labels = [
            [tag2idx[label] for label in label_list] for label_list in labels
        ]
        self.word2idx = word2idx
        self.tag2idx = tag2idx

        # Determine maximum sequence length
        self.max_len = max_len or max(len(sentence) for sentence in self.sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Pad sequences to ensure uniformity across batches
        sentence = self.sentences[idx]
        label = self.labels[idx]

        if len(sentence) < self.max_len:
            sentence += [0] * (self.max_len - len(sentence))
            label += [self.tag2idx["<PAD>"]] * (self.max_len - len(label))

        return torch.tensor(sentence), torch.tensor(label)
    
class WordProblemDataset(Dataset):
    def __init__(self, sentences, labels, word2idx, tag2idx, max_len=None):
        self.sentences = [[word2idx[word] for word in sentence] for sentence in sentences]
        self.labels = [[tag2idx[label] for label in labels] for labels in labels]
        self.max_len = max_len or max(len(sentence) for sentence in self.sentences)
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Padding
        if len(sentence) < self.max_len:
            sentence += [0] * (self.max_len - len(sentence))
            label += [self.tag2idx["<PAD>"]] * (self.max_len - len(label))

        return torch.tensor(sentence), torch.tensor(label)



def build_vocab(sentences, labels):
    """
    Dynamically build vocab and tag mappings.
    """
    word2idx = defaultdict(lambda: len(word2idx))
    tag2idx = defaultdict(lambda: len(tag2idx))
    word2idx["<PAD>"] = 0
    tag2idx["<PAD>"] = 0

    for sentence, label_list in zip(sentences, labels):
        for word, tag in zip(sentence, label_list):
            _ = word2idx[word]
            _ = tag2idx[tag]

    return word2idx, tag2idx