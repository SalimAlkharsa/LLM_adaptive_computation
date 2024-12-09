import torch
import torch.nn as nn

# class BiLSTMTagger(nn.Module):
#     def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
#         super(BiLSTMTagger, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
#         self.fc = nn.Linear(hidden_dim * 2, tagset_size)
    
#     def forward(self, x):
#         x = self.embedding(x)
#         lstm_out, _ = self.lstm(x)
#         logits = self.fc(lstm_out)
#         return logits




class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout_prob=0.3):
        """
        BiLSTM Model with Multi-layer LSTM and Dropout
        """
        super(BiLSTMTagger, self).__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Multi-layer BiLSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Fully connected layer to map LSTM outputs to tag predictions
        self.fc_entities = nn.Linear(hidden_dim * 2, tagset_size)
        self.fc_relationships = nn.Linear(hidden_dim * 2, tagset_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Embed input tokens
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # Pass through BiLSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim * 2]

        # Apply dropout for regularization
        lstm_out = self.dropout(lstm_out)

        entities_logits = self.fc_entities(lstm_out)  # Entity predictions
        relationships_logits = self.fc_relationships(lstm_out)  # Relationship predictions

        # return
        return entities_logits, relationships_logits