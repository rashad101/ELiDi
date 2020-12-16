import torch
from torch import nn


class EntityDetection(nn.Module):
    def __init__(self, num_words, emb_dim, hidden_size, num_layers, dropout=0.3, batch_f=False):
        super(EntityDetection, self).__init__()
        self.batch_first = batch_f
        self.hidden_size = hidden_size
        self.num_words = num_words
        self.emb_dim = emb_dim
        self.num_layer = num_layers
        self.dropout = dropout
        target_size = 1

        self.lstm = nn.LSTM(input_size=self.emb_dim,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layer,
                               dropout=dropout,
                               bidirectional=True)
        self.dropout = nn.Dropout(p=self.dropout)
        self.relu = nn.Tanh()
        self.hidden2tag = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.BatchNorm1d(self.hidden_size * 2),
            self.relu,
            self.dropout,
            nn.Linear(self.hidden_size * 2, target_size)
        )

    def forward(self, x, m):
        outputs, (ht, ct) = self.lstm(self.dropout(x))
        outputs = outputs * m.unsqueeze(-1)
        scores = self.hidden2tag(outputs.view(-1, outputs.size(2))) * m.contiguous().view(-1, 1)
        return scores