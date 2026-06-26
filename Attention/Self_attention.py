import torch
import torch.nn as nn
import torch.nn.functional as F

class self_attention(nn.Module):
    def __init__(self, input_dim):
        super(self_attention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.input_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights



"""Self-attention:
Let's each token attend directly to other tokens in same sequence: the model compares every token with
the rest of the sequence and build a context-aware representation. This is what allows transformers to
capture relationships accross distanceand make it possible to modellong-range dependencies without recurrence."""
