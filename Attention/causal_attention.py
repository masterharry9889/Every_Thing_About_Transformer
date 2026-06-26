import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads

        # Linear layers for query, key, and value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Register a buffer for the causal mask(lower triangular matrix)
        # shape: (1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Project inputs to query, key, and value
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)    # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # Compute the output
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads and project to output dimension
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
        output = self.out(output)  # (batch_size, seq_len, embed_dim)

        return output
    

"""The Causal Attention mechanism is a variant of multi-head attention that restricts a model 
from attending to future tokens, ensuring strict autoregressive generation.  This is implemented by applying a
 lower-triangular mask to the attention scores before the softmax function, effectively setting future positions to negative 
 infinity."""