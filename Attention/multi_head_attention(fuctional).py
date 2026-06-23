import torch
import torch.nn as nn

# Initialize with embedding dimension (e.g., 512) and number of heads (e.g., 8)
embed_dim = 512
num_heads = 8
mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

# Input tensors: (batch_size, sequence_length, embed_dim)
batch_size = 2
seq_len = 10
query = torch.randn(batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)

# Compute attention
attn_output, attn_output_weights = mha(query, key, value)
print(attn_output.shape)  # torch.Size([2, 10, 512])   
print(attn_output_weights.shape)  # torch.Size([2, 10, 10])
print(attn_output_weights)  # Attention weights for each head
print(attn_output)  # Output of the attention mechanism