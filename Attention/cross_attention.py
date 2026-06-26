import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.aelf_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, target, source, tgt_mask=None, src_mask=None):
        # Self-attention on the target sequence
        attn_output, _ = self.self_attn(target, target, target, attn_mask=tgt_mask)
        target = self.norm1(target + attn_output)

        # Cross-attention with the souce sequence
        cross_output, _ = self.cross_attn(target, source, source, attn_mask=src_mask)
        output = self.norm2(target + cross_output)

        return output
    


"""Cross-attention

Allows tokens from one sequence attend to a different sequence, bringing information from different sources or
 modalities together (for example, description and image). It uses queries from one module – usually encoder, 
 and keys and values from the other one (decoder)."""