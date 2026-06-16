import torch
import torch.nn as nn

from src.model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network applied independently to each
    sequence position: Linear -> ReLU -> Linear.
    Expands to ff_dim then projects back to d_model.
    """

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer:
      1. Multi-head self-attention + residual + LayerNorm
      2. Feed-forward + residual + LayerNorm

    Uses pre-LayerNorm (norm before sublayer, not after) — more stable
    training for deeper stacks, which is why most modern implementations
    (GPT-2 onward) use it instead of the original post-LN design.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # --- Self-attention sublayer (pre-LN) ---
        normed = self.norm1(x)
        attn_output, _ = self.self_attn(normed, normed, normed, mask=mask)
        x = x + self.dropout1(attn_output)  # residual connection

        # --- Feed-forward sublayer (pre-LN) ---
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout2(ff_output)  # residual connection

        return x