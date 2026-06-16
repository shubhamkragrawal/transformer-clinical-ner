import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Shapes:
        query, key, value: (batch, heads, seq_len, d_k)
        mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)

    Returns:
        output: (batch, heads, seq_len, d_k)
        attn_weights: (batch, heads, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # QK^T: (batch, heads, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Positions where mask == 0 get -inf so softmax zeroes them out
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # Weighted sum of values: (batch, heads, seq_len, d_k)
    output = torch.matmul(attn_weights, value)

    return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Splits d_model into num_heads parallel attention heads,
    each operating on a smaller d_k = d_model / num_heads subspace.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Single combined projection matrices — more efficient than separate per-head ones
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2)  # (batch, seq_len, num_heads, d_k)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query, key, value: (batch, seq_len, d_model)
        mask: (batch, 1, 1, seq_len) — broadcasts across heads and query positions
        """
        # Project then split into heads
        Q = self._split_heads(self.W_q(query))  # (batch, heads, seq_len, d_k)
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        # Attention per head
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # Merge heads back, then final linear projection
        merged = self._merge_heads(attn_output)  # (batch, seq_len, d_model)
        output = self.W_o(merged)

        return output, attn_weights