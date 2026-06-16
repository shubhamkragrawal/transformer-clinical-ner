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