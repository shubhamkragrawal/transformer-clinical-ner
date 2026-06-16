import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Maps token ids to dense vectors, scaled by sqrt(d_model)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).
    Not learned — injected once, added to token embeddings.
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model) — broadcast over batch
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """Combines token embedding + positional encoding."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        return x