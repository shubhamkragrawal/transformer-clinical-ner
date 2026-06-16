import torch
import torch.nn as nn

from src.model.embeddings import TransformerEmbedding
from src.model.encoder import TransformerEncoder, create_padding_mask


class TransformerNER(nn.Module):
    """
    End-to-end transformer encoder for token classification (NER).

    Pipeline: input_ids -> embeddings -> encoder stack -> linear head -> logits
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len, dropout)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, ff_dim, dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            logits: (batch, seq_len, num_labels)
        """
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)

        mask = create_padding_mask(attention_mask) if attention_mask is not None else None
        x = self.encoder(x, mask=mask)  # (batch, seq_len, d_model)

        logits = self.classifier(x)  # (batch, seq_len, num_labels)
        return logits

    @classmethod
    def from_config(cls, config: dict, num_labels: int) -> "TransformerNER":
        """Build model directly from the loaded config.yaml dict."""
        m = config["model"]
        return cls(
            vocab_size=m["vocab_size"],
            num_labels=num_labels,
            d_model=m["hidden_dim"],
            num_heads=m["num_heads"],
            num_layers=m["num_layers"],
            ff_dim=m["ff_dim"],
            max_seq_len=m["max_seq_len"],
            dropout=m["dropout"],
        )