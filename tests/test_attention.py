import torch
import pytest
from src.model.attention import scaled_dot_product_attention


@pytest.fixture
def qkv():
    torch.manual_seed(42)
    batch, heads, seq_len, d_k = 2, 1, 5, 8
    q = torch.randn(batch, heads, seq_len, d_k)
    k = torch.randn(batch, heads, seq_len, d_k)
    v = torch.randn(batch, heads, seq_len, d_k)
    return q, k, v


def test_output_shapes(qkv):
    q, k, v = qkv
    output, attn_weights = scaled_dot_product_attention(q, k, v)
    assert output.shape == q.shape
    assert attn_weights.shape == (q.size(0), q.size(1), q.size(2), k.size(2))


def test_attention_weights_sum_to_one(qkv):
    q, k, v = qkv
    _, attn_weights = scaled_dot_product_attention(q, k, v)
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_padding_mask_zeroes_masked_positions(qkv):
    q, k, v = qkv
    batch, heads, seq_len, _ = q.shape

    mask = torch.ones(batch, 1, 1, seq_len)
    mask[:, :, :, 3:] = 0  # mask out last 2 positions

    _, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

    # Masked positions get exactly zero weight
    assert torch.allclose(
        attn_weights[:, :, :, 3:], torch.zeros_like(attn_weights[:, :, :, 3:])
    )

    # Remaining weights still sum to 1
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_dropout_applied_in_training_mode(qkv):
    q, k, v = qkv
    dropout = torch.nn.Dropout(p=0.5)
    dropout.train()  # ensure dropout is active

    torch.manual_seed(0)
    _, attn_weights = scaled_dot_product_attention(q, k, v, dropout=dropout)

    # With p=0.5 dropout active, some weights should be zeroed out
    assert (attn_weights == 0).any()