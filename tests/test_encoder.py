import torch
import pytest
from src.model.encoder import FeedForward, EncoderLayer


@pytest.fixture
def dims():
    return {"batch": 2, "seq_len": 10, "d_model": 256, "num_heads": 8, "ff_dim": 512}


def test_feedforward_preserves_shape(dims):
    torch.manual_seed(42)
    ff = FeedForward(dims["d_model"], dims["ff_dim"])
    x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
    out = ff(x)
    assert out.shape == x.shape


def test_encoder_layer_preserves_shape(dims):
    torch.manual_seed(42)
    layer = EncoderLayer(dims["d_model"], dims["num_heads"], dims["ff_dim"])
    x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
    out = layer(x)
    assert out.shape == x.shape


def test_encoder_layer_with_padding_mask(dims):
    torch.manual_seed(42)
    layer = EncoderLayer(dims["d_model"], dims["num_heads"], dims["ff_dim"])
    x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])

    mask = torch.ones(dims["batch"], 1, 1, dims["seq_len"])
    mask[:, :, :, 7:] = 0

    out = layer(x, mask=mask)
    assert out.shape == x.shape


def test_encoder_layer_transforms_input(dims):
    """Residual + sublayers should meaningfully change the representation."""
    torch.manual_seed(42)
    layer = EncoderLayer(dims["d_model"], dims["num_heads"], dims["ff_dim"])
    x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
    out = layer(x)
    assert not torch.allclose(out, x)


def test_encoder_layer_gradient_flows(dims):
    """Confirm gradients reach the input — validates residual path is intact."""
    torch.manual_seed(42)
    layer = EncoderLayer(dims["d_model"], dims["num_heads"], dims["ff_dim"])
    x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"], requires_grad=True)

    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))