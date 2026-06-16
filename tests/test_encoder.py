import torch
import pytest
from src.model.encoder import (
    FeedForward,
    EncoderLayer,
    TransformerEncoder,
    create_padding_mask)



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



def test_transformer_encoder_preserves_shape(dims):
    torch.manual_seed(42)
    encoder = TransformerEncoder(
        num_layers=4,
        d_model=dims["d_model"],
        num_heads=dims["num_heads"],
        ff_dim=dims["ff_dim"],
    )
    x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
    out = encoder(x)
    assert out.shape == x.shape


def test_create_padding_mask_shape(dims):
    attention_mask = torch.ones(dims["batch"], dims["seq_len"])
    mask = create_padding_mask(attention_mask)
    assert mask.shape == (dims["batch"], 1, 1, dims["seq_len"])


def test_encoder_with_padding_mask(dims):
    torch.manual_seed(42)
    encoder = TransformerEncoder(
        num_layers=4,
        d_model=dims["d_model"],
        num_heads=dims["num_heads"],
        ff_dim=dims["ff_dim"],
    )
    x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])

    attention_mask = torch.ones(dims["batch"], dims["seq_len"])
    attention_mask[:, 7:] = 0
    mask = create_padding_mask(attention_mask)

    out = encoder(x, mask=mask)
    assert out.shape == x.shape


def test_deeper_encoder_produces_different_output(dims):
    """Sanity check that stacking layers actually does something."""
    torch.manual_seed(42)
    x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])

    torch.manual_seed(123)
    shallow = TransformerEncoder(1, dims["d_model"], dims["num_heads"], dims["ff_dim"])
    torch.manual_seed(123)
    deep = TransformerEncoder(4, dims["d_model"], dims["num_heads"], dims["ff_dim"])

    out_shallow = shallow(x)
    out_deep = deep(x)

    assert not torch.allclose(out_shallow, out_deep)