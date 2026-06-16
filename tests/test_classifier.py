import torch
import pytest
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from src.data.dataset import BC5CDRDataset, load_config, ID2LABEL
from src.model.classifier import TransformerNER


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def dataset(config, tokenizer):
    return BC5CDRDataset("train", config, tokenizer)


@pytest.fixture(scope="module")
def model(config):
    torch.manual_seed(42)
    return TransformerNER.from_config(config, num_labels=len(ID2LABEL))


def test_single_example_forward_shape(model, dataset, config):
    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0)
    attention_mask = sample["attention_mask"].unsqueeze(0)

    logits = model(input_ids, attention_mask)

    max_len = config["model"]["max_seq_len"]
    num_labels = len(ID2LABEL)
    assert logits.shape == (1, max_len, num_labels)


def test_batch_forward_shape(model, dataset, config):
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))

    logits = model(batch["input_ids"], batch["attention_mask"])

    max_len = config["model"]["max_seq_len"]
    num_labels = len(ID2LABEL)
    assert logits.shape == (4, max_len, num_labels)


def test_forward_without_attention_mask(model, dataset, config):
    """Model should still work if no mask is passed (treats all tokens as real)."""
    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0)

    logits = model(input_ids)

    max_len = config["model"]["max_seq_len"]
    num_labels = len(ID2LABEL)
    assert logits.shape == (1, max_len, num_labels)


def test_gradients_flow_to_all_parameters(model, dataset):
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))

    model.zero_grad()
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient reached {name}"


def test_from_config_matches_yaml_settings(config):
    model = TransformerNER.from_config(config, num_labels=3)
    m = config["model"]

    assert model.embedding.token_embedding.embedding.embedding_dim == m["hidden_dim"]
    assert model.classifier.out_features == 3
    assert len(model.encoder.layers) == m["num_layers"]