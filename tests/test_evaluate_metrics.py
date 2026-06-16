import torch
import pytest
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from src.data.dataset import BC5CDRDataset, load_config, ID2LABEL
from src.model.classifier import TransformerNER
from src.evaluate_metrics import decode_predictions, evaluate_f1


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def dataset(config, tokenizer):
    return BC5CDRDataset("validation", config, tokenizer)


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    return TransformerNER.from_config(config, num_labels=len(ID2LABEL))


def test_decode_predictions_excludes_ignored_positions():
    """Positions with label -100 should not appear in decoded output at all."""
    # 1 sentence, 5 positions: [CLS]=-100, O, B-Entity, -100(subword), [SEP]=-100
    labels = torch.tensor([[-100, 0, 1, -100, -100]])
    logits = torch.zeros(1, 5, 3)
    logits[0, 1, 0] = 10.0  # force prediction "O" at position 1
    logits[0, 2, 1] = 10.0  # force prediction "B-Entity" at position 2

    true_seq, pred_seq = decode_predictions(logits, labels)

    assert len(true_seq) == 1
    assert len(true_seq[0]) == 2  # only 2 valid positions survive
    assert true_seq[0] == ["O", "B-Entity"]
    assert pred_seq[0] == ["O", "B-Entity"]


def test_decode_predictions_skips_fully_masked_sentences():
    """A sentence where every position is -100 should be dropped entirely."""
    labels = torch.tensor([[-100, -100, -100]])
    logits = torch.zeros(1, 3, 3)

    true_seq, pred_seq = decode_predictions(logits, labels)

    assert len(true_seq) == 0
    assert len(pred_seq) == 0


def test_decode_predictions_handles_batch():
    """Multiple sentences in a batch should each produce independent sequences."""
    labels = torch.tensor([
        [-100, 0, 0, -100],
        [-100, 1, 2, -100],
    ])
    logits = torch.zeros(2, 4, 3)
    logits[:, :, 0] = 10.0  # default everything to predict "O"

    true_seq, pred_seq = decode_predictions(logits, labels)

    assert len(true_seq) == 2
    assert true_seq[0] == ["O", "O"]
    assert true_seq[1] == ["B-Entity", "I-Entity"]


def test_evaluate_f1_returns_expected_keys(model, dataset, config):
    loader = DataLoader(dataset, batch_size=4)
    # Use only a few batches for test speed
    from itertools import islice
    small_loader = list(islice(loader, 3))

    device = torch.device("cpu")
    results = evaluate_f1(model, small_loader, device)

    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "report" in results


def test_evaluate_f1_scores_in_valid_range(model, dataset, config):
    loader = DataLoader(dataset, batch_size=4)
    from itertools import islice
    small_loader = list(islice(loader, 3))

    device = torch.device("cpu")
    results = evaluate_f1(model, small_loader, device)

    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["f1"] <= 1.0


def test_perfect_predictions_yield_f1_of_one():
    """Sanity check: if predictions exactly match labels, F1 must be 1.0."""
    labels = torch.tensor([[-100, 1, 2, 0, -100]])
    logits = torch.zeros(1, 5, 3)
    logits[0, 1, 1] = 10.0  # B-Entity
    logits[0, 2, 2] = 10.0  # I-Entity
    logits[0, 3, 0] = 10.0  # O

    true_seq, pred_seq = decode_predictions(logits, labels)

    from seqeval.metrics import f1_score
    score = f1_score(true_seq, pred_seq)
    assert score == 1.0