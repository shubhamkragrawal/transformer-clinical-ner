import torch
import pytest
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from src.data.dataset import BC5CDRDataset, load_config, ID2LABEL
from src.model.classifier import TransformerNER
from src.train_utils import compute_loss, get_optimizer, get_scheduler, clip_gradients
import warnings
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`")

@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def dataset(config, tokenizer):
    return BC5CDRDataset("train", config, tokenizer)


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    return TransformerNER.from_config(config, num_labels=len(ID2LABEL))


@pytest.fixture
def batch(dataset):
    loader = DataLoader(dataset, batch_size=4)
    return next(iter(loader))


def test_compute_loss_is_positive_scalar(model, batch):
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = compute_loss(logits, batch["labels"], num_labels=3)

    assert loss.dim() == 0  # scalar
    assert loss.item() > 0


def test_compute_loss_ignores_masked_positions(model, batch):
    """
    When all labels are -100, CrossEntropyLoss with mean reduction
    returns nan (division by zero valid elements) — this is expected
    PyTorch behavior, not a bug. Confirms ignore_index logic is engaged.
    """
    logits = model(batch["input_ids"], batch["attention_mask"])
    fully_masked_labels = torch.full_like(batch["labels"], -100)

    loss = compute_loss(logits, fully_masked_labels, num_labels=3)
    assert torch.isnan(loss)


def test_optimizer_has_correct_lr_and_weight_decay(model, config):
    optimizer = get_optimizer(model, config)
    param_group = optimizer.param_groups[0]

    assert param_group["lr"] == config["training"]["learning_rate"]
    assert param_group["weight_decay"] == config["training"]["weight_decay"]


def test_scheduler_warmup_increases_lr(model, config):
    optimizer = get_optimizer(model, config)
    num_steps = 1000
    scheduler = get_scheduler(optimizer, config, num_steps)

    warmup_steps = config["training"]["warmup_steps"]
    lrs = []
    for _ in range(warmup_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    # LR should be non-decreasing throughout warmup
    assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))


def test_scheduler_decays_after_warmup(model, config):
    optimizer = get_optimizer(model, config)
    num_steps = 1000
    scheduler = get_scheduler(optimizer, config, num_steps)

    warmup_steps = config["training"]["warmup_steps"]

    # Step through warmup
    for _ in range(warmup_steps):
        scheduler.step()
    lr_after_warmup = scheduler.get_last_lr()[0]

    # Step further into decay phase
    for _ in range(200):
        scheduler.step()
    lr_after_decay = scheduler.get_last_lr()[0]

    assert lr_after_decay < lr_after_warmup


def test_scheduler_reaches_near_zero_at_end(model, config):
    optimizer = get_optimizer(model, config)
    num_steps = 1000
    scheduler = get_scheduler(optimizer, config, num_steps)

    for _ in range(num_steps):
        scheduler.step()

    assert scheduler.get_last_lr()[0] <= 1e-6


def test_clip_gradients_reduces_large_gradients(model, batch, config):
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = compute_loss(logits, batch["labels"], num_labels=3)
    loss.backward()

    # Artificially inflate gradients to guarantee clipping triggers
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= 1000

    pre_clip_norm = clip_gradients(model, max_norm=config["training"]["grad_clip"])

    # Recompute actual norm after clipping
    post_clip_norm = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
    ).item()

    assert pre_clip_norm > config["training"]["grad_clip"]
    assert post_clip_norm <= config["training"]["grad_clip"] + 1e-4


def test_clip_gradients_returns_float(model, batch, config):
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = compute_loss(logits, batch["labels"], num_labels=3)
    loss.backward()

    norm = clip_gradients(model, max_norm=config["training"]["grad_clip"])
    assert isinstance(norm, float)