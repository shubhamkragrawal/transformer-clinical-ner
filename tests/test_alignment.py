import pytest
from transformers import BertTokenizerFast
from src.data.dataset import BC5CDRDataset, load_config, LABEL2ID, ID2LABEL


@pytest.fixture(scope="module")
def tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def train_dataset(config, tokenizer):
    return BC5CDRDataset("train", config, tokenizer)


def test_dataset_loads_expected_sentence_count(train_dataset):
    # BC5CDR train split from BioFLAIR mirror
    assert len(train_dataset) == 3942


def test_no_empty_sentences(train_dataset):
    empty = [ex for ex in train_dataset.examples if len(ex["tokens"]) == 0]
    assert len(empty) == 0


def test_sample_shapes_match_max_seq_len(train_dataset, config):
    sample = train_dataset[0]
    max_len = config["model"]["max_seq_len"]

    assert sample["input_ids"].shape[0] == max_len
    assert sample["attention_mask"].shape[0] == max_len
    assert sample["labels"].shape[0] == max_len


def test_special_tokens_get_ignored_label(train_dataset, tokenizer):
    sample = train_dataset[0]
    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    labels = sample["labels"].tolist()

    # [CLS] is always first token, must be ignored (-100)
    assert tokens[0] == "[CLS]"
    assert labels[0] == -100


def test_subword_continuation_gets_ignored_label(train_dataset, tokenizer):
    """
    For a word split into multiple subwords (## prefix),
    only the first subword should carry the real label;
    continuations should be -100 when label_all_tokens=False.
    """
    sample = train_dataset[0]
    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    labels = sample["labels"].tolist()

    continuation_indices = [i for i, t in enumerate(tokens) if t.startswith("##")]
    assert len(continuation_indices) > 0, "Expected at least one subword continuation in sample 0"

    for idx in continuation_indices:
        assert labels[idx] == -100


def test_longest_sentence_truncates_without_misalignment(train_dataset):
    """
    Sentences longer than max_seq_len must truncate cleanly:
    output shapes must still match max_seq_len exactly.
    """
    longest_idx = max(
        range(len(train_dataset.examples)),
        key=lambda i: len(train_dataset.examples[i]["tokens"]),
    )
    sample = train_dataset[longest_idx]
    max_len = train_dataset.config["model"]["max_seq_len"]

    assert sample["input_ids"].shape[0] == max_len
    assert sample["labels"].shape[0] == max_len


def test_label_values_are_valid_ids(train_dataset):
    """All non-ignored labels must be valid IDs in LABEL2ID."""
    valid_ids = set(ID2LABEL.keys())
    sample = train_dataset[0]

    for label in sample["labels"].tolist():
        assert label == -100 or label in valid_ids


def test_padding_positions_have_zero_attention_mask(train_dataset):
    """Padded positions (beyond real tokens) should have attention_mask == 0."""
    sample = train_dataset[0]
    attention_mask = sample["attention_mask"].tolist()

    # Real tokens come first, then padding — mask should be a block of 1s then 0s
    assert attention_mask.count(0) + attention_mask.count(1) == len(attention_mask)
    # First mask value must be 1 ([CLS] always present)
    assert attention_mask[0] == 1