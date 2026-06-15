import yaml
from datasets import load_dataset
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import torch


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_bc5cdr(split: str = "train"):
    """Load BC5CDR from HuggingFace datasets."""
    dataset = load_dataset("ncbi/ncbi_disease", split=split)
    return dataset


def get_label_mapping(dataset) -> tuple[dict, dict]:
    """Extract label2id and id2label from dataset features."""
    label_names = dataset.features["ner_tags"].feature.names
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


class BC5CDRDataset(Dataset):
    def __init__(self, split: str, config: dict, tokenizer: BertTokenizerFast):
        self.config = config
        self.tokenizer = tokenizer
        self.max_len = config["model"]["max_seq_len"]
        self.label_all_tokens = config["data"]["label_all_tokens"]

        raw = load_bc5cdr(split)
        self.label2id, self.id2label = get_label_mapping(raw)
        self.examples = list(raw)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        tokens = example["tokens"]
        ner_tags = example["ner_tags"] 

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self._align_labels(encoding, ner_tags)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def _align_labels(self, encoding, ner_tags: list[int]) -> torch.Tensor:
        """
        Align word-level NER tags to subword tokens.
        - First subword of a word: gets the original label
        - Subsequent subwords: get -100 (ignored in loss) if label_all_tokens=False
        - Special tokens ([CLS], [SEP], [PAD]): get -100
        """
        word_ids = encoding.word_ids(batch_index=0)
        labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special token — ignore in loss
                labels.append(-100)
            elif word_id != prev_word_id:
                # First subword of this word — use real label
                labels.append(ner_tags[word_id])
            else:
                # Subsequent subword — ignore or copy label
                labels.append(ner_tags[word_id] if self.label_all_tokens else -100)
            prev_word_id = word_id

        # Pad to max_len
        labels += [-100] * (self.config["model"]["max_seq_len"] - len(labels))
        return torch.tensor(labels, dtype=torch.long)