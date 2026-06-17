import yaml
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from src.data.alignment import align_labels_to_subwords


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Label mapping — fixed, we know the labels from the data
LABEL2ID = {"O": 0, "B-Entity": 1, "I-Entity": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_conll(path: str) -> list[dict]:
    """
    Parse CoNLL-format file into list of sentences.
    Each sentence: {"tokens": [...], "ner_tags": [...]}
    Format: token | POS | chunk | NER_label (tab-separated)
    Sentences separated by blank lines, docs by -DOCSTART- lines.
    """
    sentences = []
    tokens, tags = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Skip doc boundary markers
            if line.startswith("-DOCSTART-"):
                continue

            # Blank line = sentence boundary
            if line.strip() == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            token = parts[0]
            label = parts[3]

            tokens.append(token)
            tags.append(LABEL2ID.get(label, 0))  # default O if unknown

    # Catch final sentence if file doesn't end with blank line
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})

    return sentences


class BC5CDRDataset(Dataset):
    def __init__(self, split: str, config: dict, tokenizer: BertTokenizerFast):
        self.config = config
        self.tokenizer = tokenizer
        self.max_len = config["model"]["max_seq_len"]
        self.label_all_tokens = config["data"]["label_all_tokens"]
        self.label2id = LABEL2ID
        self.id2label = ID2LABEL

        split_map = {"train": "train.txt", "validation": "dev.txt", "test": "test.txt"}
        path = f"data/raw/{split_map[split]}"
        self.examples = parse_conll(path)

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

        labels = align_labels_to_subwords(
            encoding, ner_tags, self.max_len, self.label_all_tokens
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }
