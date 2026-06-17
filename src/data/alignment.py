"""
Subword-to-word label alignment for token classification.

When a word like "naloxone" is split into multiple subword tokens
(na, ##lo, ##xon, ##e) by the tokenizer, only the first subword keeps
the real label. Continuations and special tokens ([CLS], [SEP], [PAD])
get -100, PyTorch's CrossEntropyLoss ignore_index sentinel, so they
never contribute to loss or gradients.
"""
import torch


def align_labels_to_subwords(
    encoding,
    ner_tags: list[int],
    max_len: int,
    label_all_tokens: bool = False,
) -> torch.Tensor:
    """
    Align word-level NER tags to subword tokens produced by a fast tokenizer.

    Args:
        encoding: a tokenizer output with .word_ids(batch_index=0) available
                  (i.e. produced by a *Fast tokenizer with is_split_into_words=True)
        ner_tags: word-level label ids, one per original word
        max_len: sequence length to pad/truncate the label tensor to
        label_all_tokens: if True, every subword of a word gets the real
                           label; if False (default), only the first subword
                           does and the rest get -100

    Returns:
        A (max_len,) tensor of label ids, with -100 at ignored positions.
    """
    word_ids = encoding.word_ids(batch_index=0)
    labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special token ([CLS], [SEP], [PAD]) — ignore in loss
            labels.append(-100)
        elif word_id != prev_word_id:
            # First subword of this word — use the real label
            labels.append(ner_tags[word_id])
        else:
            # Subsequent subword of the same word
            labels.append(ner_tags[word_id] if label_all_tokens else -100)
        prev_word_id = word_id

    # Pad to max_len (truncation is already handled by the tokenizer itself)
    labels += [-100] * (max_len - len(labels))
    return torch.tensor(labels[:max_len], dtype=torch.long)