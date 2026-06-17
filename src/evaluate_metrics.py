import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from src.data.dataset import ID2LABEL


def decode_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Converts model logits + label tensor into seqeval's expected format:
    list of sentences, each a list of string labels (e.g. "B-Entity").

    Positions where labels == -100 are excluded entirely — these are
    padding, [CLS]/[SEP], and non-first-subword positions that were
    never meant to contribute to the metric, same as they don't
    contribute to loss.

    logits: (batch, seq_len, num_labels)
    labels: (batch, seq_len)
    """
    preds = torch.argmax(logits, dim=-1)  # (batch, seq_len)

    true_labels = []
    pred_labels = []

    for i in range(labels.size(0)):
        true_seq = []
        pred_seq = []
        for j in range(labels.size(1)):
            label_id = labels[i, j].item()
            if label_id == -100:
                continue  # skip ignored positions
            true_seq.append(ID2LABEL[label_id])
            pred_seq.append(ID2LABEL[preds[i, j].item()])

        if true_seq:  # skip sentences with zero valid positions
            true_labels.append(true_seq)
            pred_labels.append(pred_seq)

    return true_labels, pred_labels


@torch.no_grad()
def evaluate_f1(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Runs the model over a full dataset and computes entity-level
    precision, recall, and F1 using seqeval's BIO-aware scoring.
    """
    model.eval()

    all_true = []
    all_pred = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        true_seq, pred_seq = decode_predictions(logits.cpu(), labels.cpu())

        all_true.extend(true_seq)
        all_pred.extend(pred_seq)

    return {
        "precision": precision_score(all_true, all_pred),
        "recall": recall_score(all_true, all_pred),
        "f1": f1_score(all_true, all_pred),
        "report": classification_report(all_true, all_pred, digits=4),
    }


def run_bert_baseline(config: dict, device: torch.device) -> dict:
    from transformers import BertForTokenClassification, BertTokenizerFast
    from torch.utils.data import DataLoader
    from src.data.dataset import BC5CDRDataset
    from src.train_utils import get_optimizer, get_scheduler
    import time

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_ds = BC5CDRDataset("train", config, tokenizer)
    val_ds = BC5CDRDataset("validation", config, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(ID2LABEL)
    ).to(device)

    optimizer = get_optimizer(model, config)
    num_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = get_scheduler(optimizer, config, num_steps)

    model.train()
    for epoch in range(config["training"]["epochs"]):
        epoch_start = time.time()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()
            scheduler.step()

            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        remaining = (config["training"]["epochs"] - epoch - 1) * epoch_time

        print(
            f"  [BERT] Epoch {epoch+1}/{config['training']['epochs']} "
            f"| train_loss={avg_loss:.4f} | time={epoch_time:.1f}s "
            f"| est. remaining={remaining/60:.1f} min"
        )

    # Evaluate using the same decode_predictions logic
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask=attention_mask).logits
            true_seq, pred_seq = decode_predictions(logits.cpu(), labels.cpu())
            all_true.extend(true_seq)
            all_pred.extend(pred_seq)

    return {
        "precision": precision_score(all_true, all_pred),
        "recall": recall_score(all_true, all_pred),
        "f1": f1_score(all_true, all_pred),
    }