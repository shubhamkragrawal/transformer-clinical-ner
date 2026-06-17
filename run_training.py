"""
Full training + entity-level F1 evaluation for the from-scratch
TransformerNER model on BC5CDR.

Usage:
    python3 run_training.py
"""
import time
import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from src.data.dataset import BC5CDRDataset, load_config, ID2LABEL
from src.model.classifier import TransformerNER
from src.train_utils import get_optimizer, get_scheduler
from src.evaluate_metrics import evaluate_f1
from train import train_one_epoch, evaluate, set_seed, save_checkpoint


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    config = load_config()
    set_seed(config["training"]["seed"])

    device = get_device()
    print(f"Device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_ds = BC5CDRDataset("train", config, tokenizer)
    val_ds = BC5CDRDataset("validation", config, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False
    )

    num_labels = len(ID2LABEL)
    model = TransformerNER.from_config(config, num_labels).to(device)

    optimizer = get_optimizer(model, config)
    num_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = get_scheduler(optimizer, config, num_steps)

    best_val_loss = float("inf")
    best_f1 = 0.0
    start = time.time()

    for epoch in range(config["training"]["epochs"]):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, num_labels,
            config["training"]["grad_clip"], device,
        )
        val_loss = evaluate(model, val_loader, num_labels, device)

        # F1 is the metric that actually matters for NER; val_loss and F1
        # don't always agree on which epoch is "best" (loss rewards
        # calibrated probabilities, F1 only cares about hard argmax
        # decisions at entity boundaries) — so we track both and
        # checkpoint on F1, the deployment-relevant metric.
        f1_results = evaluate_f1(model, val_loader, device)
        val_f1 = f1_results["f1"]

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch+1}/{config['training']['epochs']} "
            f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| val_f1={val_f1:.4f} | time={epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
            model, "checkpoints/best_model.pt", epoch,
            config=config,
            metrics={
                    "val_loss"  : float(val_loss),
                    "val_f1"    : float(val_f1),
                    "precision" : float(f1_results["precision"]),
                    "recall"    : float(f1_results["recall"]),
                },
    )
    print(f"  -> new best model saved (val_f1={val_f1:.4f})")

    total_time = time.time() - start
    print(f"\nTotal training time: {total_time/60:.1f} min")
    print(f"Best val_loss seen: {best_val_loss:.4f}")
    print(f"Best val_f1 seen:   {best_f1:.4f}")

    print("\nRe-loading best checkpoint (by F1) for final report...")
    checkpoint = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")

    results = evaluate_f1(model, val_loader, device)
    print("\nFrom-scratch model results (best checkpoint):")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")
    print(f"\n{results['report']}")


if __name__ == "__main__":
    main()