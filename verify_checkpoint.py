"""
Load a saved checkpoint and re-run evaluation to confirm it reproduces
the metrics it claims. This is the reproducibility check anyone cloning
the repo should be able to run.

Usage:
    python3 verify_checkpoint.py checkpoints/best_model.pt
"""
import sys
import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from src.data.dataset import BC5CDRDataset, ID2LABEL
from src.model.classifier import TransformerNER
from src.evaluate_metrics import evaluate_f1
from run_training import get_device


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 verify_checkpoint.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    device = get_device()
    print(f"Device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    print(f"Checkpoint from epoch: {checkpoint['epoch'] + 1}")
    print(f"Claimed metrics: {checkpoint['metrics']}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    val_ds = BC5CDRDataset("validation", config, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    num_labels = len(ID2LABEL)
    model = TransformerNER.from_config(config, num_labels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nRe-running evaluation on validation set...")
    results = evaluate_f1(model, val_loader, device)

    print("\nReproduced metrics:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")

    claimed_f1 = checkpoint["metrics"].get("val_f1")
    if claimed_f1 is not None:
        diff = abs(results["f1"] - claimed_f1)
        status = "MATCH" if diff < 1e-3 else "MISMATCH"
        print(f"\nClaimed F1: {claimed_f1:.4f} | Reproduced F1: {results['f1']:.4f} | {status} (diff={diff:.4f})")


if __name__ == "__main__":
    main()