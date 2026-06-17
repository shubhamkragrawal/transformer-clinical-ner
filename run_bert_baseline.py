"""
BERT baseline for comparison against the from-scratch TransformerNER model.

Usage:
    python3 run_bert_baseline.py
"""
import time
import torch

from src.data.dataset import load_config
from src.evaluate_metrics import run_bert_baseline
from run_training import get_device


def main():
    config = load_config()

    # BERT is pretrained, so it converges much faster than a from-scratch
    # model — 20 epochs (tuned for the from-scratch model) is unnecessary
    # and wastes time. 5 epochs is standard practice for BERT fine-tuning.
    # We override here rather than editing config.yaml, since that file
    # also controls the from-scratch model's real training run.
    config["training"]["epochs"] = 5

    device = get_device()
    print(f"Device: {device}")

    print(f"Fine-tuning BERT baseline (bert-base-uncased) for {config['training']['epochs']} epochs...")
    start = time.time()

    results = run_bert_baseline(config, device)

    elapsed = time.time() - start
    print(f"\nBERT baseline training time: {elapsed/60:.1f} min")

    print("\nBERT baseline results:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")


if __name__ == "__main__":
    main()