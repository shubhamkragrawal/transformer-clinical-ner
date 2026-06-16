import os
import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from src.data.dataset import BC5CDRDataset, load_config, ID2LABEL
from src.model.classifier import TransformerNER
from src.train_utils import compute_loss, get_optimizer, get_scheduler, clip_gradients


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    num_labels: int,
    grad_clip: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, labels, num_labels)

        loss.backward()
        clip_gradients(model, grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    num_labels: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, labels, num_labels)
        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(model: torch.nn.Module, path: str, epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, path)


def main():
    config = load_config()
    set_seed(config["training"]["seed"])

    device = torch.device(
    "mps" if torch.backends.mps.is_available() # working on macbook
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
    print(f"Using device: {device}")

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
    num_training_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = get_scheduler(optimizer, config, num_training_steps)

    best_val_loss = float("inf")
    checkpoint_dir = config["paths"]["checkpoint_dir"]

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            num_labels, config["training"]["grad_clip"], device,
        )
        val_loss = evaluate(model, val_loader, num_labels, device)

        print(f"Epoch {epoch+1}/{config['training']['epochs']} "
              f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, os.path.join(checkpoint_dir, "best_model.pt"), epoch)
            print(f"  ↳ New best model saved (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()