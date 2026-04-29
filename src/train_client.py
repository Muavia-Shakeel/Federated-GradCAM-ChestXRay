import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_round(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    local_epochs: int,
) -> dict:
    """
    Train model locally for local_epochs. Returns loss history.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    history = {"train_loss": []}

    for epoch in range(local_epochs):
        running_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"    Epoch {epoch+1}/{local_epochs}", leave=False)
        for images, labels in batch_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            batch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)

    return history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)

    return {
        "val_loss": avg_loss,
        "logits": all_logits,
        "labels": all_labels,
    }
