import copy
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
    pos_weight: torch.Tensor = None,
) -> dict:
    """
    Train model locally for local_epochs. Returns loss history.
    pos_weight: per-class (neg/pos) ratio tensor to handle class imbalance.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )
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


def train_one_round_fedprox(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    local_epochs: int,
    global_model: nn.Module,
    mu: float = 0.01,
    pos_weight: torch.Tensor = None,
) -> dict:
    """
    FedProx local training: adds proximal term (mu/2)||w - w_global||^2 to loss.
    (Li et al., 2020 — https://arxiv.org/abs/1812.06127)
    """
    global_params = {name: param.detach().clone().to(device)
                     for name, param in global_model.named_parameters()}

    model.train()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )
    history = {"train_loss": []}

    for epoch in range(local_epochs):
        running_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"    Epoch {epoch+1}/{local_epochs}", leave=False)
        for images, labels in batch_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            prox = sum(
                ((p - global_params[n]) ** 2).sum()
                for n, p in model.named_parameters()
            )
            loss = loss + (mu / 2) * prox
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
