"""
Shared PyTorch training loop for every neural-net model across all four
asset classes.

Per explicit instruction this repo standardizes every DL model on:
  - up to 100 epochs
  - early stopping, patience 20-25 (default 22), tracked on validation loss
  - best-epoch weights checkpointed and restored at the end (never ship the
    last epoch's weights if they weren't the best - that's the overfit tail)

This is a deliberate departure from the original per-pipeline epoch/patience
values (crypto DL: 15 fixed epochs no early stop; stocks LSTM: 200/15;
mufap LSTM: 150/15; commodities DL: 150/20) - unified here for consistency
and to bound overfitting risk the same way everywhere.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.progress import epoch_bar

DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 22  # middle of the requested 20-25 range


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=min(batch_size, len(ds)) or 1, shuffle=shuffle)


def train_torch_model(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: torch.device,
    desc: str,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    loss_fn: torch.nn.Module | None = None,
    grad_clip_norm: float | None = 1.0,
) -> tuple[torch.nn.Module, dict]:
    """
    Trains `model` in place (returns the same object with best-epoch weights
    loaded) and returns (model, history) where history has final train/val
    loss, epochs actually run, and whether early stopping fired.
    """
    model = model.to(device)
    criterion = loss_fn or torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=False)  # chronological order preserved
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    stopped_early = False
    start = time.perf_counter()

    bar = epoch_bar(epochs, desc)
    train_loss_final = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(1, n_batches)
        train_loss_final = train_loss

        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                val_running += criterion(pred, yb).item()
                val_batches += 1
        val_loss = val_running / max(1, val_batches)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        bar.set_postfix_str(
            f"train={train_loss:.5f} val={val_loss:.5f} best={best_val_loss:.5f} no_improve={epochs_no_improve}/{patience}"
        )
        bar.update(1)

        if epochs_no_improve >= patience:
            stopped_early = True
            break

    bar.close()
    elapsed = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "epochs_run": epoch,
        "epochs_max": epochs,
        "patience": patience,
        "stopped_early": stopped_early,
        "final_train_loss": round(train_loss_final, 6),
        "best_val_loss": round(best_val_loss, 6),
        "seconds": round(elapsed, 1),
    }
    return model, history


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
        out.append(model(xb).squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])
