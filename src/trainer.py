import torch
import torch.nn as nn
import torch.optim as optim
from config import DEFAULT_EPOCHS, DEFAULT_LR, DEVICE

def train_model(
    model,
    train_loader,
    val_loader,
    teacher_probs_train,
    teacher_probs_val,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    device=DEVICE,
):
    model.to(device)
    teacher_probs_train = teacher_probs_train.to(device)
    teacher_probs_val = teacher_probs_val.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # L2 loss between predicted and teacher class proportions

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (x_batch, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            pred_probs = model(x_batch)
            bag_pred = pred_probs.mean(dim=0)
            target = teacher_probs_train[i].to(device, dtype=bag_pred.dtype)
            loss = loss_fn(bag_pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_total_loss = 0.0
            for j, (x_batch, _) in enumerate(val_loader):
                x_batch = x_batch.to(device)
                pred_probs = model(x_batch)
                bag_pred = pred_probs.mean(dim=0)
                target = teacher_probs_val[j].to(device, dtype=bag_pred.dtype)
                loss = loss_fn(bag_pred, target)
                val_total_loss += loss.item()
            avg_val_loss = val_total_loss / len(val_loader)
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )


def evaluate_model(model, data_loader, num_classes, device=DEVICE):
    """Return average MSE and cross entropy between predicted and true proportions."""
    model.to(device)
    model.eval()
    mse_total = 0.0
    ce_total = 0.0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            pred_probs = model(x_batch)
            bag_pred = pred_probs.mean(dim=0)
            counts = torch.bincount(y_batch, minlength=num_classes).float()
            bag_true = (counts / counts.sum()).to(device, dtype=bag_pred.dtype)
            mse_total += nn.functional.mse_loss(bag_pred, bag_true).item()
            ce_total += float((-bag_true * torch.log(bag_pred + 1e-9)).sum())
    avg_mse = mse_total / len(data_loader)
    avg_ce = ce_total / len(data_loader)
    return {"mse": avg_mse, "cross_entropy": avg_ce}
