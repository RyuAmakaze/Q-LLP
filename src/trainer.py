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
