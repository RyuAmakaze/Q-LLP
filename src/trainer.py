import torch
import torch.nn as nn
import torch.optim as optim
from config import DEFAULT_EPOCHS, DEFAULT_LR

def train_model(model, dataloader, teacher_probs, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # L2 loss between predicted and teacher class proportions

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (x_batch, _) in enumerate(dataloader):
            optimizer.zero_grad()
            pred_probs = model(x_batch)
            bag_pred = pred_probs.mean(dim=0)
            target = teacher_probs[i].to(bag_pred.dtype)
            loss = loss_fn(bag_pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
