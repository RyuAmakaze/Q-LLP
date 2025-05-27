import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, teacher_probs, epochs=10, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # L2 loss between predicted and teacher class proportions

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (x_batch, _) in enumerate(dataloader):
            optimizer.zero_grad()
            pred_probs = model(x_batch)
            loss = loss_fn(pred_probs, teacher_probs[i])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
