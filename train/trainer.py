import torch
import torch.optim as optim
import torch.nn as nn
from losses.physics_loss import PhysicsInformedLoss

def train_model(model, train_loader, val_loader, device, epochs, lr, lambda_physics):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    loss_fn = PhysicsInformedLoss(model, device)

    train_losses = []
    val_losses = []
    physics_losses = []

    for epoch in range(epochs):
        model.train()

        total_train_loss = 0
        total_physics_loss = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            loss, data_loss, physics_loss = loss_fn.compute_total_loss(
                x_batch, y_batch, lambda_physics
            )

            loss.backward()
            optimizer.step()

            total_train_loss += data_loss.item()
            total_physics_loss += physics_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_physics_loss = total_physics_loss / len(train_loader)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_loss = nn.MSELoss()(pred, y_batch.unsqueeze(1))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        physics_losses.append(avg_physics_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: {avg_train_loss:.6f} | "
                f"Val: {avg_val_loss:.6f} | "
                f"Physics: {avg_physics_loss:.6f}"
            )

    return train_losses, val_losses, physics_losses