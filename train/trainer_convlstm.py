import torch
import torch.nn as nn
import torch.optim as optim

from losses.convlstm_loss import CombinedLoss


def train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        lr,
        grad_clip=1.0,
        lambda_boundary=0.1,
        lambda_gradient=0.001,
        T_0=20,
        T_mult=2
):
    """
    ConvLSTM 训练函数

    Parameters
    ----------
    model : ConvLSTM
    train_loader / val_loader : DataLoader
        X -> (B, T, C, H, W)
        Y -> (B, C, H, W)
    device : torch.device
    epochs : int
    lr : float
    grad_clip : float
        梯度裁剪 max_norm
    lambda_boundary : float
        边界损失权重
    lambda_gradient : float
        梯度损失权重
    T_0, T_mult :
        CosineAnnealingWarmRestarts 参数

    Returns
    -------
    train_losses, val_losses, best_model_state
    """

    criterion = CombinedLoss(
        boundary_pixels=4,
        boundary_weight=5.0,
        lambda_boundary=lambda_boundary,
        lambda_gradient=lambda_gradient
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=lr * 0.01
    )

    train_losses = []
    val_losses = []
    val_mse_only = []

    best_val_mse = float('inf')
    best_model_state = None

    print("\n" + "=" * 60)
    print("开始训练 ConvLSTM")
    print("=" * 60)
    print(f"Epochs:         {epochs}")
    print(f"Learning Rate:  {lr}")
    print(f"Grad Clip:      {grad_clip}")
    print(f"Lambda Boundary:{lambda_boundary}")
    print(f"Lambda Gradient:{lambda_gradient}")
    print(f"T_0 / T_mult:   {T_0} / {T_mult}")
    print(f"Batch Size:     {train_loader.batch_size}")
    print("=" * 60)

    for epoch in range(epochs):

        # ==========================
        # Train
        # ==========================
        model.train()

        epoch_train_loss = 0.0
        epoch_train_mse = 0.0

        for X, Y in train_loader:

            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()

            pred = model(X)

            loss, comps = criterion(pred, Y)

            loss.backward()

            # 梯度裁剪
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip
                )

            optimizer.step()

            epoch_train_loss += comps["total"]
            epoch_train_mse += comps["mse"]

        epoch_train_loss /= len(train_loader)
        epoch_train_mse /= len(train_loader)

        train_losses.append(epoch_train_loss)

        # ==========================
        # Validation
        # ==========================
        model.eval()

        epoch_val_loss = 0.0
        epoch_val_mse = 0.0

        with torch.no_grad():

            for X, Y in val_loader:

                X = X.to(device)
                Y = Y.to(device)

                pred = model(X)

                loss, comps = criterion(pred, Y)

                epoch_val_loss += comps["total"]
                epoch_val_mse += comps["mse"]

        epoch_val_loss /= len(val_loader)
        epoch_val_mse /= len(val_loader)

        val_losses.append(epoch_val_loss)
        val_mse_only.append(epoch_val_mse)

        scheduler.step()

        # 保存最优（以 val MSE 为准）
        if epoch_val_mse < best_val_mse:

            best_val_mse = epoch_val_mse

            best_model_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        # 打印
        if (epoch + 1) % 10 == 0:

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch+1:>4d}/{epochs}] "
                f"Train: {epoch_train_mse:.4e} "
                f"| Val MSE: {epoch_val_mse:.4e} "
                f"| LR: {current_lr:.2e}"
            )

    print("\n" + "=" * 60)
    print(f"训练完成 — Best Val MSE: {best_val_mse:.6e}")
    print("=" * 60)

    # 恢复最优权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses
