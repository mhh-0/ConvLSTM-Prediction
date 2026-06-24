import numpy as np
import torch

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def evaluate_model(
        model,
        val_loader,
        device,
        mean,
        std
):
    """
    ConvLSTM模型评估

    参数:
        model
        val_loader
        device
        mean
        std

    返回:
        predictions
        metrics
    """

    model.eval()

    all_pred = []
    all_true = []

    with torch.no_grad():

        for X, Y in val_loader:

            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)

            all_pred.append(
                pred.cpu().numpy()
            )

            all_true.append(
                Y.cpu().numpy()
            )

    pred = np.concatenate(
        all_pred,
        axis=0
    )

    true = np.concatenate(
        all_true,
        axis=0
    )

    # ==========================
    # 反归一化
    # ==========================
    pred_real = pred * std + mean
    true_real = true * std + mean

    # ==========================
    # 展平
    # ==========================
    pred_flat = pred_real.flatten()
    true_flat = true_real.flatten()

    mse = mean_squared_error(
        true_flat,
        pred_flat
    )

    mae = mean_absolute_error(
        true_flat,
        pred_flat
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        true_flat,
        pred_flat
    )

    metrics = {

        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2

    }

    return pred_real, metrics


def print_metrics(metrics):

    print("\n" + "=" * 50)
    print("ConvLSTM 验证结果")
    print("=" * 50)

    print(
        f"MSE  : {metrics['mse']:.6e}"
    )

    print(
        f"MAE  : {metrics['mae']:.6e}"
    )

    print(
        f"RMSE : {metrics['rmse']:.6e}"
    )

    print(
        f"R²   : {metrics['r2']:.6f}"
    )

    print("=" * 50)