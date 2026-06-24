import numpy as np
import matplotlib.pyplot as plt

def plot_flow_field(
        y_true,
        y_pred,
        step_idx=0,
        grid_x=None,
        grid_y=None,
        save_path=None
):
    """
    绘制某个验证样本的流场对比图

    Parameters
    ----------
    y_true : ndarray, shape (N,1,H,W)
    y_pred : ndarray, shape (N,1,H,W)
    step_idx : 验证样本编号
    grid_x  : x方向网格坐标
    grid_y  : y方向网格坐标
    save_path : 保存路径
    """

    true_field = y_true[step_idx, 0]
    pred_field = y_pred[step_idx, 0]

    # -------------------------
    # 误差场
    # -------------------------

    error_field = pred_field - true_field

    abs_error = np.abs(error_field)

    mean_abs_error = np.mean(abs_error)
    max_abs_error = np.max(abs_error)

    # -------------------------
    # 构建网格
    # -------------------------

    if grid_x is None or grid_y is None:

        H, W = true_field.shape

        X2D, Y2D = np.meshgrid(
            np.arange(W),
            np.arange(H)
        )

    else:

        X2D, Y2D = np.meshgrid(
            grid_x,
            grid_y
        )

    # -------------------------
    # 创建图形
    # -------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        dpi=150
    )

    pcm_kwargs = dict(
        shading="gouraud",
        rasterized=True
    )

    # -------------------------
    # 统一颜色轴范围
    # -------------------------

    vmin = min(true_field.min(), pred_field.min())
    vmax = max(true_field.max(), pred_field.max())

    # =====================
    # 真实流场
    # =====================

    im1 = axes[0].pcolormesh(
        X2D,
        Y2D,
        true_field,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        **pcm_kwargs
    )

    axes[0].set_title(
        "Real Velocity Field",
        fontsize=12,
        fontweight="bold"
    )

    plt.colorbar(
        im1,
        ax=axes[0]
    )

    # =====================
    # 预测流场
    # =====================

    im2 = axes[1].pcolormesh(
        X2D,
        Y2D,
        pred_field,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        **pcm_kwargs
    )

    axes[1].set_title(
        "Predicted Velocity Field",
        fontsize=12,
        fontweight="bold"
    )

    plt.colorbar(
        im2,
        ax=axes[1]
    )

    # =====================
    # 绝对误差场
    # =====================

    im3 = axes[2].pcolormesh(
        X2D,
        Y2D,
        abs_error,
        cmap="jet",
        **pcm_kwargs
    )

    axes[2].set_title(
        f"Absolute Error | Mean={mean_abs_error:.4f} | Max={max_abs_error:.4f}",
        fontsize=12,
        fontweight="bold"
    )

    plt.colorbar(
        im3,
        ax=axes[2],
        label="Absolute Error"
    )

    # -------------------------
    # 坐标比例保持
    # -------------------------

    for ax in axes:

        ax.set_aspect("equal")

        if grid_x is not None:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

    plt.suptitle(
        f"ConvLSTM Prediction Result (Sample {step_idx})",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    # -------------------------
    # 保存
    # -------------------------

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        print(f"图片已保存: {save_path}")

    plt.show()

    plt.close(fig)


def plot_training_results(
        train_losses,
        val_losses,
        y_true,
        y_pred,
        r2,
        step_idx=0,
        grid_x=None,
        grid_y=None,
        save_path=None
):
    """
    综合绘制 ConvLSTM 训练结果

    Parameters
    ----------
    train_losses : list
    val_losses   : list
    y_true : ndarray, shape (N,1,H,W)
    y_pred : ndarray, shape (N,1,H,W)
    r2     : float
    step_idx  : 展示的样本编号
    grid_x, grid_y : 网格坐标
    save_path : 保存路径
    """

    true_field = y_true[step_idx, 0]
    pred_field = y_pred[step_idx, 0]
    abs_error = np.abs(pred_field - true_field)

    H, W = true_field.shape

    if grid_x is None or grid_y is None:
        X2D, Y2D = np.meshgrid(np.arange(W), np.arange(H))
    else:
        X2D, Y2D = np.meshgrid(grid_x, grid_y)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10),
        dpi=150
    )

    pcm_kwargs = dict(shading="gouraud", rasterized=True)

    # =====================
    # Loss 曲线
    # =====================

    axes[0, 0].plot(train_losses, label="Train Loss", linewidth=2)
    axes[0, 0].plot(val_losses,   label="Val Loss",   linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # =====================
    # 预测 vs 真实 散点
    # =====================

    axes[0, 1].scatter(
        y_true.flatten(),
        y_pred.flatten(),
        s=1, alpha=0.3
    )

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    axes[0, 1].plot([vmin, vmax], [vmin, vmax], "r--", linewidth=2)

    axes[0, 1].set_xlabel("Real Velocity")
    axes[0, 1].set_ylabel("Predicted Velocity")
    axes[0, 1].set_title(f"Prediction Scatter (R²={r2:.4f})")
    axes[0, 1].grid(True)

    # =====================
    # 真实流场
    # =====================

    im1 = axes[1, 0].pcolormesh(
        X2D, Y2D, true_field,
        cmap="jet", **pcm_kwargs
    )

    axes[1, 0].set_title(
        f"Real Velocity Field (Sample {step_idx})",
        fontsize=12, fontweight="bold"
    )

    plt.colorbar(im1, ax=axes[1, 0])

    # =====================
    # 绝对误差场
    # =====================

    im2 = axes[1, 1].pcolormesh(
        X2D, Y2D, abs_error,
        cmap="jet", **pcm_kwargs
    )

    mean_err = np.mean(abs_error)
    max_err  = np.max(abs_error)

    axes[1, 1].set_title(
        f"Absolute Error | Mean={mean_err:.4f} | Max={max_err:.4f}",
        fontsize=12, fontweight="bold"
    )

    plt.colorbar(im2, ax=axes[1, 1], label="Absolute Error")

    # -------------------------
    # 坐标比例
    # -------------------------

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_aspect("equal")
        if grid_x is not None:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

    plt.suptitle("ConvLSTM Training Results", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")

    plt.show()
    plt.close(fig)