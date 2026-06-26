import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np

from config_convlstm import (
    FILE_PATHS, INPUT_STEPS, HIDDEN_DIMS, KERNEL_SIZE,
    GRID_SIZE, TRAIN_RATIO, BATCH_SIZE, EPOCHS, LR,
    GRAD_CLIP, LAMBDA_BOUNDARY, LAMBDA_GRADIENT,
    T_0, T_MULT, MODEL_SAVE_PATH,
    RESULT_FIG_PATH, FLOW_FIELD_FIG_PATH
)

from data.dataset_convlstm import build_dataloaders

from models.convlstm import ConvLSTM

from train.trainer_convlstm import train_model

from utils.metrics_convlstm import (
    evaluate_model,
    print_metrics
)

from utils.visualize_convlstm import (
    plot_training_results,
    plot_flow_field
)

from utils.save_convlstm import save_model


# =====================================
# 随机种子
# =====================================

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# =====================================
# 设备
# =====================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"使用设备: {device}")

if device.type == "cuda":

    torch.cuda.init()

    _ = torch.zeros(
        1,
        device=device
    )

    torch.cuda.synchronize()


# =====================================
# 数据加载
# =====================================

train_loader, val_loader, data_info = build_dataloaders(

    file_paths=FILE_PATHS,

    input_steps=INPUT_STEPS,

    batch_size=BATCH_SIZE,

    device=device,

    grid_size=GRID_SIZE,

    train_ratio=TRAIN_RATIO
)


# =====================================
# 构建模型
# =====================================

model = ConvLSTM(

    input_dim=1,

    hidden_dims=HIDDEN_DIMS,

    kernel_size=KERNEL_SIZE,

    output_dim=1

).to(device)

print("\n模型结构:")
print(model)

print(
    f"\n模型参数量: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# =====================================
# 训练
# =====================================

train_losses, val_losses = train_model(

    model=model,

    train_loader=train_loader,

    val_loader=val_loader,

    device=device,

    epochs=EPOCHS,

    lr=LR,

    grad_clip=GRAD_CLIP,

    lambda_boundary=LAMBDA_BOUNDARY,

    lambda_gradient=LAMBDA_GRADIENT,

    T_0=T_0,

    T_mult=T_MULT
)


# =====================================
# 验证评估
# =====================================

y_pred, metrics = evaluate_model(

    model=model,

    val_loader=val_loader,

    device=device,

    mean=data_info["mean"],

    std=data_info["std"]
)

print_metrics(metrics)


# =====================================
# 获取真实验证数据
# =====================================

all_true = []

model.eval()

with torch.no_grad():

    for X, Y in val_loader:

        all_true.append(
            Y.numpy()
        )

y_true = np.concatenate(
    all_true,
    axis=0
)

# 反归一化

y_true = (
    y_true
    * data_info["std"]
    + data_info["mean"]
)


# =====================================
# 可视化
# =====================================

plot_training_results(

    train_losses=train_losses,

    val_losses=val_losses,

    y_true=y_true,

    y_pred=y_pred,

    r2=metrics["r2"],

    step_idx=0,

    grid_x=data_info["grid_x"],

    grid_y=data_info["grid_y"],

    save_path=RESULT_FIG_PATH
)


plot_flow_field(

    y_true=y_true,

    y_pred=y_pred,

    step_idx=0,

    grid_x=data_info["grid_x"],

    grid_y=data_info["grid_y"],

    save_path=FLOW_FIELD_FIG_PATH
)


# =====================================
# 保存模型
# =====================================

save_model(

    model=model,

    path=MODEL_SAVE_PATH,

    mean=data_info["mean"],

    std=data_info["std"],

    grid_x=data_info["grid_x"],

    grid_y=data_info["grid_y"],

    input_steps=INPUT_STEPS,

    metrics=metrics
)

print("\n训练结束")