# data/dataset_convlstm.py

import os
import numpy as np
import pandas as pd
import torch

from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# ==========================================
# 读取所有ASC文件
# ==========================================
def load_all_data(file_paths):

    all_velocity = []

    x_ref = None
    y_ref = None

    for i, file_path in enumerate(file_paths):

        print(
            f"\n读取文件 {i+1}/{len(file_paths)} : "
            f"{os.path.basename(file_path)}"
        )

        df = pd.read_csv(
            file_path,
            sep=',',
            skiprows=1,
            header=None,
            engine='python'
        )

        if df.shape[1] != 6:
            raise ValueError(
                f"{file_path} 列数错误: {df.shape[1]}"
            )

        df.columns = [
            'node',
            'x',
            'y',
            'z',
            'velocity',
            'mean_velocity'
        ]

        for col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors='coerce'
            )

        df = df.dropna()

        if x_ref is None:
            x_ref = df["x"].values
            y_ref = df["y"].values

        velocity = df["mean_velocity"].values

        all_velocity.append(velocity)

        print(
            f"速度范围:"
            f"[{velocity.min():.6f},"
            f"{velocity.max():.6f}]"
        )

    all_velocity = np.array(all_velocity)

    return x_ref, y_ref, all_velocity


# ==========================================
# 插值到规则网格
# ==========================================
def build_grid_data(
        x,
        y,
        all_velocity,
        grid_size=(64, 64)
):

    H, W = grid_size

    x_grid = np.linspace(
        x.min(),
        x.max(),
        W
    )

    y_grid = np.linspace(
        y.min(),
        y.max(),
        H
    )

    X_grid, Y_grid = np.meshgrid(
        x_grid,
        y_grid
    )

    flow_fields = []

    T = all_velocity.shape[0]

    for t in range(T):

        velocity = all_velocity[t]

        field = griddata(
            (x, y),
            velocity,
            (X_grid, Y_grid),
            method='linear'
        )

        # 补洞
        mask = np.isnan(field)

        if np.any(mask):

            nearest_field = griddata(
                (x, y),
                velocity,
                (X_grid, Y_grid),
                method='nearest'
            )

            field[mask] = nearest_field[mask]

        flow_fields.append(field)

    flow_fields = np.array(flow_fields)

    print(
        f"\n流场形状:"
        f"{flow_fields.shape}"
    )

    return (
        flow_fields,
        x_grid,
        y_grid
    )


# ==========================================
# MinMax 归一化到 [-1, 1]
# ==========================================
def normalize_data(flow_fields):
    """
    MinMaxScaler(-1, 1):
      保留边界区域的相对大小关系，
      相比 z-score 更适合流场数据
    """

    T, H, W = flow_fields.shape

    # 重塑为 (N, 1) 以获得全局最小/最大值，而不是按空间位置的
    flat = flow_fields.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    flat_scaled = scaler.fit_transform(flat)

    flow_scaled = flat_scaled.reshape(T, H, W)

    # 保存用于反归一化的参数
    data_min = scaler.data_min_.item()
    data_max = scaler.data_max_.item()

    scale = (data_max - data_min) / 2.0
    shift = (data_max + data_min) / 2.0

    print(f"\n归一化: MinMaxScaler(-1, 1)")
    print(f"  原始范围: [{flat.min():.4f}, {flat.max():.4f}]")
    print(f"  缩放后范围: [{flow_scaled.min():.4f}, {flow_scaled.max():.4f}]")

    return (
        flow_scaled,
        shift,   # 相当于 mean
        scale    # 相当于 std
    )


# ==========================================
# ConvLSTM Dataset
# ==========================================
class FlowDataset(Dataset):

    def __init__(
            self,
            data,
            input_steps
    ):

        self.X = []
        self.Y = []

        total_steps = len(data)

        for i in range(
                total_steps - input_steps
        ):

            x = data[
                i:i + input_steps
            ]

            y = data[
                i + input_steps
            ]

            self.X.append(x)
            self.Y.append(y)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        print(
            f"\n构造序列完成:"
        )

        print(
            f"X shape = {self.X.shape}"
        )

        print(
            f"Y shape = {self.Y.shape}"
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = torch.FloatTensor(
            self.X[idx]
        )

        y = torch.FloatTensor(
            self.Y[idx]
        )

        return x, y


# ==========================================
# 创建DataLoader
# ==========================================
def build_dataloaders(
        file_paths,
        input_steps,
        batch_size,
        device,
        grid_size=(64, 64),
        train_ratio=0.8
):

    print("=" * 60)
    print("开始加载流场数据")
    print("=" * 60)

    x, y, all_velocity = load_all_data(
        file_paths
    )

    (
        flow_fields,
        grid_x,
        grid_y
    ) = build_grid_data(
        x,
        y,
        all_velocity,
        grid_size
    )

    flow_scaled, mean, std = normalize_data(
        flow_fields
    )

    # 增加通道维度
    flow_scaled = flow_scaled[:, None, :, :]

    print(
        f"\n最终数据形状:"
        f"{flow_scaled.shape}"
    )

    T = len(flow_scaled)

    train_end = int(
        T * train_ratio
    )

    train_data = flow_scaled[
                 :train_end
                 ]

    val_data = flow_scaled[
               train_end - input_steps:
               ]

    print(
        f"\n训练时间步:"
        f"{len(train_data)}"
    )

    print(
        f"验证时间步:"
        f"{len(val_data)}"
    )

    train_dataset = FlowDataset(
        train_data,
        input_steps
    )

    val_dataset = FlowDataset(
        val_data,
        input_steps
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    data_info = {

        "grid_x": grid_x,
        "grid_y": grid_y,

        "mean": mean,
        "std": std,

        "H": grid_size[0],
        "W": grid_size[1],

        "input_steps": input_steps,

        "n_time_steps": T
    }

    print("\n数据加载完成")
    print(
        f"训练样本数: {len(train_dataset)}"
    )
    print(
        f"验证样本数: {len(val_dataset)}"
    )

    return (
        train_loader,
        val_loader,
        data_info
    )