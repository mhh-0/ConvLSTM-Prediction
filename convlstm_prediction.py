"""
convlstm_prediction.py
======================
使用 main_convlstm.py 训练出的模型 (convlstm_flow_model.pth) 进行
多步时序流场预测，并通过可拖动的时间轴逐帧展示预测结果。

用法:
    python convlstm_prediction.py

依赖:
    pip install torch numpy pandas scipy scikit-learn matplotlib
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch

from config_convlstm import (
    FILE_PATHS, INPUT_STEPS, HIDDEN_DIMS, KERNEL_SIZE,
    GRID_SIZE, MODEL_SAVE_PATH
)
from models.convlstm import ConvLSTM
from data.dataset_convlstm import load_all_data, build_grid_data, normalize_data


# =====================================
# 0. 设备 & 随机种子
# =====================================
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================
# 1. 加载已训练的模型
# =====================================
def load_trained_model(checkpoint_path: str, device: torch.device):
    """从 checkpoint 中恢复模型和元数据。"""
    print(f"\nLoading model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ConvLSTM(
        input_dim=1, hidden_dims=HIDDEN_DIMS,
        kernel_size=KERNEL_SIZE, output_dim=1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["mean"]
    std  = checkpoint["std"]
    grid_x = checkpoint["grid_x"]
    grid_y = checkpoint["grid_y"]
    input_steps = checkpoint.get("input_steps", INPUT_STEPS)

    print(f"   input_steps  : {input_steps}")
    print(f"   mean (shift) : {mean:.6f}")
    print(f"   std  (scale) : {std:.6f}")
    print(f"   grid size    : {len(grid_y)} x {len(grid_x)}")
    print(f"   params       : {sum(p.numel() for p in model.parameters()):,}")

    if "metrics" in checkpoint:
        m = checkpoint["metrics"]
        print(f"   train R2     : {m.get('r2', float('nan')):.6f}")
        print(f"   train RMSE   : {m.get('rmse', float('nan')):.6e}")

    return model, mean, std, grid_x, grid_y, input_steps


# =====================================
# 2. 准备预测用的数据
# =====================================
def prepare_prediction_data(file_paths: list, grid_size: tuple):
    """加载并归一化全部流场数据。"""
    print("\n" + "=" * 60)
    print("Loading flow field data")
    print("=" * 60)

    x, y, all_velocity = load_all_data(file_paths)
    flow_fields, grid_x, grid_y = build_grid_data(x, y, all_velocity, grid_size)
    flow_scaled, mean, std = normalize_data(flow_fields)

    flow_scaled = flow_scaled[:, None, :, :]   # (T, 1, H, W)

    T = len(flow_scaled)
    print(f"\nTotal time steps: {T}")
    print(f"Data shape      : {flow_scaled.shape}")

    return flow_scaled, mean, std, grid_x, grid_y


# =====================================
# 3. 单步预测（Teacher-Forcing，无误差累积）
# =====================================
@torch.no_grad()
def single_step_predict(
    model: ConvLSTM,
    data_scaled: np.ndarray,   # (T, 1, H, W)
    input_steps: int,
    device: torch.device,
) -> np.ndarray:
    """
    验证集单步预测：对每个 t ∈ [input_steps, T)，
    用 [t-input_steps, t) 的真实值预测 t。
    """
    model.eval()
    T = len(data_scaled)
    predictions, targets = [], []

    for t in range(input_steps, T):
        window = torch.from_numpy(data_scaled[t - input_steps:t]).float()
        window = window.unsqueeze(0).to(device)           # (1, S, C, H, W)
        pred = model(window)                               # (1, C, H, W)
        predictions.append(pred.cpu().numpy()[0, 0])       # (H, W)
        targets.append(data_scaled[t, 0])                  # (H, W)

    return np.array(predictions), np.array(targets)


# =====================================
# 4. 自回归多步预测
# =====================================
@torch.no_grad()
def autoregressive_predict(
    model: ConvLSTM,
    seed_sequence: np.ndarray,   # (input_steps, 1, H, W)
    num_steps: int,
    device: torch.device,
) -> np.ndarray:
    """自回归多步预测。"""
    model.eval()
    window = torch.from_numpy(seed_sequence).float().unsqueeze(0).to(device)
    predictions = []

    for step in range(num_steps):
        pred = model(window)
        predictions.append(pred.cpu().numpy()[0, 0])
        if step < num_steps - 1:
            window = torch.cat([window[:, 1:], pred.unsqueeze(1)], dim=1)

    return np.array(predictions)


# =====================================
# 5. 反归一化
# =====================================
def denormalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """MinMax(-1,1) 反归一化:  original = data * std + mean"""
    return data * std + mean


# =====================================
# 6. 生成预测步的文件名
# =====================================
def build_prediction_labels(file_paths: list, num_pred: int) -> list:
    """为自回归预测步生成虚拟标签。"""
    last_path = file_paths[-1]
    base_name = os.path.basename(last_path)
    match = re.search(r'(\d+)$', base_name.replace('.asc', ''))
    if match:
        last_num = int(match.group(1))
        prefix = base_name[:match.start()]
        return [f"{prefix}{last_num + (i + 1) * 10:05d} (pred)" for i in range(num_pred)]
    return [f"pred_step_{i+1:03d}" for i in range(num_pred)]


# =====================================
# 7. 诊断面板（先弹出）
# =====================================
def plot_diagnostic_dashboard(
    true_all: np.ndarray,       # (T, H, W)
    val_pred: np.ndarray,       # (V, H, W)
    val_target: np.ndarray,     # (V, H, W)
    auto_pred: np.ndarray,      # (P, H, W)
    input_steps: int,
    file_labels: list,
):
    """弹出诊断面板。"""
    T = len(true_all)
    V = len(val_pred)
    P = len(auto_pred)

    val_mae = np.abs(val_pred - val_target).mean(axis=(1, 2))
    val_mae_overall = val_mae.mean()

    auto_means = auto_pred.mean(axis=(1, 2))
    auto_mins  = auto_pred.min(axis=(1, 2))
    auto_maxs  = auto_pred.max(axis=(1, 2))
    auto_stds  = auto_pred.std(axis=(1, 2))

    seed_means = true_all[-input_steps:].mean(axis=(1, 2))
    all_means  = true_all.mean(axis=(1, 2))
    all_mins   = true_all.min(axis=(1, 2))
    all_maxs   = true_all.max(axis=(1, 2))

    fig = plt.figure(figsize=(18, 12), dpi=120)

    # ---- 全时间线值域 ----
    ax1 = fig.add_subplot(2, 3, (1, 2))
    ax1.fill_between(range(T), all_mins, all_maxs, alpha=0.2, color='green', label='Actual range')
    ax1.plot(range(T), all_means, 'g-', linewidth=2, label='Actual mean')
    ax1.axvline(T - input_steps, color='gray', linestyle='--', alpha=0.7, label='Seed start')

    ar_x = np.arange(T, T + P)
    ax1.fill_between(ar_x, auto_mins, auto_maxs, alpha=0.2, color='red', label='AutoReg range')
    ax1.plot(ar_x, auto_means, 'r-', linewidth=2, label='AutoReg mean')
    ax1.axvline(T, color='red', linestyle=':', alpha=0.7, label='Prediction start')

    ax1.set_xlabel("Time Step Index")
    ax1.set_ylabel("Velocity")
    ax1.set_title("Value Range Drift: Actual vs Autoregressive Prediction", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- 验证集单步 MAE ----
    ax2 = fig.add_subplot(2, 3, 3)
    val_indices = np.arange(input_steps, T)
    colors = ['darkgreen' if i < T * 0.8 else 'darkblue' for i in val_indices]
    _ = ax2.bar(range(V), val_mae, color=colors, edgecolor='white')
    ax2.axhline(val_mae_overall, color='red', linestyle='--', linewidth=2,
                label=f'Mean MAE = {val_mae_overall:.4f}')
    ax2.set_xlabel("Validation Sample Index")
    ax2.set_ylabel("MAE")
    ax2.set_title("Single-Step MAE (no accumulation)", fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # ---- 值域漂移 ----
    ax3 = fig.add_subplot(2, 3, 4)
    steps = np.arange(1, P + 1)
    ax3.plot(steps, auto_means, 'b-o', markersize=4, linewidth=2, label='AutoReg Mean')
    ax3.fill_between(steps, auto_mins, auto_maxs, alpha=0.15, color='blue')
    ax3.axhline(seed_means[-1], color='green', linestyle='--', linewidth=1.5,
                label=f'Last seed mean = {seed_means[-1]:.4f}')
    ax3.axhline(all_means.mean(), color='gray', linestyle=':', linewidth=1.5,
                label=f'Global mean = {all_means.mean():.4f}')
    if P >= 2:
        drift_start = auto_means[:5].mean()
        drift_end = auto_means[-5:].mean()
        drift_pct = (drift_end - drift_start) / (abs(drift_start) + 1e-8) * 100
        ax3.text(0.5, 0.05, f'Drift: {drift_pct:+.1f}% over {P} steps',
                 transform=ax3.transAxes, fontsize=12, color='red',
                 ha='center', fontweight='bold')
    ax3.set_xlabel("Prediction Step")
    ax3.set_ylabel("Velocity")
    ax3.set_title("Autoregressive Value Drift", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- 标准差漂移 ----
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(steps, auto_stds, 'r-o', markersize=4, linewidth=2, label='AutoReg Std')
    seed_std = true_all[-input_steps:].std(axis=(1, 2))
    ax4.axhline(seed_std.mean(), color='green', linestyle='--', linewidth=1.5,
                label=f'Seed mean std = {seed_std.mean():.4f}')
    ax4.axhline(true_all.std(axis=(1, 2)).mean(), color='gray', linestyle=':', linewidth=1.5,
                label=f'Global mean std = {true_all.std(axis=(1,2)).mean():.4f}')
    ax4.set_xlabel("Prediction Step")
    ax4.set_ylabel("Spatial Std")
    ax4.set_title("Spatial Variability Drift", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ---- 最佳/最差样本 ----
    ax5 = fig.add_subplot(2, 3, 6)
    best_idx = int(np.argmin(val_mae))
    worst_idx = int(np.argmax(val_mae))
    ax5.barh(['Best', 'Worst'], [val_mae[best_idx], val_mae[worst_idx]],
             color=['darkgreen', 'darkred'], edgecolor='white', height=0.5)
    ax5.set_xlabel("MAE")
    ax5.set_title(f"Best vs Worst Single-Step\n"
                  f"Best:  t={best_idx + input_steps}  "
                  f"({file_labels[best_idx + input_steps][:30]}...)\n"
                  f"Worst: t={worst_idx + input_steps}  "
                  f"({file_labels[worst_idx + input_steps][:30]}...)",
                  fontsize=8)
    for i, v in enumerate([val_mae[best_idx], val_mae[worst_idx]]):
        ax5.text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

    plt.suptitle("ConvLSTM Prediction Diagnostic Dashboard", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =====================================
# 8. 构建交叉排列的帧列表
# =====================================
def build_interleaved_frames(
    true_frames: np.ndarray,        # (T, H, W)
    val_pred_frames: np.ndarray,    # (V, H, W)
    val_target_frames: np.ndarray,  # (V, H, W)
    auto_pred_frames: np.ndarray,   # (P, H, W)
    true_labels: list,              # [str] * T
    pred_labels: list,              # [str] * P
    input_steps: int,
    val_mae: np.ndarray,            # (V,)
) -> tuple:
    """
    构建交叉排列的帧序列:

      [t0..t7 真实]  (t8 真实, t8 预测), (t9 真实, t9 预测), ...  [自回归预测...]

    Returns
    -------
    frames     : list of (H, W) arrays
    frame_info : list of dicts with keys: type, label, color, mae, target
    """
    V = len(val_pred_frames)
    P = len(auto_pred_frames)

    frames = []
    frame_info = []

    # ---- 1. 种子窗口：前 input_steps 帧真实（无对应预测） ----
    for i in range(input_steps):
        frames.append(true_frames[i])
        frame_info.append({
            'type': 'ACTUAL (Seed)', 'label': true_labels[i],
            'color': 'darkblue', 'mae': None, 'target': None,
            'true_idx': i,
        })

    # ---- 2. 交叉排列：tᵢ 真实 → tᵢ 预测（i = input_steps .. T-1） ----
    for vi in range(V):
        ti = vi + input_steps  # 在 true_frames 中的索引

        # 真实帧
        frames.append(true_frames[ti])
        frame_info.append({
            'type': 'ACTUAL',
            'label': true_labels[ti],
            'color': 'darkgreen',
            'mae': val_mae[vi],
            'target': val_pred_frames[vi],    # 存储对应预测，用于显示对比
            'true_idx': ti,
        })

        # 单步预测帧
        frames.append(val_pred_frames[vi])
        frame_info.append({
            'type': 'SINGLE-STEP PRED',
            'label': f"pred <- {true_labels[ti]}",
            'color': 'darkorange',
            'mae': val_mae[vi],
            'target': val_target_frames[vi],  # 存储真实目标，用于 Tab 切换
            'true_idx': ti,
        })

    # ---- 3. 自回归预测 ----
    for ai in range(P):
        frames.append(auto_pred_frames[ai])
        frame_info.append({
            'type': 'AUTOREGRESSIVE',
            'label': pred_labels[ai],
            'color': 'darkred',
            'mae': None,
            'target': None,
            'true_idx': None,
        })

    return frames, frame_info


# =====================================
# 9. 可拖动时间轴流场浏览器
# =====================================
class FlowFieldViewer:
    """
    交互式流场查看器 —— 真实帧和预测帧交叉排列，用颜色区分。
    按 Tab 可在单步预测帧上切换 预测/目标 显示。
    """

    def __init__(
        self,
        frames: list,            # [(H, W), ...]
        frame_info: list,        # [{type, label, color, mae, target}, ...]
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        title: str = "ConvLSTM Flow Field Prediction",
    ):
        self.frames = frames
        self.frame_info = frame_info
        self.num_total = len(frames)
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.base_title = title

        # 全局颜色范围
        all_data = np.stack(frames, axis=0)
        self.vmin = all_data.min()
        self.vmax = all_data.max()

        # 对单步预测帧，维护 "显示预测" 还是 "显示目标" 的状态
        self.show_target = {}   # frame_idx -> bool

        # 创建图形
        self.fig = plt.figure(figsize=(16, 11), dpi=120)

        # 默认跳到第一对（真实 + 预测），即 input_steps 位置
        self.current_idx = 0
        self._setup_ui()

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._update_frame(self.current_idx)

    # ------------------------------------------------------------------
    def _get_display_data(self, idx: int):
        """
        返回实际要显示的 (field, info)。

        对单步预测帧，如果 show_target[idx] == True，显示 target 而非 pred。
        """
        info = self.frame_info[idx].copy()

        if info['type'] == 'SINGLE-STEP PRED' and self.show_target.get(idx, False):
            field = info['target']   # 显示真实目标
            info['type'] = 'VAL TARGET'
            info['color'] = 'darkblue'
            info['label'] = f"target <- {info['label'].replace('pred <- ', '')}"
        else:
            field = self.frames[idx]

        return field, info

    # ------------------------------------------------------------------
    def _setup_ui(self):
        """搭建布局：主流场 + 误差/梯度图 + 统计面板 + 滑块 + 信息栏。"""
        # 主图
        self.ax_field = self.fig.add_axes([0.06, 0.18, 0.60, 0.72])
        self.ax_field.set_aspect("equal")
        self.ax_field.set_xlabel("X")
        self.ax_field.set_ylabel("Y")

        # 右侧上图：误差/梯度
        self.ax_aux = self.fig.add_axes([0.70, 0.45, 0.28, 0.45])
        self.ax_aux.set_aspect("equal")
        self.ax_aux.set_xlabel("X")
        self.ax_aux.set_ylabel("Y")

        # 右侧下图：统计面板
        self.ax_stats = self.fig.add_axes([0.70, 0.18, 0.28, 0.24])
        self.ax_stats.axis("off")

        # 滑块
        self.ax_slider = self.fig.add_axes([0.10, 0.06, 0.80, 0.04])
        self.slider = Slider(
            ax=self.ax_slider, label="Time Step",
            valmin=0, valmax=max(self.num_total - 1, 0),
            valinit=0, valfmt="%d", valstep=1,
        )
        self.slider.on_changed(self._on_slider)

        # 信息栏
        self.ax_info = self.fig.add_axes([0.10, 0.92, 0.80, 0.04])
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(
            0.5, 0.5, "",
            transform=self.ax_info.transAxes,
            ha="center", va="center", fontsize=9, fontweight="bold",
        )

        self._cbar_main = None
        self._cbar_aux = None

    # ------------------------------------------------------------------
    def _update_frame(self, idx: int):
        """刷新所有面板。"""
        idx = int(np.clip(idx, 0, self.num_total - 1))
        self.current_idx = idx

        field, info = self._get_display_data(idx)
        X2D, Y2D = np.meshgrid(self.grid_x, self.grid_y)

        # ==== 主图 ====
        self.ax_field.clear()
        im = self.ax_field.pcolormesh(
            X2D, Y2D, field,
            cmap="jet", shading="gouraud", rasterized=True,
            vmin=self.vmin, vmax=self.vmax,
        )
        if self._cbar_main:
            self._cbar_main.remove()
        self._cbar_main = self.fig.colorbar(im, ax=self.ax_field, label="Velocity")

        self.ax_field.set_aspect("equal")
        self.ax_field.set_xlabel("X")
        self.ax_field.set_ylabel("Y")
        self.ax_field.set_title(
            f"[{info['type']}]\nFrame {idx} / {self.num_total - 1}",
            fontsize=13, fontweight="bold", color=info['color'],
        )

        # ==== 右侧辅助图 ====
        self.ax_aux.clear()

        if info['type'] in ('SINGLE-STEP PRED', 'VAL TARGET'):
            # 显示预测 vs 目标的误差场（取 mae/target 中存储的值）
            # 此时 info['target'] 可能是 target（当 show_target=False）或 pred（当 show_target=True）
            # 我们需要从原始 frame_info 中取对应的 pair
            orig = self.frame_info[idx]
            if orig['type'] == 'SINGLE-STEP PRED':
                # pred vs target
                pred_field = self.frames[idx]
                target_field = orig['target']
                error = np.abs(pred_field - target_field)
            else:
                # 已被 get_display_data 转为 VAL TARGET，则 target 存的是 pred
                # 实际上 _get_display_data 处理了这个情况
                # 从原始帧数据获取 pred
                pred_field = self.frames[idx]
                target_field = orig['target']
                error = np.abs(pred_field - target_field)

            im_aux = self.ax_aux.pcolormesh(
                X2D, Y2D, error,
                cmap="hot", shading="gouraud", rasterized=True,
            )
            if self._cbar_aux:
                self._cbar_aux.remove()
            self._cbar_aux = self.fig.colorbar(im_aux, ax=self.ax_aux, label="Absolute Error")
            mae_val = orig.get('mae', np.nan)
            self.ax_aux.set_title(
                f"|Prediction - Target|\nMAE = {mae_val:.4f}",
                fontsize=10, fontweight="bold", color="darkred",
            )
        else:
            # 真实帧 / 自回归帧：显示空间梯度
            gy, gx = np.gradient(field)
            grad_mag = np.sqrt(gx**2 + gy**2)
            im_aux = self.ax_aux.pcolormesh(
                X2D, Y2D, grad_mag,
                cmap="viridis", shading="gouraud", rasterized=True,
            )
            if self._cbar_aux:
                self._cbar_aux.remove()
            self._cbar_aux = self.fig.colorbar(im_aux, ax=self.ax_aux, label="Gradient Mag")
            self.ax_aux.set_title(
                "Spatial Gradient Magnitude",
                fontsize=10, fontweight="bold", color=info['color'],
            )

        self.ax_aux.set_aspect("equal")
        self.ax_aux.set_xlabel("X")
        self.ax_aux.set_ylabel("Y")

        # ==== 统计面板 ====
        self.ax_stats.clear()
        self.ax_stats.axis("off")

        tab_hint = ""
        if info['type'] in ('SINGLE-STEP PRED', 'VAL TARGET'):
            tab_hint = "\n[Tab] Toggle Pred/Target\n"

        stats = (
            f"Frame:  {idx} / {self.num_total - 1}\n"
            f"Type:   {info['type']}\n"
            f"File:   {info['label']}\n"
            f"{'─' * 30}\n"
            f"Min:    {field.min():.4f}\n"
            f"Max:    {field.max():.4f}\n"
            f"Mean:   {field.mean():.4f}\n"
            f"Std:    {field.std():.4f}\n"
            f"{'─' * 30}\n"
        )
        if info.get('mae') is not None:
            stats += f"MAE:    {info['mae']:.4f}\n"
        stats += (
            f"{tab_hint}"
            f"[<- ->] Navigate\n"
            f"[Home / End] Jump"
        )

        self.ax_stats.text(
            0.05, 0.95, stats,
            transform=self.ax_stats.transAxes,
            fontsize=9, fontfamily='monospace',
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

        # ==== 信息栏 ====
        v_min, v_max, v_mean = field.min(), field.max(), field.mean()
        mae_str = f"  |  MAE: {info['mae']:.4f}" if info.get('mae') is not None else ""
        self.info_text.set_text(
            f"[{info['type']}]  {info['label']}  |  "
            f"Min: {v_min:.4f}  |  Max: {v_max:.4f}  |  Mean: {v_mean:.4f}{mae_str}"
        )
        self.info_text.set_color(info['color'])

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_slider(self, val):
        self._update_frame(int(round(val)))

    # ------------------------------------------------------------------
    def _on_key(self, event):
        if event.key in ("right", "down"):
            self._update_frame(self.current_idx + 1)
            self.slider.set_val(self.current_idx)
        elif event.key in ("left", "up"):
            self._update_frame(self.current_idx - 1)
            self.slider.set_val(self.current_idx)
        elif event.key == "home":
            self._update_frame(0)
            self.slider.set_val(0)
        elif event.key == "end":
            self._update_frame(self.num_total - 1)
            self.slider.set_val(self.num_total - 1)
        elif event.key == "tab":
            info = self.frame_info[self.current_idx]
            if info['type'] in ('SINGLE-STEP PRED', 'VAL TARGET'):
                self.show_target[self.current_idx] = not self.show_target.get(
                    self.current_idx, False
                )
                self._update_frame(self.current_idx)

    # ------------------------------------------------------------------
    def show(self):
        plt.show()


# =====================================
# 10. 主流程
# =====================================
def main():
    # ---- 10.1 加载模型 ----
    model, mean, std, grid_x, grid_y, input_steps = load_trained_model(
        MODEL_SAVE_PATH, device
    )

    # ---- 10.2 加载数据 ----
    flow_scaled, mean_d, std_d, _, _ = prepare_prediction_data(FILE_PATHS, GRID_SIZE)
    if abs(mean - mean_d) > 1e-6 or abs(std - std_d) > 1e-6:
        print("\nWarning: checkpoint norm differs from current data. Using checkpoint.")

    T = len(flow_scaled)
    all_true_frames = denormalize(flow_scaled[:, 0, :, :], mean, std)
    true_labels = [os.path.basename(fp) for fp in FILE_PATHS]

    # ---- 10.3 单步预测 ----
    print(f"\nSingle-step predictions on {T - input_steps} samples (teacher-forcing) ...")
    val_pred_scaled, val_target_scaled = single_step_predict(
        model=model, data_scaled=flow_scaled,
        input_steps=input_steps, device=device,
    )
    val_pred = denormalize(val_pred_scaled, mean, std)
    val_target = denormalize(val_target_scaled, mean, std)
    val_mae = np.abs(val_pred - val_target).mean(axis=(1, 2))

    print(f"Single-step MAE: mean={val_mae.mean():.4f}  "
          f"best={val_mae.min():.4f}  worst={val_mae.max():.4f}")

    # ---- 10.4 自回归预测 ----
    num_auto = 50
    print(f"\nAutoregressive prediction: {num_auto} steps ...")
    seed_scaled = flow_scaled[-input_steps:]
    auto_pred_scaled = autoregressive_predict(
        model=model, seed_sequence=seed_scaled,
        num_steps=num_auto, device=device,
    )
    auto_pred = denormalize(auto_pred_scaled, mean, std)

    drift = auto_pred.mean(axis=(1, 2))[-1] - auto_pred.mean(axis=(1, 2))[0]
    print(f"Mean drift over {num_auto} steps: {drift:+.4f}")

    pred_labels = build_prediction_labels(FILE_PATHS, num_auto)

    # ---- 10.5 诊断面板 ----
    print("\n" + "=" * 60)
    print("Diagnostic Dashboard (close to continue)")
    print("=" * 60)
    plot_diagnostic_dashboard(
        true_all=all_true_frames, val_pred=val_pred, val_target=val_target,
        auto_pred=auto_pred, input_steps=input_steps, file_labels=true_labels,
    )

    # ---- 10.6 构建交叉帧序列 ----
    frames, frame_info = build_interleaved_frames(
        true_frames=all_true_frames,
        val_pred_frames=val_pred,
        val_target_frames=val_target,
        auto_pred_frames=auto_pred,
        true_labels=true_labels,
        pred_labels=pred_labels,
        input_steps=input_steps,
        val_mae=val_mae,
    )

    total = len(frames)
    n_auto = num_auto
    n_pairs = T - input_steps
    print(f"\nFrame layout ({total} total):")
    print(f"  0..{input_steps - 1}      : Seed window (actual, no pred)")
    print(f"  {input_steps}..{input_steps + 2 * n_pairs - 1}  : Interleaved (actual, pred) x {n_pairs}")
    print(f"  {input_steps + 2 * n_pairs}..{total - 1}  : Autoregressive x {n_auto}")
    print("\nControls: <- -> navigate  |  Tab = toggle pred/target  |  Home/End = jump")

    viewer = FlowFieldViewer(
        frames=frames, frame_info=frame_info,
        grid_x=grid_x, grid_y=grid_y,
        title="ConvLSTM Flow Field Prediction",
    )
    viewer.show()


if __name__ == "__main__":
    main()
