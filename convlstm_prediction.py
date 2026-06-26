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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
# 7. 构建时间轴帧数据（三通道）
# =====================================
def build_timeline_data(
    true_frames: np.ndarray,        # (T, H, W)
    val_pred_frames: np.ndarray,    # (V, H, W)
    val_target_frames: np.ndarray,  # (V, H, W)
    auto_pred_frames: np.ndarray,   # (P, H, W)
    true_labels: list,              # [str] * T
    pred_labels: list,              # [str] * P
    input_steps: int,
) -> tuple:
    """
    按时间索引构建三通道帧数据。

    Returns
    -------
    real_frames  : list of (H,W) or None   — None = 无真实值（自回归段）
    pred_frames  : list of (H,W) or None   — None = 无预测（种子段）
    error_frames : list of (H,W) or None
    labels       : list of str
    frame_types  : list of str  — 'Seed' | 'Actual+Pred' | 'AutoReg'
    """
    T = len(true_frames)
    P = len(auto_pred_frames)

    real_frames = []
    pred_frames = []
    error_frames = []
    labels = []
    frame_types = []

    # ---- t = 0 .. T-1: 真实数据段 ----
    for t in range(T):
        real_frames.append(true_frames[t])
        labels.append(true_labels[t])

        if t < input_steps:
            # 种子窗口：仅有真实帧，无预测
            pred_frames.append(None)
            error_frames.append(None)
            frame_types.append('Seed')
        else:
            vi = t - input_steps
            pred_frames.append(val_pred_frames[vi])
            error_frames.append(np.abs(val_pred_frames[vi] - val_target_frames[vi]))
            frame_types.append('Actual+Pred')

    # ---- t = T .. T+P-1: 自回归预测段 ----
    for ai in range(P):
        real_frames.append(None)            # 无真实值
        pred_frames.append(auto_pred_frames[ai])
        error_frames.append(None)           # 无误差（无真实值可对比）
        labels.append(pred_labels[ai])
        frame_types.append('AutoReg')

    return real_frames, pred_frames, error_frames, labels, frame_types


# =====================================
# 8. 三面板流场浏览器
# =====================================
class TriplePanelViewer:
    """
    三面板流场浏览器 —— 真实帧 | 预测帧 | 误差场，共用一个时间轴。
    自回归阶段（无真实值）时，真实帧和误差场冻结在最后可用值。
    """

    def __init__(
        self,
        real_frames: list,       # [(H,W) or None, ...]
        pred_frames: list,       # [(H,W) or None, ...]
        error_frames: list,      # [(H,W) or None, ...]
        labels: list,            # [str, ...]
        frame_types: list,       # ['Seed' | 'Actual+Pred' | 'AutoReg', ...]
        grid_x: np.ndarray,
        grid_y: np.ndarray,
    ):
        self.real_frames = real_frames
        self.pred_frames = pred_frames
        self.error_frames = error_frames
        self.labels = labels
        self.frame_types = frame_types
        self.num_total = len(labels)
        self.grid_x = grid_x
        self.grid_y = grid_y

        # ---- 共享颜色轴范围：基于真实帧+单步预测帧（排除自回归段） ----
        all_valid = []
        for i, f in enumerate(real_frames):
            if f is not None and self.frame_types[i] != 'AutoReg':
                all_valid.append(f.flatten())
        for i, f in enumerate(pred_frames):
            if f is not None and self.frame_types[i] != 'AutoReg':
                all_valid.append(f.flatten())
        if all_valid:
            all_cat = np.concatenate(all_valid)
            self.vmin_shared = float(all_cat.min())
            self.vmax_shared = float(all_cat.max())
        else:
            self.vmin_shared, self.vmax_shared = 0.0, 1.0

        # ---- 误差颜色轴范围 ----
        err_all = []
        for f in error_frames:
            if f is not None:
                err_all.append(f.flatten())
        if err_all:
            err_cat = np.concatenate(err_all)
            self.vmin_err = float(err_cat.min())
            self.vmax_err = float(err_cat.max())
        else:
            self.vmin_err, self.vmax_err = 0.0, 1.0

        # ---- 记录最后可用的真实帧和误差帧（自回归段冻结用） ----
        self._last_real = None
        self._last_error = None
        for f in real_frames:
            if f is not None:
                self._last_real = f.copy()
        for f in error_frames:
            if f is not None:
                self._last_error = f.copy()

        # ---- 创建图形 ----
        self.fig = plt.figure(figsize=(20, 12), dpi=120)
        self.current_idx = 0
        self._setup_ui()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._update_frame(0)

    # ------------------------------------------------------------------
    def _setup_ui(self):
        """三面板 + 颜色条 + 时间轴 + 滑块布局。"""
        plot_l, plot_w = 0.06, 0.73
        plot_h = 0.26
        cbar_l, cbar_w = 0.81, 0.015

        # 三个主面板
        self.ax_real = self.fig.add_axes([plot_l, 0.64, plot_w, plot_h])
        self.ax_real.set_aspect("equal")

        self.ax_pred = self.fig.add_axes([plot_l, 0.36, plot_w, plot_h])
        self.ax_pred.set_aspect("equal")

        self.ax_error = self.fig.add_axes([plot_l, 0.08, plot_w, plot_h])
        self.ax_error.set_aspect("equal")

        # 共享颜色条（Real + Pred）
        self.ax_cbar = self.fig.add_axes([cbar_l, 0.36, cbar_w, 0.54])

        # 误差颜色条
        self.ax_cbar_err = self.fig.add_axes([cbar_l, 0.08, cbar_w, 0.26])

        # 时间轴颜色条
        self.ax_timeline = self.fig.add_axes([0.06, 0.045, 0.88, 0.018])
        self._draw_timeline()
        self.timeline_line = self.ax_timeline.axvline(
            x=0.5, color='white', linewidth=2.5, linestyle='-', zorder=5
        )

        # 滑块
        self.ax_slider = self.fig.add_axes([0.06, 0.01, 0.88, 0.025])
        self.slider = Slider(
            ax=self.ax_slider, label="Time Step",
            valmin=0, valmax=max(self.num_total - 1, 0),
            valinit=0, valfmt="%d", valstep=1,
        )
        self.slider.on_changed(self._on_slider)

        # 信息栏
        self.ax_info = self.fig.add_axes([0.06, 0.925, 0.88, 0.035])
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(
            0.5, 0.5, "",
            transform=self.ax_info.transAxes,
            ha="center", va="center", fontsize=9, fontweight="bold",
        )

    # ------------------------------------------------------------------
    def _update_frame(self, idx: int):
        """刷新三个面板。"""
        idx = int(np.clip(idx, 0, self.num_total - 1))
        self.current_idx = idx

        real = self.real_frames[idx]
        pred = self.pred_frames[idx]
        error = self.error_frames[idx]
        label = self.labels[idx]
        ftype = self.frame_types[idx]

        # 自回归段冻结处理
        real_frozen = (real is None)
        if real is None:
            real = self._last_real
        if error is None and pred is not None and real is not None:
            # 自回归段有预测但无真实值：用冻结的 real 计算误差
            error = np.abs(pred - real)
        elif error is None:
            error = self._last_error

        X2D, Y2D = np.meshgrid(self.grid_x, self.grid_y)

        # ---- Panel 1: Real ----
        self.ax_real.clear()
        if real is not None:
            self.ax_real.pcolormesh(
                X2D, Y2D, real, cmap="jet",
                shading="gouraud", rasterized=True,
                vmin=self.vmin_shared, vmax=self.vmax_shared,
            )
        self.ax_real.set_aspect("equal")
        self.ax_real.set_xlabel("X")
        self.ax_real.set_ylabel("Y")
        freeze_str = "  [FROZEN — no ground truth]" if real_frozen else ""
        self.ax_real.set_title(
            f"1. Real Velocity Field{freeze_str}",
            fontsize=13, fontweight="bold", color="darkgreen",
        )

        # ---- Panel 2: Pred ----
        self.ax_pred.clear()
        if pred is not None:
            self.ax_pred.pcolormesh(
                X2D, Y2D, pred, cmap="jet",
                shading="gouraud", rasterized=True,
                vmin=self.vmin_shared, vmax=self.vmax_shared,
            )
        self.ax_pred.set_aspect("equal")
        self.ax_pred.set_xlabel("X")
        self.ax_pred.set_ylabel("Y")
        if ftype == 'AutoReg':
            pred_title = "2. Autoregressive Predicted Velocity Field"
            pred_color = "darkred"
        elif ftype == 'Seed':
            pred_title = "2. Predicted Velocity Field (no prediction yet)"
            pred_color = "gray"
        else:
            pred_title = "2. Predicted Velocity Field (single-step)"
            pred_color = "darkorange"
        self.ax_pred.set_title(pred_title, fontsize=13, fontweight="bold", color=pred_color)

        # ---- Panel 3: Error ----
        self.ax_error.clear()
        if error is not None:
            self.ax_error.pcolormesh(
                X2D, Y2D, error, cmap="hot",
                shading="gouraud", rasterized=True,
                vmin=self.vmin_err, vmax=self.vmax_err,
            )
        self.ax_error.set_aspect("equal")
        self.ax_error.set_xlabel("X")
        self.ax_error.set_ylabel("Y")
        err_frozen = (self.error_frames[idx] is None and ftype == 'AutoReg')
        err_str = "  [FROZEN — no ground truth]" if err_frozen else ""
        mae_val = float(np.mean(error)) if error is not None else float('nan')
        self.ax_error.set_title(
            f"3. |Prediction − Real|  (mean = {mae_val:.4f}){err_str}",
            fontsize=13, fontweight="bold", color="darkred",
        )

        # ---- 共享颜色条 (Real + Pred) ----
        self.ax_cbar.clear()
        sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=self.vmin_shared, vmax=self.vmax_shared),
            cmap="jet",
        )
        sm.set_array([])
        self.fig.colorbar(sm, cax=self.ax_cbar, label="Velocity")

        # ---- 误差颜色条 ----
        self.ax_cbar_err.clear()
        sm_err = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=self.vmin_err, vmax=self.vmax_err),
            cmap="hot",
        )
        sm_err.set_array([])
        self.fig.colorbar(sm_err, cax=self.ax_cbar_err, label="|Error|")

        # ---- 信息栏 ----
        real_min = f"{real.min():.4f}" if real is not None else "N/A"
        real_max = f"{real.max():.4f}" if real is not None else "N/A"
        pred_min = f"{pred.min():.4f}" if pred is not None else "N/A"
        pred_max = f"{pred.max():.4f}" if pred is not None else "N/A"
        self.info_text.set_text(
            f"Frame {idx}/{self.num_total - 1}  |  {ftype}  |  {label}  |  "
            f"Real: [{real_min}, {real_max}]  |  Pred: [{pred_min}, {pred_max}]  |  "
            f"MAE: {mae_val:.4f}"
        )

        self.fig.canvas.draw_idle()

        # 更新时间轴位置标记
        self.timeline_line.set_xdata([idx + 0.5, idx + 0.5])

    # ------------------------------------------------------------------
    def _draw_timeline(self):
        """在滑块上方绘制颜色编码的时间轴条。"""
        color_map = {
            'Seed':        np.array([0.0, 0.0, 0.545]),    # darkblue
            'Actual+Pred': np.array([0.0, 0.392, 0.0]),    # darkgreen
            'AutoReg':     np.array([0.545, 0.0, 0.0]),    # darkred
        }
        img = np.zeros((1, self.num_total, 3))
        for i, ft in enumerate(self.frame_types):
            img[0, i] = color_map.get(ft, [0.5, 0.5, 0.5])
        self.ax_timeline.imshow(img, aspect='auto', interpolation='nearest')
        self.ax_timeline.set_yticks([])

        # 分段文字标签
        short_label = {
            'Seed': 'Seed', 'Actual+Pred': 'Real + 1-Step Pred', 'AutoReg': 'AutoReg',
        }
        prev_type = None
        seg_start = 0
        for i, ft in enumerate(self.frame_types):
            if ft != prev_type:
                if prev_type is not None:
                    mid = (seg_start + i) / 2
                    self.ax_timeline.text(
                        mid, 0, short_label.get(prev_type, '?'),
                        ha='center', va='center', fontsize=6,
                        color='white', fontweight='bold',
                    )
                seg_start = i
                prev_type = ft
        if prev_type is not None:
            mid = (seg_start + self.num_total) / 2
            self.ax_timeline.text(
                mid, 0, short_label.get(prev_type, '?'),
                ha='center', va='center', fontsize=6,
                color='white', fontweight='bold',
            )

        # 段分隔线
        prev_type2 = None
        for i, ft in enumerate(self.frame_types):
            if ft != prev_type2:
                if i > 0:
                    self.ax_timeline.axvline(x=i, color='white', linewidth=1, alpha=0.6)
                prev_type2 = ft

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

    # ------------------------------------------------------------------
    def show(self):
        plt.show()


# =====================================
# 9. 主流程
# =====================================
def main():
    # ---- 9.1 加载模型 ----
    model, mean, std, grid_x, grid_y, input_steps = load_trained_model(
        MODEL_SAVE_PATH, device
    )

    # ---- 9.2 加载数据 ----
    flow_scaled, mean_d, std_d, _, _ = prepare_prediction_data(FILE_PATHS, GRID_SIZE)
    if abs(mean - mean_d) > 1e-6 or abs(std - std_d) > 1e-6:
        print("\nWarning: checkpoint norm differs from current data. Using checkpoint.")

    T = len(flow_scaled)
    all_true_frames = denormalize(flow_scaled[:, 0, :, :], mean, std)
    true_labels = [os.path.basename(fp) for fp in FILE_PATHS]

    # ---- 9.3 单步预测 ----
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

    # ---- 9.4 自回归预测 ----
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

    # ---- 9.5 构建时间轴帧数据 ----
    real_frames, pred_frames, error_frames, labels, frame_types = build_timeline_data(
        true_frames=all_true_frames,
        val_pred_frames=val_pred,
        val_target_frames=val_target,
        auto_pred_frames=auto_pred,
        true_labels=true_labels,
        pred_labels=pred_labels,
        input_steps=input_steps,
    )

    n_seed = input_steps
    n_pred = T - input_steps
    total = len(labels)
    print(f"\nFrame layout ({total} total):")
    print(f"  0..{n_seed - 1}     : Seed window (real only, no prediction)")
    print(f"  {n_seed}..{n_seed + n_pred - 1} : Real + single-step prediction + error")
    print(f"  {n_seed + n_pred}..{total - 1} : Autoregressive prediction only")
    print("\nControls:  <- -> navigate  |  Home/End jump")

    viewer = TriplePanelViewer(
        real_frames=real_frames,
        pred_frames=pred_frames,
        error_frames=error_frames,
        labels=labels,
        frame_types=frame_types,
        grid_x=grid_x,
        grid_y=grid_y,
    )
    viewer.show()


if __name__ == "__main__":
    main()
