import torch
import torch.nn as nn


# ==========================================
# 残差块（Refinement Head 使用）
# ==========================================
class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.relu(
            self.bn1(self.conv1(x))
        )

        out = self.bn2(self.conv2(out))

        return self.relu(out + residual)


# ==========================================
# ConvLSTM Cell (单层)
# ==========================================
class ConvLSTMCell(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim,
            kernel_size
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, h_cur, c_cur):

        combined = torch.cat([x, h_cur], dim=1)

        conv_output = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(
            conv_output,
            self.hidden_dim,
            dim=1
        )

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# ==========================================
# 多层 ConvLSTM + Refinement Head
# ==========================================
class ConvLSTM(nn.Module):

    def __init__(
            self,
            input_dim=1,
            hidden_dims=None,
            kernel_size=3,
            output_dim=1,
            num_refine_blocks=2
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128]

        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # ---- 构建多层 ConvLSTM Cell ----
        self.cells = nn.ModuleList()

        for i, hd in enumerate(hidden_dims):

            in_dim = input_dim if i == 0 else hidden_dims[i - 1]

            self.cells.append(
                ConvLSTMCell(
                    input_dim=in_dim,
                    hidden_dim=hd,
                    kernel_size=kernel_size
                )
            )

        # ---- Refinement Head ----
        last_hidden = hidden_dims[-1]
        refine_ch = min(last_hidden, 128)

        self.refine_in = nn.Sequential(
            nn.Conv2d(last_hidden, refine_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(refine_ch),
            nn.ReLU(inplace=True)
        )

        self.refine_blocks = nn.Sequential(
            *[ResBlock(refine_ch) for _ in range(num_refine_blocks)]
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(refine_ch, refine_ch // 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(refine_ch // 2, output_dim,
                      kernel_size=1)
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        return: (B, C, H, W)   [output_dim 通道]
        """

        B, T, C, H, W = x.shape

        # ---- 初始化多层隐状态 ----
        h_states = []
        c_states = []

        for hd in self.hidden_dims:

            h_states.append(
                torch.zeros(B, hd, H, W, device=x.device)
            )

            c_states.append(
                torch.zeros(B, hd, H, W, device=x.device)
            )

        # ---- 逐时间步推进 ----
        for t in range(T):

            layer_input = x[:, t]  # (B, C, H, W)

            for i, cell in enumerate(self.cells):

                h_states[i], c_states[i] = cell(
                    layer_input,
                    h_states[i],
                    c_states[i]
                )

                layer_input = h_states[i]  # 下一层输入

        # 最后一层隐状态
        final_h = h_states[-1]  # (B, last_hidden, H, W)

        # ---- Refinement ----
        out = self.refine_in(final_h)
        out = self.refine_blocks(out)
        out = self.output_layer(out)

        return out


if __name__ == "__main__":

    model = ConvLSTM(
        input_dim=1,
        hidden_dims=[64, 128],
        kernel_size=3,
        output_dim=1
    )

    x = torch.randn(2, 4, 1, 64, 64)
    y = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", y.shape)

    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}")
