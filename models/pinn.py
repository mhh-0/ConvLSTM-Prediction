import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        ])

        self.activation = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)