import torch
from torch import nn
from torch.nn import functional as F

from utils import rand_partial_isometry


class Decoder(nn.Module):
    def __init__(self, layers: list[tuple[int]]):
        super().__init__()
        modules = []

        for idx, (input_dim, output_dim) in enumerate(layers):
            if idx < len(layers) - 1:
                modules.append(nn.Linear(input_dim, output_dim))
                modules.append(nn.Tanh())
            else:
                modules.append(nn.Linear(input_dim, output_dim * 2))
                modules.append(StandardActivation())

        self.input_dim = layers[0][0]
        self.output_dim = layers[-1][1]
        self.encoder = nn.Sequential(*modules)
        self.res_model = ResNet(self.input_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        mu, var = self.encoder(x)
        out_res, out_res_2 = self.res_model(x)
        return mu + out_res, var + out_res_2


class StandardActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        mean, log_var = torch.chunk(x, 2, dim=-1)
        return mean, F.softplus(log_var)


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        orthonormal_cols = rand_partial_isometry(input_dim, output_dim, 1.0)
        self.W = nn.Parameter(torch.from_numpy(orthonormal_cols).float())
        self.b1 = nn.Parameter(torch.zeros(output_dim))
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.a = torch.tensor(1.0).float()

    def forward(self, x):
        return torch.matmul(x, self.W) + self.b1, self.a * torch.log1p(torch.exp(self.b2))
