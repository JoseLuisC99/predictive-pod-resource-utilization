import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple


class CasualConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, :-self.conv.padding[0]]


class TemporalGatedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.casual_conv = CasualConv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(x, self.casual_conv(x))


class STConv(nn.Module):
    def __init__(self, in_channels: int, spatial_channels: int, out_channels: int, kernel_size: int,
                 n_nodes: int, dropout_p: float = 0.5):
        super().__init__()
        self.temp_conv1 = TemporalGatedConv(in_channels, out_channels, kernel_size)
        self.graph_conv = GCNConv(out_channels, spatial_channels)
        self.temp_conv2 = TemporalGatedConv(spatial_channels, out_channels, kernel_size)
        self.layer_norm = nn.LayerNorm([n_nodes, out_channels])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        x = self.temp_conv1(x)
        x = self.graph_conv(x, graph)
        x = self.relu(x)
        x = self.temp_conv1(x)
        x = self.layer_norm(x)
        return self.dropout(x)
