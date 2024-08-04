# Copyright © 2024 Gökdeniz Gülmez

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from kan import KANLinear
from kan.kan_conv.kan_convolution import KAN_Convolutional_Layer

class LlamaKANMLP(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_dim: int,
            out_features: int,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            hidden_act=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            ):
        super().__init__()
        self.w1 = KANLinear(
            in_features=in_features,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        self.w3 = KANLinear(
            in_features=in_features,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        self.w2 = KANLinear(
            in_features=hidden_dim,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=True,
            hidden_act=nn.Identity,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
    def __call__(self, x):
        return self.w2(self.w1(x) * self.w3(x))

class SmallKANMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        hidden_act=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.layer1 = KANLinear(
            in_features=in_features,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        self.layer2 = KANLinear(
            in_features=hidden_dim,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def __call__(self, x):
        return self.layer2(self.layer1(x))

class MiddleKANMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        hidden_act=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.layer1 = KANLinear(
            in_features=in_features,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        self.layer2 = KANLinear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        self.layer3 = KANLinear(
            in_features=hidden_dim,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def __call__(self, x):
        x = self.layer1(x) # (in_features -> hidden_dim)
        x = self.layer2(x) # (hidden_dim -> hidden_dim)
        x = self.layer3(x) # (hidden_dim -> out_features)
        return x

class BigKANMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        hidden_act=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.layer1 = KANLinear(
            in_features=in_features,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        self.layer2 = KANLinear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        self.layer3 = KANLinear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        self.layer4 = KANLinear(
            in_features=hidden_dim,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def __call__(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))



class KANC_MLP(nn.Module):
    def __init__(
            self,
            in_features: int = 625,
            hidden_dim: int = 256,
            out_features: int = 10,
            n_convs: int = 5,
            kernel_size: Tuple[int, int] = (3, 3)
        ):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(n_convs=n_convs, kernel_size=kernel_size)
        self.conv2 = KAN_Convolutional_Layer(n_convs=5, kernel_size=kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def __call__(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool1(self.conv2(x))
        x = mx.flatten(x)
        x = nn.ReLU(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, axis=1)