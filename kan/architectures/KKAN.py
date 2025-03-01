# Copyright © 2024 Gökdeniz Gülmez

import mlx.core as mx
import mlx.nn as nn

from ..kan import KANLinear
from .conv import KAN_Convolutional_Layer

class KKAN_Convolutional_Network(nn.Module):
    def __init__(
        self,
        in_features=625,
        out_features=10,
        grid_size=10,
        spline_order=3,
        scale_noise=0.01,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        hidden_act=nn.SiLU,
        grid_eps=0.02,
        grid_range=[0,1],
    ):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size= (3,3)
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size = (3,3)
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )

        self.kan1 = KANLinear(
            in_features=in_features,
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
        x = self.pool1(self.conv1(x))
        x = mx.flatten(self.pool1(self.conv2(x)))
        x = self.kan1(x)
        return nn.log_softmax(x, axis=1)