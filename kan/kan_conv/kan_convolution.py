# Copyright © 2024 Gökdeniz Gülmez

import math

import mlx.core as mx
import mlx.nn as nn

from ..kan import KANLinear
from .convolution import kan_conv2d, multiple_convs_kan_conv2d

# Define a KAN convolution object
class KAN_Convolution(nn.Module):
    def __init__(
            self,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            hidden_act=nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1]
        ):
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize KANLinear as convolution operation
        self.conv = KANLinear(
            in_features = math.prod(kernel_size),
            out_features = 1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def __call__(self, x: mx.array, update_grid=False):
        return kan_conv2d(x, self.conv, self.kernel_size[0], self.stride, self.dilation, self.padding)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum( layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)

# Define a KAN convolutional layer with multiple convolutions
class KAN_Convolutional_Layer(nn.Module):
    def __init__(
            self,
            n_convs: int = 1,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order:int = 3,
            scale_noise:float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            hidden_act=nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1]
        ):
        """
        Kan Convolutional Layer with multiple convolutions
        
        Args:
            n_convs (int): Number of convolutions to apply
            kernel_size (tuple): Size of the kernel
            stride (tuple): Stride of the convolution
            padding (tuple): Padding of the convolution
            dilation (tuple): Dilation of the convolution
            grid_size (int): Size of the grid
            spline_order (int): Order of the spline
            scale_noise (float): Scale of the noise
            scale_base (float): Scale of the base
            scale_spline (float): Scale of the spline
            hidden_act (torch.nn.Module): Activation function
            grid_eps (float): Epsilon of the grid
            grid_range (tuple): Range of the grid
        """


        super(KAN_Convolutional_Layer, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.convs = []
        self.n_convs = n_convs
        self.stride = stride


        # Create n_convs KAN_Convolution objects
        for _ in range(n_convs):
            self.convs.append(
                KAN_Convolution(
                    kernel_size= kernel_size,
                    stride = stride,
                    padding=padding,
                    dilation = dilation,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    hidden_act=hidden_act,
                    grid_eps=grid_eps,
                    grid_range=grid_range
                )
            )

    def __call__(self, x: mx.array, update_grid=False):
        # If there are multiple convolutions, apply them all
        if self.n_convs>1:
            return multiple_convs_kan_conv2d(x, self.convs, self.kernel_size[0], self.stride, self.dilation, self.padding)
        
        # If there is only one convolution, apply it
        return self.convs[0].forward(x)