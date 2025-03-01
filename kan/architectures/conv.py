# Copyright © 2024 Gökdeniz Gülmez

import math
from typing import Tuple, List, Union

import mlx.core as mx
import mlx.nn as nn

from ..kan import KANLinear

class KANConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        hidden_act = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1]
    ):
        super().__init__()
        
        # Convert scalar parameters to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Calculate the number of input features for the KANLinear layer
        in_features = kernel_size[0] * kernel_size[1] * in_channels // groups
        
        # Create a KANLinear layer for each output channel
        self.kan_layers = [
            KANLinear(
                in_features=in_features,
                out_features=1,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                hidden_act=hidden_act,
                grid_eps=grid_eps,
                grid_range=grid_range
            ) for _ in range(out_channels)
        ]
        
    def _unfold(self, x):
        """Extract sliding local blocks from input tensor"""
        batch_size, channels, height, width = x.shape
        
        # Use mx.pad instead of nn.pad
        padded_x = mx.pad(
            x, 
            ((0, 0), (0, 0), 
             (self.padding[0], self.padding[0]), 
             (self.padding[1], self.padding[1]))
        )
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        # Extract patches
        patches = []
        for i in range(0, height + 2 * self.padding[0] - self.kernel_size[0] + 1, self.stride[0]):
            for j in range(0, width + 2 * self.padding[1] - self.kernel_size[1] + 1, self.stride[1]):
                patch = padded_x[:, :, i:i+self.kernel_size[0], j:j+self.kernel_size[1]]
                patches.append(patch)
        
        # Stack and reshape patches
        if patches:
            patches = mx.stack(patches, axis=-1)
            patches = patches.reshape(batch_size, channels, self.kernel_size[0] * self.kernel_size[1], -1)
            patches = patches.transpose(0, 3, 1, 2)
            patches = patches.reshape(batch_size * out_height * out_width, -1)
        else:
            # Handle case where no patches were extracted
            patches = mx.zeros((batch_size * out_height * out_width, 
                               channels * self.kernel_size[0] * self.kernel_size[1]))
        
        return patches, (batch_size, out_height, out_width)
        
    def __call__(self, x, update_grid=False):
        # Handle input shape - ensure it's 4D (batch, channels, height, width)
        if len(x.shape) == 2:
            # If input is (batch, features), reshape to (batch, channels, 1, features/channels)
            batch_size, features = x.shape
            if features % self.in_channels != 0:
                raise ValueError(f"Input features ({features}) must be divisible by in_channels ({self.in_channels})")
            
            width = features // self.in_channels
            x = x.reshape(batch_size, self.in_channels, 1, width)
        elif len(x.shape) == 3:
            # If input is (batch, height, width), add channel dimension
            batch_size, height, width = x.shape
            x = x.reshape(batch_size, 1, height, width)
        elif len(x.shape) != 4:
            raise ValueError(f"Expected input with 2, 3, or 4 dimensions, got {len(x.shape)}")
        
        batch_size, channels, height, width = x.shape
        
        # Check if channels match
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}")
        
        # Unfold input to patches
        patches, (batch_size, out_height, out_width) = self._unfold(x)
        
        # Process each output channel with its KAN layer
        outputs = []
        for i, kan_layer in enumerate(self.kan_layers):
            if update_grid:
                kan_layer.update_grid(patches)
            
            # Apply KAN layer to patches
            channel_output = kan_layer(patches)
            outputs.append(channel_output)
        
        # Stack outputs and reshape to proper output format
        output = mx.stack(outputs, axis=1)
        output = output.reshape(batch_size, self.out_channels, out_height, out_width)
        
        return output
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) 
                  for layer in self.kan_layers)


class KANConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        n_convs: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        hidden_act = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
        use_residual: bool = False
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1 or stride == (1, 1))
        
        # Create a sequence of convolutional layers
        self.convs = nn.ModuleList()
        
        # First conv layer (input_channels -> out_channels)
        self.convs.append(
            KANConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
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
        
        # Additional conv layers (out_channels -> out_channels)
        for _ in range(1, n_convs):
            self.convs.append(
                KANConv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,  # Use stride 1 for intermediate layers
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
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
        
        # Activation function
        self.activation = hidden_act()
        
    def __call__(self, x, update_grid=False):
        identity = x
        
        for i, conv in enumerate(self.convs):
            x = conv(x, update_grid=update_grid)
            if i < len(self.convs) - 1 or not self.use_residual:
                x = self.activation(x)
        
        # Apply residual connection if enabled
        if self.use_residual:
            x = x + identity
            x = self.activation(x)
            
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(conv.regularization_loss(regularize_activation, regularize_entropy) 
                  for conv in self.convs)


# Simple implementation for backward compatibility with original code
class KAN_Convolution(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = (2, 2),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int]] = (0, 0),
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        hidden_act = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1]
    ):
        super().__init__()
        
        # Convert to tuples if needed
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create a KANLinear layer for the convolution
        self.conv = KANLinear(
            in_features=math.prod(kernel_size),
            out_features=1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            hidden_act=hidden_act,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        
    def _extract_patches(self, x):
        """Simple patch extraction for 1-channel input"""
        # Ensure input is 4D (batch, channels, height, width)
        if len(x.shape) == 2:
            batch_size, features = x.shape
            x = x.reshape(batch_size, 1, 1, features)
        elif len(x.shape) == 3:
            batch_size, height, width = x.shape
            x = x.reshape(batch_size, 1, height, width)
            
        batch_size, channels, height, width = x.shape
        
        # Apply padding using mx.pad
        padded_x = mx.pad(
            x, 
            ((0, 0), (0, 0), 
             (self.padding[0], self.padding[0]), 
             (self.padding[1], self.padding[1]))
        )
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        # Extract patches
        patches = []
        for i in range(0, height + 2 * self.padding[0] - self.kernel_size[0] + 1, self.stride[0]):
            for j in range(0, width + 2 * self.padding[1] - self.kernel_size[1] + 1, self.stride[1]):
                patch = padded_x[:, :, i:i+self.kernel_size[0], j:j+self.kernel_size[1]]
                patches.append(patch)
        
        # Stack patches
        if patches:
            patches = mx.stack(patches, axis=1)
            patches = patches.reshape(batch_size, -1, self.kernel_size[0] * self.kernel_size[1])
        else:
            # Handle case where no patches were extracted
            patches = mx.zeros((batch_size, 0, self.kernel_size[0] * self.kernel_size[1]))
            
        return patches, (batch_size, out_height, out_width)
        
    def __call__(self, x, update_grid=False):
        # Extract patches from input
        patches, (batch_size, out_height, out_width) = self._extract_patches(x)
        
        # Reshape patches for KANLinear
        patches_flat = patches.reshape(-1, self.kernel_size[0] * self.kernel_size[1])
        
        # Update grid if needed
        if update_grid:
            self.conv.update_grid(patches_flat)
            
        # Apply KANLinear to patches
        output = self.conv(patches_flat)
        
        # Reshape output to proper dimensions
        output = output.reshape(batch_size, out_height, out_width)
        
        return output
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.conv.regularization_loss(regularize_activation, regularize_entropy)


# For backward compatibility
class KAN_Convolutional_Layer(nn.Module):
    def __init__(
        self,
        n_convs: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = (2, 2),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int]] = (0, 0),
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        hidden_act = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1]
    ):
        super().__init__()
        
        self.n_convs = n_convs
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create multiple KAN_Convolution layers
        self.convs = nn.ModuleList([
            KAN_Convolution(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                hidden_act=hidden_act,
                grid_eps=grid_eps,
                grid_range=grid_range
            ) for _ in range(n_convs)
        ])
        
    def __call__(self, x, update_grid=False):
        # Apply each convolution and sum the results
        if self.n_convs > 1:
            outputs = []
            for conv in self.convs:
                outputs.append(conv(x, update_grid=update_grid))
            return sum(outputs)  # Use sum instead of mx.add for multiple items
        
        # If only one convolution, just apply it
        return self.convs[0](x, update_grid=update_grid)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(conv.regularization_loss(regularize_activation, regularize_entropy) 
                  for conv in self.convs)


# Helper function to handle different input shapes for KAN convolutions
def multiple_convs_kan_conv2d(x, convs, kernel_size, stride, dilation, padding):
    """Apply multiple KAN convolutions and sum the results"""
    outputs = []
    for conv in convs:
        outputs.append(conv(x))
    return sum(outputs)  # Use sum instead of mx.add for multiple items


# Helper function for kan_conv2d to maintain backward compatibility
def kan_conv2d(x, conv, kernel_size, stride, dilation, padding):
    """Apply a KAN convolution to input x"""
    # Create a KAN_Convolution instance if one isn't provided
    if not isinstance(conv, KAN_Convolution):
        temp_conv = KAN_Convolution(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        temp_conv.conv = conv  # Use the provided KANLinear
        return temp_conv(x)
    
    # Otherwise just call the provided convolution
    return conv(x)