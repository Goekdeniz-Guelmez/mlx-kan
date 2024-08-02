# Copyright © 2024 Gökdeniz Gülmez

import numpy as np
from typing import Union, List

import mlx.core as mx
import mlx.nn as nn
import torch

def extract_patches(arr, patch_size):
    """ Extract patches of size patch_size from arr. """
    arr_shape = arr.shape
    patch_shape = (arr_shape[0] - patch_size[0] + 1, arr_shape[1] - patch_size[1] + 1, patch_size[0], patch_size[1])
    
    strides = arr.strides + (arr.strides[0], arr.strides[1])
    
    return np.lib.stride_tricks.as_strided(arr, shape=patch_shape, strides=strides)

# Calculate output dimensions for convolution
def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size, n_channels, n, m = matrix.shape
    h_out = np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    b = [kernel_side // 2, kernel_side// 2]
    return h_out, w_out, batch_size, n_channels

# Perform multiple convolutions using KAN
def multiple_convs_kan_conv2d(
    matrix,
    kernels, 
    kernel_side,
    stride= (1, 1), 
    dilation= (1, 1), 
    padding= (0, 0)
) -> mx.array:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    # Compute output dimensions
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)
    
    # Initialize output tensor
    matrix_out = mx.zeros((batch_size, n_channels * n_convs, h_out, w_out))
    
    # Unfold the input tensor using PyTorch
    unfold = torch.nn.Unfold(kernel_size=(kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride)
    unfolded = unfold(torch.Tensor(matrix))
    
    # Convert unfolded tensor to NumPy array for processing in MXNet
    unfolded_np = unfolded.numpy()
    unfolded_mx = mx.array(unfolded_np)
    
    # Reshape and transpose unfolded tensor
    unfolded_mx = unfolded_mx.view(batch_size, n_channels, kernel_side * kernel_side, h_out * w_out).transpose((0, 2, 1, 3))
    
    # Apply convolutional kernels
    for channel in range(n_channels):
        for kern in range(n_convs):
            conv_result = kernels[kern](unfolded_mx[:, channel, :, :].flatten(0, 1))
            matrix_out[:, kern + channel * n_convs, :, :] = mx.rray(conv_result.reshape(batch_size, h_out, w_out))
    
    return matrix_out

# Perform a single KAN convolution
def kan_conv2d(
    matrix: Union[List[List[float]], mx.array],
    kernel, 
    kernel_side,
    stride= (1, 1), 
    dilation= (1, 1), 
    padding= (0, 0)
) -> mx.array:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    
    matrix_out = mx.zeros((batch_size,n_channels,h_out,w_out))
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)

    for channel in range(n_channels):
        conv_groups = mx.array(unfold(torch.Tensor(matrix[:,channel,:,:]).unsqueeze(1)).transpose(1, 2))
        for k in range(batch_size):
            matrix_out[k,channel,:,:] = kernel.forward(conv_groups[k,:,:]).reshape((h_out,w_out))
    return matrix_out