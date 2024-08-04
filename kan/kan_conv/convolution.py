# Copyright © 2024 Gökdeniz Gülmez

import numpy as np
from typing import Union, List, Tuple

import mlx.core as mx

def extract_patches(arr, patch_size):
    """ Extract patches of size patch_size from arr. """
    arr_shape = arr.shape
    patch_shape = (arr_shape[0] - patch_size[0] + 1, arr_shape[1] - patch_size[1] + 1, patch_size[0], patch_size[1])
    
    strides = arr.strides + (arr.strides[0], arr.strides[1])
    
    return np.lib.stride_tricks.as_strided(arr, shape=patch_shape, strides=strides)

# Calculate output dimensions for convolution
def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    # Ensure the matrix has 4 dimensions
    if len(matrix.shape) != 4:
        raise ValueError("Input matrix must have 4 dimensions (batch_size, n_channels, height, width)")

    batch_size, n_channels, n, m = matrix.shape
    h_out = np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    b = [kernel_side // 2, kernel_side // 2]
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
        kernels (List[function]): List of 2D kernels to apply.
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """

    def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
        batch_size, n_channels, height, width = matrix.shape
        padded_height = height + 2 * padding[0]
        padded_width = width + 2 * padding[1]
        kernel_height = kernel_side * dilation[0] - (dilation[0] - 1)
        kernel_width = kernel_side * dilation[1] - (dilation[1] - 1)
        h_out = (padded_height - kernel_height) // stride[0] + 1
        w_out = (padded_width - kernel_width) // stride[1] + 1
        return h_out, w_out, batch_size, n_channels

    # Compute output dimensions
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)

    # Initialize output tensor
    matrix_out = np.zeros((batch_size, n_channels * n_convs, h_out, w_out))

    # Pad the input matrix
    matrix_padded = np.pad(matrix, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')

    # Apply convolutional kernels
    for batch in range(batch_size):
        for channel in range(n_channels):
            for kern_idx, kernel in enumerate(kernels):
                for i in range(0, h_out):
                    for j in range(0, w_out):
                        # Calculate window start and end indices
                        row_start = i * stride[0]
                        row_end = row_start + kernel_side * dilation[0]
                        col_start = j * stride[1]
                        col_end = col_start + kernel_side * dilation[1]

                        # Extract the patch
                        patch = matrix_padded[batch, channel, row_start:row_end:dilation[0], col_start:col_end:dilation[1]]

                        # Apply kernel and store the result
                        conv_result = kernel(patch)
                        matrix_out[batch, channel * n_convs + kern_idx, i, j] = conv_result

    return matrix_out

# Perform a single KAN convolution
def kan_conv2d(
    matrix: Union[List[List[float]], np.ndarray],
    kernel, 
    kernel_side,
    stride= (1, 1), 
    dilation= (1, 1), 
    padding= (0, 0)
) -> np.ndarray:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel (function): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """

    def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
        batch_size, n_channels, height, width = matrix.shape
        padded_height = height + 2 * padding[0]
        padded_width = width + 2 * padding[1]
        kernel_height = kernel_side * dilation[0] - (dilation[0] - 1)
        kernel_width = kernel_side * dilation[1] - (dilation[1] - 1)
        h_out = (padded_height - kernel_height) // stride[0] + 1
        w_out = (padded_width - kernel_width) // stride[1] + 1
        return h_out, w_out, batch_size, n_channels

    # Compute output dimensions
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)

    # Initialize output tensor
    matrix_out = np.zeros((batch_size, n_channels, h_out, w_out))

    # Pad the input matrix
    matrix_padded = np.pad(matrix, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')

    # Apply convolutional kernel
    for batch in range(batch_size):
        for channel in range(n_channels):
            for i in range(0, h_out):
                for j in range(0, w_out):
                    # Calculate window start and end indices
                    row_start = i * stride[0]
                    row_end = row_start + kernel_side * dilation[0]
                    col_start = j * stride[1]
                    col_end = col_start + kernel_side * dilation[1]

                    # Extract the patch
                    patch = matrix_padded[batch, channel, row_start:row_end:dilation[0], col_start:col_end:dilation[1]]

                    # Apply kernel and store the result
                    conv_result = kernel(patch)
                    matrix_out[batch, channel, i, j] = conv_result

    return matrix_out