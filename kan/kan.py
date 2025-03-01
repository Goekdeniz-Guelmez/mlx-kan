# Copyright © 2024 Gökdeniz Gülmez

import os
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from kan.args import ModelArgs
from global_utils.utils import load_config

class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        hidden_act = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.hidden_act = hidden_act()
        self.grid_eps = grid_eps

        # Calculate grid points more efficiently
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid_points = mx.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.grid = mx.tile(grid_points.reshape(-1, 1), (1, in_features))

        # Initialize parameters
        self.base_weight = mx.zeros((out_features, in_features))
        self.spline_weight = mx.zeros((out_features, in_features, grid_size + spline_order))
        
        if enable_standalone_scale_spline:
            self.spline_scaler = mx.zeros((out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Use more efficient initialization
        self.base_weight = mx.random.uniform(low=-0.1, high=0.1, shape=self.base_weight.shape) * self.scale_base
        self.spline_weight = mx.random.uniform(low=-0.1, high=0.1, shape=self.spline_weight.shape) * self.scale_spline
        
        if self.enable_standalone_scale_spline:
            self.spline_scaler = mx.random.uniform(low=-0.1, high=0.1, shape=self.spline_scaler.shape) * self.scale_spline

    def b_splines(self, x):
        batch_size = x.shape[0]
        
        # Reshape operations optimized
        x_expanded = x.reshape(batch_size, self.in_features, 1)
        grid_reshaped = self.grid[:-1].T.reshape(1, self.in_features, -1)
        next_grid_reshaped = self.grid[1:].T.reshape(1, self.in_features, -1)
        
        # Compute base splines
        bases = mx.logical_and(x_expanded >= grid_reshaped, x_expanded < next_grid_reshaped).astype(mx.float32)
        
        # Pre-compute grid differences for efficiency
        for k in range(1, self.spline_order + 1):
            grid_k = self.grid[k:-1].T.reshape(1, self.in_features, -1)
            grid_0 = self.grid[:-(k+1)].T.reshape(1, self.in_features, -1)
            grid_k1 = self.grid[(k+1):].T.reshape(1, self.in_features, -1)
            grid_1 = self.grid[1:(-k)].T.reshape(1, self.in_features, -1)
            
            # Avoid division by zero with small epsilon
            denom1 = mx.maximum(grid_k - grid_0, 1e-6)
            denom2 = mx.maximum(grid_k1 - grid_1, 1e-6)
            
            bases = ((x_expanded - grid_0) / denom1 * bases[:, :, :-1]) + \
                   ((grid_k1 - x_expanded) / denom2 * bases[:, :, 1:])
                   
        return bases

    def curve2coeff(self, x, y):
        # Optimize matrix operations
        A = self.b_splines(x).transpose(1, 0, 2)
        B = y.transpose(1, 0, 2)
        
        # Use more stable solution method
        solution = mx.linalg.lstsq(A, B).solution
        return solution.transpose(0, 2, 1)

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler[:, :, None]
        return self.spline_weight

    def __call__(self, x):
        # Compute base output
        base_output = mx.matmul(self.hidden_act(x), self.base_weight.T)
        
        # Compute spline bases once
        spline_bases = self.b_splines(x)
        
        # Reshape for efficient matrix multiplication
        spline_bases_flat = spline_bases.reshape(x.shape[0], -1)
        spline_weights_flat = self.scaled_spline_weight.reshape(self.out_features, -1)
        
        # Compute spline output
        spline_output = mx.matmul(spline_bases_flat, spline_weights_flat.T)
        
        return base_output + spline_output

    def update_grid(self, x, margin=0.01):
        batch = x.shape[0]
        
        # Compute splines and outputs more efficiently
        splines = self.b_splines(x).transpose(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.transpose(1, 2, 0)
        unreduced_spline_output = mx.matmul(splines, orig_coeff).transpose(1, 0, 2)
        
        # Sort x once and reuse
        x_sorted = mx.sort(x, axis=0)
        
        # Use vectorized operations for grid creation
        indices = mx.linspace(0, batch - 1, self.grid_size + 1, dtype=mx.int32).astype(mx.int64)
        grid_adaptive = x_sorted[indices]
        
        # Compute uniform grid
        x_min, x_max = x_sorted[0], x_sorted[-1]
        uniform_step = (x_max - x_min + 2 * margin) / self.grid_size
        grid_uniform = mx.arange(self.grid_size + 1).reshape(-1, 1).astype(mx.float32) * uniform_step + x_min - margin
        
        # Blend grids
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        
        # Create extended grid
        steps_before = uniform_step * mx.arange(self.spline_order, 0, -1).reshape(-1, 1)
        steps_after = uniform_step * mx.arange(1, self.spline_order + 1).reshape(-1, 1)
        
        grid = mx.concatenate([
            grid[:1] - steps_before,
            grid,
            grid[-1:] + steps_after
        ], axis=0)
        
        # Update grid and weights
        self.grid = grid.T
        self.spline_weight = self.curve2coeff(x, unreduced_spline_output)
        
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # More numerically stable implementation
        abs_weights = mx.abs(self.spline_weight)
        l1_norm = abs_weights.mean(axis=-1)
        reg_activation = l1_norm.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        p = l1_norm / (reg_activation + epsilon)
        reg_entropy = -mx.sum(p * mx.log(p + epsilon))
        
        return regularize_activation * reg_activation + regularize_entropy * reg_entropy

class KAN(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layers_hidden: Optional[List[int]] = None,
    ):
        super().__init__()

        self.args = args

        # Determine layer dimensions
        if layers_hidden is None:
            if args.layers_hidden is None:
                layers_hidden = [args.in_features * args.out_features] + [args.hidden_dim] * (args.num_layers - 1) + [args.num_classes]
            else:
                layers_hidden = args.layers_hidden
        else:
            self.args.layers_hidden = layers_hidden

        self.grid_size = args.grid_size
        self.spline_order = args.spline_order
        
        # Create layers as a ModuleList for better parameter management
        self.layers = nn.ModuleList([
            KANLinear(
                in_features=in_dim,
                out_features=out_dim,
                grid_size=args.grid_size,
                spline_order=args.spline_order,
                scale_noise=args.scale_noise,
                scale_base=args.scale_base,
                scale_spline=args.scale_spline,
                hidden_act=args.hidden_act,
                grid_eps=args.grid_eps,
                grid_range=args.grid_range,
            )
            for in_dim, out_dim in zip(layers_hidden, layers_hidden[1:])
        ])
    
    def __call__(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # More efficient implementation using sum
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) 
                  for layer in self.layers)
    
    @staticmethod
    def load_model(folder_path: str):
        config_path = os.path.join(folder_path, "config.json")
        model_path = os.path.join(folder_path, "model.safetensors")
        
        config = load_config(config_path)
        model = KAN(config)
        
        # Load weights
        weights = tree_unflatten(list(mx.load(model_path).items()))
        model.update(weights)
        
        return model