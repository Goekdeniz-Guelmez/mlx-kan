# Copyright © 2024-2025 Gökdeniz Gülmez

from typing import List, Optional, Union, Callable

import mlx.core as mx
import mlx.nn as nn

from kan.args import ModelArgs

class ActivationFactory:
    """Factory for creating activation functions with caching."""
    
    _ACTIVATIONS = {
        'silu': lambda: nn.SiLU(),
        'gelu': lambda: nn.GELU(),
        'relu': lambda: nn.ReLU(),
        'tanh': mx.tanh,
        'sigmoid': mx.sigmoid,
        'identity': lambda x: x,
    }
    
    @classmethod
    def create(cls, activation: Union[str, Callable, None]) -> Callable:
        """Create activation function from various input types."""
        if activation is None:
            return lambda x: x
        
        if isinstance(activation, str):
            act_name = activation.lower()
            if act_name not in cls._ACTIVATIONS:
                raise ValueError(f"Unsupported activation: {activation}. "
                               f"Available: {list(cls._ACTIVATIONS.keys())}")
            return cls._ACTIVATIONS[act_name]()
        
        if callable(activation):
            try:
                # Try to instantiate if it's a class
                return activation()
            except TypeError:
                # Already a function
                return activation
        
        return activation
    
class KANLinear(nn.Module):
    """Optimized KAN Linear layer with improved performance and memory usage."""
    
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
        hidden_act: Union[str, Callable, None] = 'silu',
        grid_eps: float = 0.02,
        grid_range: List[float] = None,
        bias: bool = True
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
        self.grid_eps = grid_eps
        self.grid_range = grid_range if grid_range is not None else [-1.0, 1.0]
        self.bias = bias
        
        # Validation
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.spline_order <= 0:
            raise ValueError("spline_order must be positive")
        if len(self.grid_range) != 2:
            raise ValueError("grid_range must have exactly 2 elements")
        
        # Create activation function
        self.hidden_act = ActivationFactory.create(hidden_act)
        
        # Pre-compute grid for efficiency
        self._initialize_grid()
        self._initialize_parameters()
    
    def _initialize_grid(self):
        """Initialize grid points with vectorized operations."""
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        
        # Create grid points in one go
        grid_indices = mx.arange(
            -self.spline_order, 
            self.grid_size + self.spline_order + 1,
            dtype=mx.float32
        )
        grid_points = grid_indices * h + self.grid_range[0]
        
        # Broadcast to all input features
        self.grid = mx.broadcast_to(
            grid_points[:, None], 
            (len(grid_points), self.in_features)
        )
    
    def _initialize_parameters(self):
        """Initialize all parameters with proper scaling."""
        # Base weights
        self.base_weight = mx.random.normal(
            shape=(self.out_features, self.in_features)
        ) * (self.scale_base / (self.in_features ** 0.5))
        
        # Spline weights
        spline_shape = (
            self.out_features, 
            self.in_features, 
            self.grid_size + self.spline_order
        )
        self.spline_weight = mx.random.normal(shape=spline_shape) * (
            self.scale_spline / (spline_shape[-1] ** 0.5)
        )
        
        # Optional spline scaler
        if self.enable_standalone_scale_spline:
            self.spline_scaler = mx.ones((self.out_features, self.in_features))
        
        # Bias
        if self.bias:
            self.bias_param = mx.zeros(self.out_features)
    
    def b_splines(self, x: mx.array) -> mx.array:
        """Compute B-spline basis functions with optimized operations."""
        batch_size, in_features = x.shape
        
        # Expand dimensions for broadcasting
        x_expanded = x[:, :, None]  # [batch, in_features, 1]
        
        # Get grid segments
        grid_left = self.grid[:-1].T[None, :, :]  # [1, in_features, grid_points-1]
        grid_right = self.grid[1:].T[None, :, :]   # [1, in_features, grid_points-1]
        
        # Initialize with order 0 (indicator functions)
        bases = ((x_expanded >= grid_left) & (x_expanded < grid_right)).astype(mx.float32)
        
        # Iteratively build higher order splines
        for k in range(1, self.spline_order + 1):
            if bases.shape[-1] <= 1:
                break
                
            # Get grid points for current order
            grid_k_left = self.grid[k:-1].T[None, :, :]
            grid_k_right = self.grid[:-(k+1)].T[None, :, :]
            grid_k1_left = self.grid[(k+1):].T[None, :, :]
            grid_k1_right = self.grid[1:(-k)].T[None, :, :]
            
            # Compute denominators with numerical stability
            eps = 1e-8
            denom1 = mx.maximum(grid_k_left - grid_k_right, eps)
            denom2 = mx.maximum(grid_k1_left - grid_k1_right, eps)
            
            # Cox-de Boor recursion
            left_term = (x_expanded - grid_k_right) / denom1 * bases[:, :, :-1]
            right_term = (grid_k1_left - x_expanded) / denom2 * bases[:, :, 1:]
            
            bases = left_term + right_term
        
        return bases
    
    def curve2coeff(self, x: mx.array, y: mx.array) -> mx.array:
        """Convert curve samples to spline coefficients using least squares."""
        # Get B-spline basis
        A = self.b_splines(x)  # [batch, in_features, grid_size + spline_order]
        
        # Reshape for batch processing
        A_reshaped = A.transpose(1, 0, 2)  # [in_features, batch, basis_size]
        y_reshaped = y.transpose(1, 0, 2)  # [in_features, batch, out_features]
        
        # Solve least squares problem for each input feature
        coefficients = []
        for i in range(self.in_features):
            coeff = mx.linalg.lstsq(A_reshaped[i], y_reshaped[i]).solution
            coefficients.append(coeff)
        
        # Stack and reshape
        coefficients = mx.stack(coefficients, axis=0)  # [in_features, basis_size, out_features]
        return coefficients.transpose(2, 0, 1)  # [out_features, in_features, basis_size]
    
    @property
    def scaled_spline_weight(self) -> mx.array:
        """Get scaled spline weights."""
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler[:, :, None]
        return self.spline_weight
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with optimized computations."""
        # Input validation
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, got {x.shape[-1]}")
        
        # Base transformation
        base_output = mx.matmul(self.hidden_act(x), self.base_weight.T)
        
        # Spline transformation
        spline_bases = self.b_splines(x)  # [batch, in_features, basis_size]
        
        # Efficient matrix multiplication
        batch_size = x.shape[0]
        spline_bases_flat = spline_bases.reshape(batch_size, -1)
        spline_weight_flat = self.scaled_spline_weight.reshape(self.out_features, -1)
        
        spline_output = mx.matmul(spline_bases_flat, spline_weight_flat.T)
        
        # Combine outputs
        output = base_output + spline_output
        
        # Add bias if configured
        if self.bias:
            output = output + self.bias_param
        
        return output
    
    def update_grid(self, x: mx.array, margin: float = 0.01):
        """Update grid points based on input distribution."""
        batch_size = x.shape[0]
        
        # Store original outputs for coefficient computation
        spline_bases = self.b_splines(x)
        orig_coeffs = self.scaled_spline_weight
        
        # Compute original spline outputs
        spline_bases_flat = spline_bases.reshape(batch_size, -1)
        spline_weight_flat = orig_coeffs.reshape(self.out_features, -1)
        orig_outputs = mx.matmul(spline_bases_flat, spline_weight_flat.T)
        orig_outputs = orig_outputs.reshape(batch_size, self.out_features, 1)
        
        # Create adaptive grid based on input quantiles
        x_sorted = mx.sort(x, axis=0)
        
        # Use quantiles for better distribution coverage
        quantile_indices = mx.linspace(0, batch_size - 1, self.grid_size + 1)
        quantile_indices = mx.clip(quantile_indices, 0, batch_size - 1).astype(mx.int32)
        
        grid_adaptive = x_sorted[quantile_indices]
        
        # Create uniform grid as backup
        x_min, x_max = x_sorted[0], x_sorted[-1]
        grid_uniform = mx.linspace(
            x_min - margin, 
            x_max + margin, 
            self.grid_size + 1
        )[:, None]
        
        # Broadcast uniform grid to match adaptive grid shape
        grid_uniform = mx.broadcast_to(grid_uniform, grid_adaptive.shape)
        
        # Blend adaptive and uniform grids
        grid_new = (
            self.grid_eps * grid_uniform + 
            (1 - self.grid_eps) * grid_adaptive
        )
        
        # Extend grid for spline order
        step_size = (grid_new[-1] - grid_new[0]) / self.grid_size
        
        # Add padding for spline order
        left_indices = mx.arange(self.spline_order, 0, -1)
        right_indices = mx.arange(1, self.spline_order + 1)
        
        left_padding = grid_new[0] - step_size * left_indices[:, None]
        right_padding = grid_new[-1] + step_size * right_indices[:, None]
        
        # Concatenate all parts
        grid_extended = mx.concatenate([left_padding, grid_new, right_padding], axis=0)
        
        # Update grid and recompute coefficients
        self.grid = grid_extended
        self.spline_weight = self.curve2coeff(x, orig_outputs)
    
    def regularization_loss(
        self, 
        regularize_activation: float = 1.0, 
        regularize_entropy: float = 1.0
    ) -> mx.array:
        """Compute regularization loss with numerical stability."""
        # L1 regularization on spline weights
        l1_weights = mx.abs(self.spline_weight).mean(axis=-1)  # [out_features, in_features]
        reg_activation = l1_weights.sum()
        
        # Entropy regularization
        eps = 1e-8
        total_activation = reg_activation + eps
        probabilities = l1_weights.flatten() / total_activation
        
        # Compute entropy with numerical stability
        log_probs = mx.log(probabilities + eps)
        reg_entropy = -mx.sum(probabilities * log_probs)
        
        return regularize_activation * reg_activation + regularize_entropy * reg_entropy

class KAN(nn.Module):
    """Kolmogorov-Arnold Network with optimized architecture."""
    
    def __init__(
        self,
        args: ModelArgs,
        layers_hidden: Optional[List[int]] = None,
        config: Optional[ModelArgs] = ModelArgs(),
    ):
        super().__init__()
        
        self.args = args
        
        # Determine layer architecture
        if layers_hidden is None:
            layers_hidden = self._get_default_architecture(args)
        
        self.config = config
        self.layers_hidden = layers_hidden
        
        self.layers = [
            KANLinear(
                in_features=in_dim,
                out_features=out_dim,
                grid_size=getattr(args, 'grid_size', 5),
                spline_order=getattr(args, 'spline_order', 3),
                scale_noise=getattr(args, 'scale_noise', 0.1),
                scale_base=getattr(args, 'scale_base', 1.0),
                scale_spline=getattr(args, 'scale_spline', 1.0),
                enable_standalone_scale_spline=getattr(args, 'enable_standalone_scale_spline', True),
                hidden_act=getattr(args, 'hidden_act', 'silu'),
                grid_eps=getattr(args, 'grid_eps', 0.02),
                grid_range=getattr(args, 'grid_range', [-1.0, 1.0]),
                bias=getattr(args, 'bias', True)
            )
            for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ]
    
    def _get_default_architecture(self, args: ModelArgs) -> List[int]:
        """Get default layer architecture."""
        if hasattr(args, 'layers_hidden') and args.layers_hidden is not None:
            return args.layers_hidden
        
        # Create default architecture
        input_dim = getattr(args, 'in_features', 784)
        output_dim = getattr(args, 'num_classes', 10)
        hidden_dim = getattr(args, 'hidden_dim', 64)
        num_layers = getattr(args, 'num_layers', 3)
        
        if num_layers == 1:
            return [input_dim, output_dim]
        
        return [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
    
    def __call__(self, x: mx.array, update_grid: bool = False) -> mx.array:
        """Forward pass through the network."""
        for i in range(self.num_layers):
            layer = getattr(self, f'layer_{i}')
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def __call__(self, x: mx.array, update_grid: bool = False) -> mx.array:
        """Forward pass through the network."""
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(
        self, 
        regularize_activation: float = 1.0, 
        regularize_entropy: float = 1.0
    ) -> mx.array:
        """Compute total regularization loss."""
        total_loss = mx.array(0.0)
        
        for layer in self.layers:
            total_loss = total_loss + layer.regularization_loss(
                regularize_activation, regularize_entropy
            )
        
        return total_loss