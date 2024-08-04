from typing import Optional, List
from dataclasses import dataclass

import mlx.nn as nn


@dataclass
class ModelArgs:
    layers_hidden: Optional[List[int]] = None
    model_type: str = "KAN"
    num_layers: int = 2
    in_features: int = 28
    out_features: int = 28
    hidden_dim: int = 64
    num_classes: int = 10
    grid_size: int = 5
    spline_order: float = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    hidden_act = nn.SiLU
    grid_eps: float = 0.02
    grid_range = [-1, 1]