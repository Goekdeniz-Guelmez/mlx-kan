from typing import Optional, Tuple
from dataclasses import dataclass

import mlx.nn as nn


@dataclass
class TrainArgs:
    train_algorithm: str = "simple"
    dataset: str = "custom"
    max_steps: int = 0
    epochs: int = 2
    max_train_batch_size: int = 32
    max_val_batch_size: int = 32
    max_test_batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    clip_grad_norm: bool = False
    save_path: str = "./models"