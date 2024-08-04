from typing import Optional, Tuple
from dataclasses import dataclass

import mlx.nn as nn


@dataclass
class TrainArgs:
    train_algorythm: str = "KAN"
    dataset_type: str = "custom"
    max_steps: Optional[int] = 1000
    epochs: Optional[int] = 2
    max_train_batch_size: int = 32
    max_val_batch_size: int = 32
    max_test_batch_size: int = 32