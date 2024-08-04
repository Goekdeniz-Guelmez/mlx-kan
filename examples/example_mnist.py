import numpy as np

import mlx.core as mx

from kan import KAN
from kan.args import ModelArgs
from trainer.simpletrainer import SimpleTrainer
from trainer.trainer_args import TrainArgs

import quick_scripts.mnist as mnist

num_layers = 2
in_features = 28
out_features = 28
hidden_dim = 64
num_classes = 10

kan_model = KAN(
    layers_hidden=[in_features * out_features] + [hidden_dim] * (num_layers - 1) + [num_classes],
    args=ModelArgs
)

train_images, train_labels, test_images, test_labels = map(mx.array, getattr(mnist, "mnist")())

TrainArgs.max_steps = 1000
SimpleTrainer(
    model=kan_model,
    args=TrainArgs,
    train_set=(train_images, train_labels),
    validation_set=(test_images, test_labels),
    test_set=(test_images, test_labels),
    validation_interval=1000,
    logging_interval=10
)