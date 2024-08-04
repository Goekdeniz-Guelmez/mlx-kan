import numpy as np

import mlx.core as mx

from kan import KAN
# from kan.args import ModelArgs
from trainer.simpletrainer import SimpleTrainer
from trainer.trainer_args import TrainArgs

import quick_scripts.mnist as mnist

num_layers = 2
in_features = 28
out_features = 28
hidden_dim = 64
num_classes = 10


kan_model = KAN([in_features * out_features] + [hidden_dim] * (num_layers - 1) + [num_classes])

def create_custom_dataset(num_samples: int, num_features: int, num_classes: int):
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, num_classes, size=num_samples)
    return X, y

# Generate custom datasets
train_samples, val_samples, test_samples = 1000, 200, 200
num_features = in_features * out_features
num_classes = num_classes

train_X, train_y = create_custom_dataset(train_samples, num_features, num_classes)
val_X, val_y = create_custom_dataset(val_samples, num_features, num_classes)
test_X, test_y = create_custom_dataset(test_samples, num_features, num_classes)


train_images, train_labels, test_images, test_labels = map(mx.array, getattr(mnist, "mnist")()) # can be ["mnist", "fashion_mnist"]

TrainArgs.max_steps = 1000
SimpleTrainer(
    model=kan_model,
    args=TrainArgs,
    train_set=(train_X, train_y),
    validation_set=(val_X, val_y),
    test_set=(test_X, test_y),
    validation_interval=1000,
    logging_interval=10
)