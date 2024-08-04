import numpy as np

import mlx.core as mx

from kan import KAN
# from kan.args import ModelArgs
from trainer.simpletrainer import SimpleTrainer
from trainer.trainer_args import TrainArgs

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


SimpleTrainer(
    model=kan_model,
    dataset_type="custom",
    train_set=(train_X, train_y),
    validation_set=(val_X, val_y),
    test_set=(test_X, test_y),
    max_steps=1000,
    epochs=2,
    max_train_batch_size=32,
    max_val_batch_size=32,
    max_test_batch_size=32
)