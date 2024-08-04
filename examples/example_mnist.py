# Import necessary libraries
import numpy as np
import mlx.core as mx  # Import mlx core module for array manipulation

# Import KAN model and arguments
from kan import KAN
from kan.args import ModelArgs

# Import SimpleTrainer and training arguments
from trainer.simpletrainer import SimpleTrainer
from trainer.trainer_args import TrainArgs

# Import MNIST dataset loader
import quick_scripts.mnist as mnist

# Define the model parameters
num_layers = 2  # Number of layers in the model
in_features = 28  # Input feature dimension (e.g., image width for MNIST)
out_features = 28  # Output feature dimension (e.g., image height for MNIST)
hidden_dim = 64  # Dimension of hidden layers
num_classes = 10  # Number of output classes for classification (e.g., digits 0-9)

# Initialize the KAN model with specified architecture
kan_model = KAN(
    layers_hidden=[in_features * out_features] + [hidden_dim] * (num_layers - 1) + [num_classes],  # Define layers: input layer, hidden layers, output layer
    args=ModelArgs  # Pass model arguments
)

# Load the MNIST dataset
train_images, train_labels, test_images, test_labels = map(mx.array, getattr(mnist, "mnist")())  # Convert dataset to mlx arrays

# Set training arguments
TrainArgs.max_steps = 1000  # Maximum number of training steps

# Initialize and run the SimpleTrainer
SimpleTrainer(
    model=kan_model,  # Model to be trained
    args=TrainArgs,  # Training arguments
    train_set=(train_images, train_labels),  # Training dataset
    validation_set=(test_images, test_labels),  # Validation dataset (using test set for validation)
    test_set=(test_images, test_labels),  # Testing dataset
    validation_interval=1000,  # Interval for validation
    logging_interval=10  # Interval for logging
)