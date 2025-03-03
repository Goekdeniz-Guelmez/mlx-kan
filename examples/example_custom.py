# Copyright © 2024-2025 Gökdeniz Gülmez

# Import necessary libraries
import numpy as np

# Import KAN model and arguments
from kan import KAN
from kan.args import ModelArgs

# Import SFTTrainer and training arguments
from kan.trainer.sft import SFTTrainer
from kan.trainer.trainer_args import TrainArgs

# Define the model parameters
num_layers = 8  # Number of layers in the model
in_features = 28  # Input feature dimension
out_features = 28  # Output feature dimension
hidden_dim = 64  # Dimension of hidden layers
num_classes = 10  # Number of output classes for classification

# Initialize the KAN model with specified architecture
kan_model = KAN(
    layers_hidden=[in_features * out_features] + [hidden_dim] * (num_layers - 1) + [num_classes],  # Define layers: input layer, hidden layers, output layer
    args=ModelArgs  # Pass model arguments
)

def create_custom_dataset(num_samples: int, num_features: int, num_classes: int):
    """
    Create a custom dataset with random features and labels.
    
    Args:
        num_samples (int): Number of samples in the dataset.
        num_features (int): Number of features per sample.
        num_classes (int): Number of classes for classification.
    
    Returns:
        Tuple of numpy arrays: Features (X) and labels (y).
    """
    X = np.random.randn(num_samples, num_features)  # Generate random features
    y = np.random.randint(0, num_classes, size=num_samples)  # Generate random labels
    return X, y

# Generate custom datasets for training, validation, and testing
train_samples, val_samples, test_samples = 1000, 200, 200  # Number of samples for each dataset
num_features = in_features * out_features  # Calculate total number of features
num_classes = num_classes  # Number of classes (redundant but explicit)

train_X, train_y = create_custom_dataset(train_samples, num_features, num_classes)  # Training dataset
val_X, val_y = create_custom_dataset(val_samples, num_features, num_classes)  # Validation dataset
test_X, test_y = create_custom_dataset(test_samples, num_features, num_classes)  # Testing dataset

# Set training arguments
TrainArgs.max_steps = 100  # Maximum number of training steps

# Initialize and run the SFTTrainer
SFTTrainer(
    model=kan_model,  # Model to be trained
    args=TrainArgs,  # Training arguments
    train_set=(train_X, train_y),  # Training dataset
    validation_set=(val_X, val_y),  # Validation dataset
    test_set=(test_X, test_y),  # Testing dataset
    validation_interval=1000,  # Interval for validation
    logging_interval=100  # Interval for logging
)