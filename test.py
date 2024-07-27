# Copyright © 2024 Gökdeniz Gülmez

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from kan.kan import KAN
from kan.architectures.ckan import CKAN
from kan.architectures.kkan import KKAN_Convolutional_Network

# Initialize models
kan1 = KAN(2)
kan2 = KAN([2, 3])
kkan = KKAN_Convolutional_Network()
ckan = CKAN()

# Print model summaries
print(kan1)
print(kan2)
print(kkan)
print(ckan)

# Determine input size based on the first layer's expected input features
input_size_kan1 = kan1.layers[0].in_features
input_size_kan2 = kan2.layers[0].in_features
input_size_kkan = (1, 25, 25)
input_size_ckan = (1, 28, 28)  # Example size, adjust according to the input size expected by CKAN

# Dummy datasets
X_kan1 = mx.random.uniform(0, 1, shape=(100, input_size_kan1))  # 100 samples, input_size_kan1 features
y = mx.random.randint(0, 2, shape=(100,))  # 100 samples, binary classification

X_kan2 = mx.random.uniform(0, 1, shape=(100, input_size_kan2))  # 100 samples, input_size_kan2 features
X_kkan = mx.random.uniform(0, 1, shape=(100, *input_size_kkan))  # 100 samples, 1 channel, 25x25 input size
X_ckan = mx.random.uniform(0, 1, shape=(100, *input_size_ckan))  # 100 samples, 1 channel, 28x28 input size (adjust as needed)

# Training function
def test_train(model, X, y):
    def loss_fn(model, X, y):
        logits = model(X)
        loss = nn.losses.cross_entropy(logits, y)
        return mx.mean(loss)
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)

    model.train()
    loss, grads = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss.item()

# Testing the training function
try:
    loss = test_train(kan1, X_kan1, y)
    print(f"Initial loss for KAN1: {loss}")
except Exception as e:
    print(f"Error training KAN1: {e}")

try:
    loss = test_train(kan2, X_kan2, y)
    print(f"Initial loss for KAN2: {loss}")
except Exception as e:
    print(f"Error training KAN2: {e}")

try:
    loss = test_train(kkan, X_kkan, y)
    print(f"Initial loss for KKAN: {loss}")
except Exception as e:
    print(f"Error training KKAN: {e}")

try:
    loss = test_train(ckan, X_ckan, y)
    print(f"Initial loss for CKAN: {loss}")
except Exception as e:
    print(f"Error training CKAN: {e}")