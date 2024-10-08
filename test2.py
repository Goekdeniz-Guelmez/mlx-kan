import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from kan import KANLinear
from kan.architectures.KANMLP import KANC_MLP


# Dummy input for inference
dummy_input = mx.random.randint(0, 256, (1, 1, 28, 28))
kanc_mlp = KANC_MLP()
print(kanc_mlp)
output = kanc_mlp(dummy_input)
print("Inference output:", output)

# Dummy inputs for training
dummy_inputs = mx.random.randint(0, 256, (8, 1, 28, 28))
dummy_targets = mx.random.randint(0, 10, (8,))



######## TRAINING UTILS
def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def train_epoch(model: nn.Module, optimizer: optim.Optimizer, train_images: mx.array, train_labels: mx.array, loss_and_grad_fn) -> float:
    model.train()
    total_loss = 0.0
    num_steps = 10
    for step in range(num_steps):
        loss, grads = loss_and_grad_fn(model, train_images, train_labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
        
    return total_loss / num_steps


def train(model, train_images, train_labels, num_epochs=100):
    optimizer = optim.AdamW(learning_rate=0.0004, weight_decay=0.003)  # Initialize a new optimizer for each model
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, optimizer, train_images, train_labels, loss_and_grad_fn)
        
        # Update grid points at the end of each epoch
        for name, layer in model.__dict__.items():
            if isinstance(layer, KANLinear):
                with mx.no_grad():
                    layer.update_grid(train_images)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')
