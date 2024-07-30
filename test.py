import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tqdm import tqdm

from kan import KANLinear
from kan.architectures.KANMLP import LlamaKANMLP, SmallKANMLP, MiddleKANMLP, BigKANMLP


######## ARGS
input_data = mx.array([[12, 4]])
output_data = mx.array([[2, 4, 1]])
in_features = 2
hidden_dim = 4
out_features = 3


######## INFERENCE
smallmodel = SmallKANMLP(in_features, hidden_dim, out_features)
print(smallmodel)
out = smallmodel(input_data)
print(out)


middlemodel = MiddleKANMLP(in_features, hidden_dim, out_features)
print(middlemodel)
out = middlemodel(input_data)
print(out)


bigmodel = BigKANMLP(in_features, hidden_dim, out_features)
print(bigmodel)
out = bigmodel(input_data)
print(out)


llamamodel = LlamaKANMLP(in_features, hidden_dim, out_features)
print(llamamodel)
out = llamamodel(input_data)
print(out)




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




######## TRAINING smallmodel
print("Training smallmodel")

train(smallmodel, input_data, output_data, num_epochs=10)


######## TRAINING middlemodel
print("Training middlemodel")

train(middlemodel, input_data, output_data, num_epochs=10)


######## TRAINING bigmodel
print("Training bigmodel")

train(bigmodel, input_data, output_data, num_epochs=10)



######## TRAINING llamamodel
print("Training llamamodel")
train(llamamodel, input_data, output_data, num_epochs=10)