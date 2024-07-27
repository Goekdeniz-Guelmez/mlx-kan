# Copyright © 2024 Gökdeniz Gülmez

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from kan.kan import KAN
from kan.architectures.ckan import CKAN
from kan.architectures.kkan import KKAN_Convolutional_Network

kan1 = KAN(2)
print(kan1)

kan2 = KAN([2, 3])
print(kan2)

kkan = KKAN_Convolutional_Network()
print(kkan)

ckan = CKAN()
print(ckan)

def test_train(model):
    def loss_fn(model, X, y):
        return mx.mean(nn.losses.cross_entropy(model(X), y))
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)

    model.train()
    loss, grads = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)