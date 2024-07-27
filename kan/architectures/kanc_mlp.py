# Copyright © 2024 Gökdeniz Gülmez

import mlx.core as mx
import mlx.nn as nn

from kan.kan_convolution.kanConv import KAN_Convolutional_Layer

class KANC_MLP(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size= (3,3),
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size = (3,3),
            device = device
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.linear1 = nn.Linear(625, 256)
        self.linear2 = nn.Linear(256, 10)


    def __call__(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = mx.flatten(self.pool1(x))
        x = self.linear1(x)
        x = self.linear2(x)
        x = nn.log_softmax(x, axis=1)
        return x