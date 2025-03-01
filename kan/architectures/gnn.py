# Copyright © 2024 Gökdeniz Gülmez

import mlx.core as mx
import mlx.nn as nn

from ..kan import KANLinear
from .conv import KAN_Convolutional_Layer

class KanGNN(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_dim: int,
            out_features: int,
            grid_size: int = 5,
            layers_hidden: int = 6,
            use_bias: bool = False
        ):
        super().__init__()
        self.layers_hidden = layers_hidden
        
        self.ln1 = nn.Linear(in_features, hidden_dim, bias=use_bias)
        #self.ln1 = KANLayer(in_feat, hidden_dim, grid_size, addbias=use_bias)

        self.layers = []
        for i in range(layers_hidden):
            self.layers.append(KANLinear(hidden_dim, hidden_dim, grid_size, addbias=use_bias))

        self.layers.append(nn.Linear(hidden_dim, out_features, bias=False))
        #self.layers.append(KANLayer(hidden_dim, out_features, grid_size, addbias=False))

        # self.layers = []
        # self.layers.append(nn.Linear(in_features, hidden_dim, bias=use_bias))
        # for i in range(layers_hidden):
        #     self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))
        # self.layers.append(nn.Linear(hidden_dim, out_features, bias=use_bias))

    
    def forward(self, x, adj):
        x = self.ln1(x)
        #x = self.ln1(spmm(adj, x))
        for layer in self.layers[:self.layers_hidden-1]:
            x = layer(spmm(adj, x))
            #x = layer(x)
        x = self.layers[-1](x)
            
        return x.log_softmax(dim=-1)