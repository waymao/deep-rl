from typing import List
from torch import nn
import torch
import numpy as np

def init_weights(m, scale=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            nn.init.orthogonal_(m.weight)
            m.weight.data *= scale
            m.bias.data.fill_(0.0)

def get_MLP(
        num_features: int, 
        num_actions: int, 
        hidden_layers: List[int],
        use_relu=True,
        final_layer_softmax=False,
        output_init_scale=np.sqrt(2),
        device="cpu"
    ) -> nn.Module:
        layers = []
        last_input = num_features
        # first few layers except for the last one
        for layer_neurons in hidden_layers:
            linear_layer = nn.Linear(last_input, layer_neurons)
            # init_weights(linear_layer)
            layers.append(linear_layer)
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            last_input = layer_neurons

        last_layer = nn.Linear(last_input, num_actions)
        # init_weights(last_layer, output_init_scale)
        layers.append(last_layer)
        if final_layer_softmax:
             layers.append(nn.Softmax(-1))
        network = nn.Sequential(*layers)

        return network.to(device)
