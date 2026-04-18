from itertools import chain
from typing import Iterator

import torch
import torch.nn as nn

from src.models.components.mlp import MLP


class AirflowNN(nn.Module):
    def __init__(self, nb_hidden_layers, size_hidden_layers, bn_bool, encoder, decoder):
        super(AirflowNN, self).__init__()

        self.nb_hidden_layers = nb_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.bn_bool = bn_bool
        self.activation = nn.ReLU()

        self.encoder = MLP(encoder, batch_norm=False)
        self.decoder = MLP(decoder, batch_norm=False)

        self.dim_enc = encoder[-1]

        self.nn = MLP(
            [self.dim_enc] + [self.size_hidden_layers] * self.nb_hidden_layers + [self.dim_enc],
            batch_norm=self.bn_bool,
        )

    def forward(self, data):
        z = self.encoder(data)
        z = self.nn(z)
        z = self.decoder(z)
        return z


class AirflowNNMTLDecoder(nn.Module):
    def __init__(self, nb_hidden_layers, size_hidden_layers, bn_bool, encoder, decoders, number_decoders):
        super().__init__()

        self.bn_bool = bn_bool
        self.activation = nn.ReLU()
        self.number_decoders = number_decoders

        self.encoder = MLP(encoder, batch_norm=False)
        for i in range(number_decoders):
            setattr(
                self,
                f"decoder_{i}",
                MLP(decoders, batch_norm=False),
            )

        self.dim_enc = encoder[-1]

        self.nn = MLP(
            [self.dim_enc] + [size_hidden_layers] * nb_hidden_layers + [self.dim_enc],
            batch_norm=self.bn_bool,
        )

    def forward(self, data):
        z = self.encoder(data)
        z = self.nn(z)
        return torch.concatenate(
            ([getattr(self, f"decoder_{i}")(z) for i in range(self.number_decoders)]),
            dim=-1,
        )

    def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return chain(
            self.encoder.parameters(),
            self.nn.parameters(),
        )

    def task_specific_parameters(
        self,
    ) -> Iterator[torch.nn.parameter.Parameter]:
        return chain([getattr(self, f"decoder_{i}").parameters() for i in range(self.number_decoders)])

    def last_shared_parameters(self):
        return []

    def reset(self):
        self.encoder.zero_grad(set_to_none=False)
        self.nn.zero_grad(set_to_none=False)
