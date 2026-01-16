# MIT License
#
# Original work:
# Copyright (c) 2023 PyG Team <team@pyg.org>
#
# Modifications:
# Copyright (c) 2026 D-Stiv
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)

from torch_geometric.utils import softmax
from train_util import exponential_decay

class GINEConv(MessagePassing):
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')  # Use 'add' aggregation by default
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None
        
        self.half_life = 2
        self.lin_N0 = torch.nn.Sequential(
            Linear(edge_dim, 1),
            torch.nn.ReLU()
        )
        self.lin_scale = torch.nn.Sequential(
            Linear(edge_dim, 1),
            torch.nn.ReLU()
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        
        ts = edge_attr[:, :1]
        edge_attr = edge_attr[:, 1:]

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor, ts: OptTensor)
        out = self.propagate(x=x, edge_attr=edge_attr, edge_index=edge_index, ts=ts, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, ts: Tensor, edge_index: Tensor) -> Tensor:
        # Transform edge features if necessary
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        # Compute the message: ReLU(x_j + e_ji)
        message = (x_j + edge_attr).relu()

        # Compute alpha_{i,j} using softmax over timestamps
        if ts.min() == ts.max():
            beta = 1
        else:
            N0 = self.lin_N0(edge_attr).mean()
            mp_scale = self.lin_scale(edge_attr).mean()
            ts = exponential_decay(N0=N0, decay_constant=1/self.half_life, t=1/ts)

            alpha = softmax(ts, index=edge_index[1])
            beta = 1 + mp_scale*alpha  # Normalize timestamps for each node

        # Weight the messages by alpha_{i,j}
        weighted_message = message * beta

        return weighted_message

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'