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
# 
# ----
# Modifications points: lines 49-62; 74-75; 112-132
# - inserted weight decay 

import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import PNAConv
from utils import exponential_decay

from torch_geometric.utils import softmax

class TemporalPNAConv(PNAConv):
    def __init__(
        self,
        *args,
        half_life: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.half_life = half_life

        if self.edge_dim is None:
            raise ValueError("TemporalPNAConv requires edge features")

        self.lin_N0 = torch.nn.Sequential(
            torch.nn.Linear(self.edge_dim, 1),
            torch.nn.ReLU()
        )

        self.lin_scale = torch.nn.Sequential(
            torch.nn.Linear(self.edge_dim, 1),
            torch.nn.ReLU()
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
    ) -> Tensor:

        if edge_attr is None:
            raise ValueError("edge_attr must include timestamps")

        ts = edge_attr[:, :1]
        edge_attr = edge_attr[:, 1:]

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            ts=ts,
        )

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)
    
    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: OptTensor,
        edge_index: Tensor,
        ts: Tensor,
    ) -> Tensor:

        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            message = torch.cat([x_j, edge_attr], dim=-1)
        else:
            message = x_j

        # Temporal weighting
        if ts is None or ts.min() == ts.max():
            beta = torch.ones_like(ts)
        else:
            N0 = self.lin_N0(edge_attr).mean()
            mp_scale = self.lin_scale(edge_attr).mean()

            ts_weighted = exponential_decay(
                N0=N0,
                decay_constant=1 / self.half_life,
                t=1 / ts,
            )

            alpha = softmax(ts_weighted, index=edge_index[1])
            beta = 1 + mp_scale * alpha

        weighted_message = message * beta.unsqueeze(-1)
        h = torch.cat([x_i, weighted_message], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)
