import torch

import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear
from emlp.groups import SO
from emlp.reps import Scalar, Vector
from emlp.nn.pytorch import Linear as ELinear

from utils import euclidean_feats, unsorted_segment_sum


class EB(MessagePassing):
    def __init__(
        self, n_hidden: int, c_weight: float = 1.0, is_last: bool = False
    ) -> None:
        super(EB, self).__init__(aggr="add", flow="source_to_target")

        # dims for norm & inner product + (delr, delphi, delz, delR)
        self.n_edge_attributes = 6
        G = SO(3)
        self.repin = (Vector + Vector + self.n_edge_attributes * Scalar)(G)
        self.repout = (n_hidden * Scalar)(G)

        # Controls the scale of x during updates
        self.c_weight = c_weight

        # MLP to create the message
        self.phi_e = Sequential(
            ELinear(self.repin, self.repout),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            ELinear(self.repout, self.repout),
            nn.ReLU(),
        )

        layer = Linear(n_hidden, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        # MLP to generate attention weights
        self.phi_x = Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), layer)

        # MLP to generate weights for the messages
        self.phi_m = Sequential(ELinear(self.repout, Scalar(G)), nn.Sigmoid())

        # Meesage vector
        self.m = None

    def forward(self, x, edge_index, edge_attr):
        norms, dots, x_diff = euclidean_feats(edge_index, x)
        x_tilde = self.propagate(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
            norms=norms,
            dots=dots,
            x_diff=x_diff,
            size=None,
        )
        return x_tilde, self.m

    def message(self, x_i, x_j, edge_attr, norms, dots):
        # x_i -> incoming
        # x_j -> outgoing
        m_ij = torch.cat([x_i, x_j, edge_attr, norms, dots], dim=1)
        m_ij = self.phi_e(m_ij)
        w = self.phi_m(m_ij)
        self.m = m_ij * w
        return self.m

    def aggregate(self, m_ij, x, edge_index, x_diff):
        row, _ = edge_index
        update_val = x_diff * self.phi_x(m_ij)
        # LorentzNet authors clamp the update tensor as a precautionary measure
        update_val = torch.clamp(update_val, min=-100, max=100)
        x_agg = unsorted_segment_sum(update_val, row, num_segments=x.size(0))
        return x_agg

    def update(self, x_agg, x):
        x = x + self.c_weight * x_agg
        return x


class EuclidNet(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_layers: int,
        n_output: int,
        c_weight: float = 1e-3,
    ) -> None:
        super(EuclidNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_output = n_output

        self.EBs = nn.ModuleList(
            [
                EB(n_hidden=self.n_hidden, c_weight=c_weight)
                for _ in range(self.n_layers)
            ]
        )

        # MLP to produce edge weights
        self.edge_mlp = nn.Sequential(
            Linear(2 * n_input, self.n_hidden),
            nn.ReLU(),
            Linear(self.n_hidden, n_hidden),
            nn.ReLU(),
            Linear(self.n_hidden, self.n_output),
        )

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.n_layers):
            x, m = self.EBs[i](x, edge_index, edge_attr)
        out = torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=1)
        return torch.sigmoid(self.edge_mlp(out))
