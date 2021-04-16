from typing import Union, Optional, List, Dict, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import ModuleList, Sequential, Linear, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree
from pna.aggregators import AGGREGATORS
from pna.scalers import SCALERS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyPNAConv(MessagePassing):
    def __init__(self, in_channels: Union[int,List[int]], out_channels: int,
                 deg: Tensor, edge_dim: Optional[int] = None, hidden_channels:int=8,
                 aggregators: List[str] = ['mean', 'min', 'max', 'std'], 
                 scalers: List[str] = ['identity', 'amplification', 'attenuation'],
                 pre_layers: int = 1, post_layers: int = 1, **kwargs):

        super(self.__class__, self).__init__(aggr=None, node_dim=0, **kwargs)

        if isinstance(in_channels,int):
            self.in_channels = (in_channels,in_channels)
        else:
            assert(len(in_channels)==2)
            self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]
        self.edge_dim = edge_dim

        self.F_in = self.in_channels
        self.F_out = self.out_channels
        self.F_hidden = self.hidden_channels

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in[0])

        # self.pre_nns = ModuleList()
        # self.post_nns = ModuleList()

        modules = [Linear(self.F_in[1] + (2 if self.edge_dim is not None else 1) * self.F_in[0], self.F_in[0])]
        for _ in range(pre_layers - 1):
            modules += [ReLU()]
            modules += [Linear(self.F_in[0], self.F_in[0])]
        self.pre_nns = (Sequential(*modules))

        in_channels = (len(aggregators) * len(scalers)) * self.F_in[0] + self.F_in[1]
        modules = [Linear(in_channels, self.F_hidden)]
        for _ in range(post_layers - 1):
            modules += [ReLU()]
            modules += [Linear(self.F_out, self.F_out)]
        self.post_nns = (Sequential(*modules))

        self.lin = Linear(self.F_hidden, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Union[Tensor,List[Tensor]], edge_index: Adj,
                edge_attr: OptTensor = None, size:Union[None,List[int]]=None) -> Tensor:
        if edge_index.nelement() == 0:
            out = torch.zeros([x[1].shape[0],(len(self.aggregators) * len(self.scalers)) * self.F_in[0]]).to(device)
        else:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = torch.cat([x[1], out], dim=-1)
        out = self.post_nns(out)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:
        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        hs = self.pre_nns(h)
        return hs

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype).view(-1,1)
        outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, dim={self.edge_dim})')
