import torch
from torch.nn import Linear,ModuleList as MList
from typing import Union, Optional, List
import itertools as it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_matrix(l,n):
    return [l[i:i+n] for i in range(0, len(l), n)]

class HeterogeneousGraph(torch.nn.Module):
    def __init__(self, MODEL, c_node_in, c_node_out, c_hidden = 8):
        super(self.__class__,self).__init__()

        self.category = len(c_node_in)
        assert(c_node_in.shape == (self.category,))
        # assert(c_edge.shape == (self.category,self.category))
        assert(c_node_out.shape == (self.category,))
        # assert(c_edge.shape == (self.category,self.category))

        self.c_node_in = c_node_in
        # self.c_edge = c_edge
        self.c_node_out = c_node_out
        self.c_hidden = c_hidden

        self.model = MList([MList([MODEL((c_node_in[i],c_node_in[j]),c_hidden) for j in range(self.category)]) for i in range(self.category)])

        self.lin = MList([Linear(self.category*c_hidden,c_node_out[i]) for i in range(self.category)])

    def forward(self,x,edge_index):

        edge_index = to_matrix(edge_index,self.category)
        # edge_attr = to_matrix(edge_attr,self.category)
        assert(len(x) == self.category)
        assert(len(edge_index) == self.category)
        # assert(len(edge_attr) == self.category)

        o = [[self.model[i][j]((x[i],x[j]),edge_index[i][j],size=(None if i==j else (x[i].shape[0],x[j].shape[0]))) for j in range(self.category)] for i in range(self.category)]
        
        outs = [self.lin[i](torch.cat([o[j][i] for j in range(self.category)],dim=-1)) for i in range(self.category)]
        return outs

class MultiHeterGraph(torch.nn.Module):
    def __init__(self, MODEL, channels, c_edge):
        super(self.__class__,self).__init__()
        self.layer_num = channels.shape[0]-1
        self.channels = channels
        self.c_edge = c_edge
        self.models = MList([HeterogeneousGraph(MODEL,channels[i],channels[i+1],c_edge) for i in range(self.layer_num)])
    
    def forward(self,x,edge_index,edge_attr):
        o = x
        for m in self.models:
            o = m(o,edge_index,edge_attr)
        return o