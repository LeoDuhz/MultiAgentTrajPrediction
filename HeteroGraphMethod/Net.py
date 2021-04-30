import torch
import numpy as np
from parameters import playerNum
from pna.mypna import MyPNAConv
from mys2v import MultiS2V
from torch.nn import Linear as Lin, ModuleList as MList
from torch.nn.functional import relu
import torch.nn.functional as F

node_input_channels = np.array([4,4,4])
node_features = np.array([4,4,4])
output_channels = np.array([2,2,2])
# c_edge = np.array([[2,2,2],[2,2,2],[2,2,2]])
# c_edge = np.array([[1,3],[2,4]])

node_num = [playerNum,playerNum,1]
category = len(node_num)

class s2vNet(torch.nn.Module):
    def __init__(self):
        super(s2vNet, self).__init__()
        
        self.conv1 = MultiS2V(node_input_channels, output_channels, node_features)
        # self.lin1 = Linear(6,6)
        # self.lin2 = Linear(6,6)
        # self.lin1 = MList([Lin(output_channels[i],output_channels[i]) for i in range(3)])
        # self.lin2 = MList([Lin(output_channels[i],output_channels[i]) for i in range(3)])
        # self.lin3 = MList([Lin(output_channels[i],output_channels[i]) for i in range(3)])
        # self.lin4 = MList([Lin(node_input_channels[i],node_input_channels[i]) for i in range(3)])

    def forward(self, x, edges):
        # for i in range(3):
        #     x[i] = self.lin4[i](x[i])
        edge_index = edges
        
        x = self.conv1(x, edge_index)
        # print('x shape: ', x[0].size())
        for i in range(3):
            # x[i] = F.relu(x[i])
            # x[i] = self.lin1[i](x[i])
            # x[i] = self.lin2[i](x[i])
            # x[i] = self.lin3[i](x[i])
            # x[i] = F.dropout(x[i], training=self.training)
            x[i] = torch.sigmoid(x[i])

        return x


class pnaNet(torch.nn.Module):
    def __init__(self):
        super(pnaNet, self).__init__()
        
        self.conv1 = MyPNAConv(node_input_channels, output_channels, deg=torch.Tensor([12]))
        # self.lin4 = MList([Lin(node_input_channels[i],node_input_channels[i]) for i in range(3)])

    def forward(self, data):
        x = data.x
        edge_index = data.edges
        edge_attr = [torch.ones([edge_index[i*category+j].shape[-1],c_edge[i][j]]) for i in range(len(node_input_channels)) for j in range(len(node_input_channels))]
        avg_deg = 12

        # for i in range(3):
        #     x[i] = self.lin4[i](x[i])
        
        x = self.conv1(x, edge_index)
        # print('x shape: ', x[0].size())
        for i in range(3):
            # x[i] = F.relu(x[i])
            # x[i] = self.lin1[i](x[i])
            # x[i] = self.lin2[i](x[i])
            # x[i] = self.lin3[i](x[i])
            # x[i] = F.dropout(x[i], training=self.training)
            x[i] = torch.sigmoid(x[i])

        return x

def PNAMODEL(c_in, c_out):
    return MyPNAConv(c_in, c_out, torch.Tensor([12]))      