import torch
import numpy as np
from mys2v import MultiS2V
from torch.nn import Linear as Lin, ModuleList as MList
from torch.nn.functional import relu
import torch.nn.functional as F
from parameters import playerNum

node_input_channels = np.array([4,4,4])
node_features = np.array([4,4,4])
output_channels = np.array([2,2,2])

node_num = [playerNum,playerNum,1]
category = len(node_num)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = MultiS2V(node_input_channels, output_channels, node_features)
        # self.lin1 = Linear(6,6)
        # self.lin2 = Linear(6,6)
        # self.lin1 = MList([Lin(output_channels[i],output_channels[i]) for i in range(3)])
        # self.lin2 = MList([Lin(output_channels[i],output_channels[i]) for i in range(3)])
        # self.lin3 = MList([Lin(output_channels[i],output_channels[i]) for i in range(3)])
        self.lin4 = MList([Lin(node_input_channels[i],node_input_channels[i]) for i in range(3)])

    def forward(self, data):
        x = data.x
        edge_index = data.edges
        for i in range(3):
            x[i] = self.lin4[i](x[i])
        
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
