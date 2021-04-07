'''
https://arxiv.org/abs/1704.01665
'''

from parameters import device
from itertools import chain
import torch
from torch.nn import Linear as Lin, ModuleList as MList
from torch.nn.functional import relu
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops
# from torchviz import make_dot

# device = G_DEVICE #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiS2V(torch.nn.Module):
    def __init__(self,node_input_channels,output_channels,node_feature_channels=(6,6,4),iteration_num=1):
        super(self.__class__,self).__init__()

        ##### TODO
        self.category = 3
        ##### need more general module

        assert(node_input_channels.shape == (self.category,))
        # assert(edge_channels.shape == (self.category,self.category))
        assert(output_channels.shape == (self.category,))
        # assert(edge_channels.shape == (self.category,self.category))
        
        self.lin1 = MList([Lin(node_input_channels[i],node_feature_channels[i]) for i in range(self.category)])
        self.lin2 = MList([MList([Lin(node_feature_channels[i],node_feature_channels[j]) for j in range(self.category)]) for i in range(self.category)])
        self.lin3 = MList([Lin(node_feature_channels[i],node_feature_channels[i]) for i in range(self.category)])
        # self.lin4 = MList([MList([Lin(edge_channels[i][j],node_feature_channels[j]) for j in range(self.category)]) for i in range(self.category)])
        self.lin5 = MList([Lin(2*node_feature_channels[i],output_channels[i]) for i in range(self.category)])
        self.lin6 = MList([MList([Lin(node_feature_channels[i],node_feature_channels[j]) for j in range(self.category)]) for i in range(self.category)])
        self.lin7 = MList([Lin(node_feature_channels[i],node_feature_channels[i]) for i in range(self.category)])

        self.node_input_channels = node_input_channels
        self.node_feature_channels = node_feature_channels
        # self.edge_channels = edge_channels
        self.output_channels = output_channels
        self.iteration_num = iteration_num

    def forward(self,x,edge_index):
        '''
        obs : ( x, edge_index, edge_attr, ( additions ))
            additions : ( start, now)
        Q function return : out
        '''
        assert(len(x) == self.category)
        assert(len(edge_index) == self.category**2)
        # assert(len(edge_attr) == self.category**2)

        features = x
        # print('features: ', features )
        for i in range(self.iteration_num):
            features = self.s2v(x,features,edge_index)
        
        out = []
        for i in range(self.category):
            o = torch.zeros(x[i].shape[0],2*self.node_feature_channels[i]).to(device)
            for j in range(self.category):
                eii = j*self.category+i
                o1 = self.lin6[j][i](features[j].sum(dim=0).repeat(x[i].size(0),1))
                o2 = self.lin7[i](features[i])
                o = o + torch.cat([o1,o2],dim=1)
            o = self.lin5[i](relu(o))
            out.append(o)
        
        return out

    def s2v(self,x,features,edge_index,need_add_loops=False,verbose=False):
        assert(len(features) == self.category)

        # for i in range(self.category):
        #     index = i*self.category+i
        #     edge_index[index],_ = add_self_loops(edge_index[index])
        
        # for o in edge_index:
        #     print(o.shape)
        #     print(o)

        o1 = [self.lin1[i](x[i]) for i in range(self.category)] # c

        o2 = []
        for i in range(self.category):
            o = torch.zeros([x[i].shape[0],self.node_feature_channels[i]]).to(device)
            for j in range(self.category):
                eii = j*self.category+i
                out = features[j].index_select(0,edge_index[eii][0])
                # print('i, j, out: ', i, j, out)
                index = edge_index[eii][1]
                # print('before scatter, out shape: ', out.shape)
                out = scatter_add(out,index,dim=0)
                # print('after scatter_add, out shape: ', out.shape)
                o = o + self.lin2[j][i](out)
            o2.append(o)

     

        out = [o1[i]+o2[i] for i in range(self.category)]
        # out = [o2[i] for i in range(self.category)]

        return out

class MyS2V(torch.nn.Module):
    def __init__(self,node_input_channels,edge_channels,output_channels,node_feature_channels=10,iteration_num=3):
        super(self.__class__,self).__init__()
        self.lin1 = Lin(node_input_channels,node_feature_channels,bias=True)
        self.lin2 = Lin(node_feature_channels,node_feature_channels,bias=True)
        self.lin3 = Lin(node_feature_channels,node_feature_channels,bias=True)
        self.lin4 = Lin(edge_channels,node_feature_channels,bias=True)
        self.lin5 = Lin(2*node_feature_channels,output_channels,bias=True)
        self.lin6 = Lin(node_feature_channels,node_feature_channels,bias=True)
        self.lin7 = Lin(node_feature_channels,node_feature_channels,bias=True)

        self.node_input_channels = node_input_channels
        self.node_feature_channels = node_feature_channels
        self.edge_channels = edge_channels
        self.output_channels = output_channels
        self.iteration_num = iteration_num

    def forward(self, x, edge_index, edge_attr, with_sum:bool=False):
        '''
        obs : ( x, edge_index, edge_attr, ( additions ))
            additions : ( start, now)
        Q function return : out,features
        '''
        features = torch.zeros(x.shape[0], self.node_feature_channels).to(device)
        for i in range(self.iteration_num):
            features = self.s2v(x, features, edge_index, edge_attr)
        
        o1 = self.lin6(features.sum(dim=0)).repeat(x.size(0),1)
        o2 = self.lin7(features)
        out = torch.cat([o1, o2], dim=1)
        out = self.lin5(relu(out))
        if with_sum:
            return out, out.sum(dim=0)
        else:
            return out

    def s2v(self, x, features, edge_index, edge_attr):
        edge_index_with_loop, _ = add_self_loops(edge_index)

        o1 = self.lin1(x)

        out = features.index_select(0, edge_index_with_loop[0])
        index = edge_index_with_loop[1]
        out = scatter_add(out, index, dim=0)
        o2 = self.lin2(out)

        out = self.lin4(edge_attr)
        out = relu(out)
        out = scatter_add(out, edge_index[1], dim=0)
        o3 = self.lin3(out)

        out = relu(o1 + o2 + o3)
        # out = relu(o2 + o3)

        return out


if __name__ == '__main__':
    def test_mys2v():
        node_num = 5
        edge_num = 16
        node_input_channels = 1
        node_feature_channels = 4
        edge_channels = 1
        output_channels = 1

        node_input = torch.randint(0,2,(node_num,node_input_channels)).type(torch.FloatTensor)
        edge_index = torch.LongTensor([[0,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],[1,2,3,4,0,2,4,0,1,3,0,2,4,0,1,3]])
        edge_attr = torch.randn(edge_num,edge_channels)

        model = MyS2V(node_input_channels,node_feature_channels,edge_channels,output_channels)
        out = model(node_input,edge_index,edge_attr)
        # print(model)
        # print(out.shape,features.shape)
        # print("model out : \n",out)
        # print("inputs : \n",node_input)
    
    def test_multis2v():
        import numpy as np
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        node_input_channels = np.array([2,1])
        node_features = np.array([10,8])
        edge_channels = np.array([[4,3],[2,1]])
        output_channels = np.array([2,1])
        model = MultiS2V(node_input_channels,edge_channels,output_channels,node_features).to(device)

        import itertools
        n_num = [3,4]
        category = len(n_num)
        a = [list(range(n)) for n in n_num]
        x = [torch.randn(n_num[i],node_input_channels[i]).to(device) for i in range(category)]
        # print('x: ', x)
        features = [torch.randn(n_num[i],node_features[i]).to(device) for i in range(category)]
        # print('features: ', features)
        edges = [torch.LongTensor(list(itertools.product(a[i],a[j]))).T.to(device) for i in range(category) for j in range(category)]
        # print('edges: ', edges)
        e_attr = [torch.randn(edges[i*category+j].shape[1],edge_channels[i][j]).to(device) for i in range(category) for j in range(category)]
        # print('e_attr: ', e_attr)
        # for l in edges:
        #     print(l.shape)
        #     print(l)
        out = model(x,edges,e_attr)
        # print('out: ', out)
        # for o in out:
            # print(o.shape)

        # for i in range(len(out)):
        # for i in range(1):
        #     make_dot(out[i],params=dict(list(model.named_parameters()))).render("__temp__"+str(i), format="png")


if __name__ == '__main__':
    test_multis2v()