import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from dataFormat import Player, BallData, GameData, TimeSeqGameData
from dataPreprocess import generateTimeSeqData, convertPlayData2List, convertPlayData2Y, convertBallData2List, convertBallData2Y, generateY, generateFullyConnetedEdge, checkPlayerDataValid, checkBallDataValid


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.data, self.slices = None, None
        self.timeSeqGameData = timeSeqGameData
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['./data.pt']
    
    def download(self):
        pass

    def process(self):
        # process allData into geometric data format
        data_list = []
        for raw_data in self.timeSeqGameData:
            blue_num = len(raw_data.currentData.blueData)
            yellow_num = len(raw_data.currentData.yellowData) 
            x = torch.tensor(convertPlayData2List(raw_data.currentData), dtype=torch.float)
            y = torch.tensor(generateY(raw_data.predictData), dtype=torch.float)
            edge_index = torch.tensor(generateFullyConnetedEdge(blue_num+yellow_num+1), dtype=torch.long)
            edge_index = np.transpose(edge_index)
            data = Data(x=x, y=y, edge_index=edge_index.contiguous())
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
