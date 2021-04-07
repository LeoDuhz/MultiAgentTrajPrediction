import torch
from itertools import chain
import itertools
from dataPreprocess import generateTimeSeqData, convertPlayData2List, convertPlayData2Y, convertBallData2List, convertBallData2Y, checkPlayerDataValid, checkBallDataValid
from parameters import playerNum, device


class SSLData():
    def __init__(self, x, edges, y):
        self.x = x
        self.edges = edges
        self.y = y

class SSLDataset():
    def __init__(self, timeSeqData):
        self.timeSeqData = timeSeqData
        self.dataset = []
    

    def process(self):
        for dat in self.timeSeqData:
            cur = dat.currentData
            predict = dat.predictData
            x = []
            x.append(torch.tensor(convertPlayData2List(cur.blueData)).to(device))
            x.append(torch.tensor(convertPlayData2List(cur.yellowData)).to(device))
            x.append(torch.tensor(convertBallData2List(cur.ballData)).to(device))
            
            node_num = [playerNum,playerNum,1]
            a = [list(range(n)) for n in node_num]
            edges = [torch.LongTensor(list(itertools.product(a[i],a[j]))).T.to(device) for i in range(3) for j in range(3)]
            
            y = []
            y.append(torch.tensor(convertPlayData2Y(predict.blueData)).to(device))
            y.append(torch.tensor(convertPlayData2Y(predict.yellowData)).to(device))
            y.append(torch.tensor(convertBallData2Y(predict.ballData)).to(device))
            
            data = SSLData(x, edges, y)
            self.dataset.append(data)
