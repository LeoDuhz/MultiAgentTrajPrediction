#test our model, draw pictures, get accuracy
import numpy as np
from pathlib import Path
import torch
from torch_geometric.data import Data
from torch.utils.data.dataset import Dataset
import torch.optim as optim
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops
import random
import math
from torch.utils.tensorboard import SummaryWriter
from dataFormat import Player, BallData, GameData, TimeSeqGameData
from parameters import playerNum, fieldLength, fieldWidth, penaltyLength, penaltyWidth, timeDiff, savePic, accThreshold, accThresholdReal, device
from dataPreprocess import readFromText, generateTimeSeqData, convertPlayData2List, convertPlayData2Y, convertBallData2List, convertBallData2Y, checkPlayerDataValid, checkBallDataValid, processMin_Max_Norm
from visualize import plotTimeSeqData, plotData, drawRobots
from SSLDataset import SSLData, SSLDataset
from Net import Net

epo = 0
picNum = 0

# File preprocessing
print('Data preprocessing begins!')

allData = readFromText('../dhzData')
allData = processMin_Max_Norm(allData)
print('dataset Min-Max normalization end!')

timeSeqGameData = generateTimeSeqData(allData)
random.shuffle(timeSeqGameData)
print('read time Seq Data number: ', len(timeSeqGameData))

#dataset construction
ssldata = SSLDataset(timeSeqGameData)
ssldata.process()

length = len(timeSeqGameData)
train_len = int(0.7 * length)
val_len = int(0.15 * length)
test_len = length - train_len - val_len

model = torch.load('./modelname.pth')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
batch_size = 30
batch_num = int(train_len/batch_size) + 1

def evaluate(loader, draw=False):
    model.eval()
    loss_all = 0
    accNum = 0
    accNumReal = 0
    ERROR_ALL = 0
    with torch.no_grad():
        for data in loader:
            # accNum = 0
            out = model(data)
            label = data.y
            loss = criterion(out[0], label[0])
            loss = loss + criterion(out[1], label[1])
            loss_all += loss.item()

            o = []
            l = []
            for i in range(3):
                temp = out[i].cpu().detach().numpy().tolist()
                temp2 = label[i].cpu().detach().numpy().tolist()
                o.append(temp)
                l.append(temp2)
            
            for j in range(2):
                for k in range(len(o[j])):
                    error = np.linalg.norm(np.array(o[j][k]) - np.array(l[j][k]))
                    # print('error: ', error)
                    if error <= accThreshold:
                        accNum += 1

            for j in range(2):
                for k in range(len(o[j])):
                    o[j][k][0] = o[j][k][0] * fieldLength - fieldLength/2
                    o[j][k][1] = o[j][k][1] * fieldWidth - fieldWidth/2
                    l[j][k][0] = l[j][k][0] * fieldLength - fieldLength/2
                    l[j][k][1] = l[j][k][1] * fieldWidth - fieldWidth/2

                    error_real = np.linalg.norm(np.array(o[j][k]) - np.array(l[j][k]))
                    ERROR_ALL += error_real
                    if error_real <= accThresholdReal:
                        accNumReal += 1
            # print('single data accuracy: ', accNum/2/playerNum)

            if draw:
                drawRobots(o,l)
    print('length of loader:', len(loader), "accNum: ", accNum, 'real accuracy: ', accNumReal/len(loader)/2/playerNum, 'average error: ', ERROR_ALL/len(loader)/2/playerNum)
    return accNum / (len(loader) * (2 * playerNum)), loss_all / len(loader)

def main():
    print('Start training')
    # f = open("./epochAcc.txt", "w")
    for epoch in range(1):
        global epo
        global picNum
        global batch_size
        global batch_num
        
        picNum = 0

        if epoch > -1:
            # if epoch > 40:
            #     draw = True
            train_acc, train_loss = evaluate(ssldata.dataset[:train_len])
            # val_acc, val_loss = evaluate(ssldata.dataset[train_len:train_len+val_len])
            test_acc, test_loss = evaluate(ssldata.dataset[train_len+val_len:])

        print('epoch:{:03d}, train loss:{:.5f}, train accuracy:{:.5f}, test loss:{:.5f}, test accuracy:{:.5f} '.format(epoch, loss, train_acc, test_loss, test_acc))
        # f.write('epoch:{:03d}, train loss:{:.5f}, train accuracy:{:.5f}, valid accuracy:{:.5f}, test accuracy:{:.5f} '.format(epoch, loss, train_acc, val_acc, test_acc))
        epo += 1


if __name__ == "__main__":

    main()
    print('End of All Tasks!!!')


        