import torch

#parameters initialization
category = 3
playerNum = 8
fieldLength = 9000
fieldWidth = 6000
penaltyLength = 2000
penaltyWidth = 1000
timeDiff = 1
savePic = False
accThreshold = 0.08
accThresholdReal = 650
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
