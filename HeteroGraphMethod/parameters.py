import torch

#parameters initialization
playerNum = 6
fieldLength = 9000
fieldWidth = 6000
penaltyLength = 2000
penaltyWidth = 1000
timeDiff = 2
savePic = False
accThreshold = 0.08
accThresholdReal = 650
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')