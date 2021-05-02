import torch
import math
from dataFormat import TimeSeqGameData
from parameters import timeDiff, playerNum, fieldLength, fieldWidth, penaltyLength, penaltyWidth, device
from dataFormat import Player, BallData, GameData, TimeSeqGameData
from referee import NORMAL_START, FORCE_START


def generateTimeSeqData(allData):
    allTimeSeqData = []
    for i in range(len(allData)):
        if(i%3 != 0):
            continue
        # time = allData[i].time
        for j in range(i+1, len(allData)):
            tempSeqData = TimeSeqGameData()
            if(allData[j].time - allData[i].time >= timeDiff and allData[j].time - allData[i].time <= timeDiff + 0.2):
                print('time diff: ', allData[j].time - allData[i].time)
                tempSeqData.reset(allData[i], allData[j])
                allTimeSeqData.append(tempSeqData)
                # print('add one time Seq data, total num: ', len(allTimeSeqData))
                break
    return allTimeSeqData

def convertPlayData2List(playerData):
    data_list = []
    for data in playerData:
        temp = [data.x, data.y, data.velx, data.vely]
        data_list.append(temp)
    
    return data_list

def convertPlayData2Y(playerData):
    data_list = []
    for data in playerData:
        temp = [data.x, data.y]
        data_list.append(temp)
    
    return data_list

def convertBallData2List(ballData):
    data_list = []
    temp = [ballData.x, ballData.y, ballData.velx, ballData.vely]
    data_list.append(temp)

    return data_list

def convertBallData2Y(ballData):
    data_list = []
    temp = [ballData.x, ballData.y]
    data_list.append(temp)

    return data_list

def generateEdges(gameData1, gameData2):
    source = []
    target = []

    #special check for ballData for it is not list type data
    if(type(gameData1).__name__ != 'list'):
        gameData1 = [gameData1]

    if(type(gameData2).__name__ != 'list'):
        gameData2 = [gameData2]


    for i in range(len(gameData1)):
        for j in range(len(gameData2)):
            if(checkConnection(gameData1[i], gameData2[j])):
                source.append(i)
                target.append(j)
    
    return [source, target]

def findKNearestNeighbor(gameData):
    K = 8

    blueNeighborList = []
    yellowNeighborList = []
    ballNeighborList = []
    #blue data's K nearest neighbour
    for blue_i in range(len(gameData.blueData)):
        bluePlayer = gameData.blueData[blue_i]
        blueDist = []

        for i in range(len(gameData.blueData)):
            dist = calDist(gameData.blueData[i], bluePlayer)
            blueDist.append(('blue', i, dist))
        
        for i in range(len(gameData.yellowData)):
            dist = calDist(gameData.yellowData[i], bluePlayer)
            blueDist.append(('yellow', i, dist))
        
        dist = calDist(gameData.ballData, bluePlayer)
        blueDist.append(('ball', 0, dist))
        blueDist.sort(key=lambda x:x[2])
        blueKNeighbor = blueDist[:K]
        blueKNeighbor.sort(key=lambda x:x[0])  #yellow blue ball
        blueNeighborList.append(blueKNeighbor)

    #yellow data's K nearest neighbour
    for yellow_i in range(len(gameData.yellowData)):
        yellowPlayer = gameData.yellowData[yellow_i]
        yellowDist = []

        for i in range(len(gameData.blueData)):
            dist = calDist(gameData.blueData[i], yellowPlayer)
            yellowDist.append(('blue', i, dist))
        
        for i in range(len(gameData.yellowData)):
            dist = calDist(gameData.yellowData[i], yellowPlayer)
            yellowDist.append(('yellow', i, dist))
        
        dist = calDist(gameData.ballData, yellowPlayer)
        yellowDist.append(('ball', 0, dist))
        yellowDist.sort(key=lambda x:x[2])
        yellowKNeighbor = yellowDist[:K]
        yellowKNeighbor.sort(key=lambda x:x[0])  #yellow blue ball
        yellowNeighborList.append(yellowKNeighbor)

    #ball data's K nearest neighbour
    ballPlayer = gameData.ballData
    ballDist = []

    for i in range(len(gameData.blueData)):
        dist = calDist(gameData.blueData[i], ballPlayer)
        ballDist.append(('blue', i, dist))
    
    for i in range(len(gameData.yellowData)):
        dist = calDist(gameData.yellowData[i], ballPlayer)
        ballDist.append(('yellow', i, dist))
    
    dist = calDist(gameData.ballData, ballPlayer)
    ballDist.append(('ball', 0, dist))
    ballDist.sort(key=lambda x:x[2])
    ballKNeighbor = ballDist[:K]
    ballKNeighbor.sort(key=lambda x:x[0])  #yellow blue ball
    ballNeighborList.append(ballKNeighbor)

    #generate edges
    bb_s, bb_t, by_s, by_t, bba_s, bba_t = [], [], [], [], [], []
    yb_s, yb_t, yy_s, yy_t, yba_s, yba_t = [], [], [], [], [], []
    bab_s, bab_t, bay_s, bay_t, baba_s, baba_t = [], [], [], [], [], []

    #generate blue edges as target node
    for i in range(len(blueNeighborList)):
        for j in range(len(blueNeighborList[i])):
            neighbour = blueNeighborList[i][j]
            if neighbour[0] == 'blue':
                bb_s.append(neighbour[1])
                bb_t.append(i)
            if neighbour[0] == 'yellow':
                yb_s.append(neighbour[1])
                yb_t.append(i)
            if neighbour[0] == 'ball':
                bab_s.append(neighbour[1])
                bab_t.append(i)
    

    #generate yellow edges as target node
    for i in range(len(yellowNeighborList)):
        for j in range(len(yellowNeighborList[i])):
            neighbour = blueNeighborList[i][j]
            if neighbour[0] == 'blue':
                by_s.append(neighbour[1])
                by_t.append(i)
            if neighbour[0] == 'yellow':
                yy_s.append(neighbour[1])
                yy_t.append(i)
            if neighbour[0] == 'ball':
                bay_s.append(neighbour[1])
                bay_t.append(i)
    
    #generate ball edges as target node
    for i in range(len(ballNeighborList)):
        for j in range(len(ballNeighborList[i])):
            neighbour = ballNeighborList[i][j]
            if neighbour[0] == 'blue':
                bba_s.append(neighbour[1])
                bba_t.append(i)
            if neighbour[0] == 'yellow':
                yba_s.append(neighbour[1])
                yba_t.append(i)
            if neighbour[0] == 'ball':
                baba_s.append(neighbour[1])
                baba_t.append(i)
    
    blueblue = torch.LongTensor([bb_s, bb_t]).to(device)
    blueyellow = torch.LongTensor([by_s, by_t]).to(device)
    blueball = torch.LongTensor([bba_s, bba_t]).to(device)
    yellowblue = torch.LongTensor([yb_s, yb_t]).to(device)
    yellowyellow = torch.LongTensor([yy_s, yy_t]).to(device)
    yellowball = torch.LongTensor([yba_s, yba_t]).to(device)
    ballblue = torch.LongTensor([bab_s, bab_t]).to(device)
    ballyellow = torch.LongTensor([bay_s, bay_t]).to(device)
    ballball = torch.LongTensor([baba_s, baba_t]).to(device)

    return [blueblue, blueyellow, blueball, yellowblue, yellowyellow, yellowball, ballblue, ballyellow, ballball]
    

def calDist(player1, player2):
    player1_x = player1.x * fieldLength - fieldLength/2
    player1_y = player1.y * fieldWidth - fieldWidth/2
    player2_x = player2.x * fieldLength - fieldLength/2
    player2_y = player2.y * fieldWidth - fieldWidth/2

    return math.hypot(player1_x-player2_x, player1_y-player2_y)

def checkConnection(source, target):
    threshold = 3000

    #reverse normalization
    source_x = source.x * fieldLength - fieldLength/2
    source_y = source.y * fieldWidth - fieldWidth/2
    target_x = target.x * fieldLength - fieldLength/2
    target_y = target.y * fieldWidth - fieldWidth/2

    if(math.hypot(source_x-target_x, source_y-target_y) <= threshold):
        return True
    
    return False


def checkPlayerDataValid(playerData):
    if(len(playerData) > playerNum):
        print('Player number is too large, pass')
        return False

    if(len(playerData) != playerNum):
        print('Player number is not ', playerNum)
        return False

    return True

def checkBallDataValid(ballData):
    if ballData.x == -99999 and ballData.y == -99999:
        print('Get invalid ball data, pass')
        return False
    
    if (ballData.x >= fieldLength/2 - penaltyWidth and abs(ballData.y) <= penaltyLength/2) or  (ballData.x <= -fieldLength/2 + penaltyWidth and abs(ballData.y) <= penaltyLength/2):
        print('Get in penalty ball data, pass')
        return False

    if (abs(ballData.x) >= fieldLength/2 or abs(ballData.y) >= fieldWidth/2):
        print('Out of field data, pass')
        return False

    return True

def readFromText(fileName):
    f = open(fileName, 'r')
    
    allData = []
    line = f.readline()
    dat = line.split(' ')
    dataValid = True
    ignoreCnt = 0
    startTime = 500

    #extract data from text file
    while True:  
        # dat = line.split(' ')
        
        if not line:
            break
        if(dat[0] == 'time:'):
            ignoreCnt += 1
            time = dat[1]
            line = f.readline()
            dat = line.split(' ')
            if(dat[0] == 'referee:'):
                referee = int(dat[1])
                if(referee == NORMAL_START or referee == FORCE_START):
                    gameData = GameData()
                    gameData.time = float(time) / 1e3
                    while True:
                        line = f.readline()
                        dat = line.split(' ')
                        if(dat[0] != 'team:' and dat[0] != 'ballpos:'):
                            # update all data
                            if(checkBallDataValid(gameData.ballData) and checkPlayerDataValid(gameData.blueData) and checkPlayerDataValid(gameData.yellowData) and ignoreCnt >= startTime):
                                allData.append(gameData)
                                # plt.clf()
                                # plotData(gameData)
                                # print('length of all data',len(allData))
                            break
                        elif(dat[0] == 'team:'):
                            team = int(dat[1])
                            # process blue player data
                            if(team == 0): 
                                if(dat[2] == 'num:'): 
                                    number = int(dat[3])
                                if(dat[4] == 'position:'):
                                    x = float(dat[5])
                                    y = float(dat[6])
                                    orientation = float(dat[7])
                                if(dat[8] == 'velocity:'):
                                    velx =float(dat[9])
                                    vely = float(dat[10])
                                    w = float(dat[11])
                                bluePlayer = Player(team, number, x, y, orientation, velx, vely, w)
                                gameData.blueData.append(bluePlayer)

                            # process yellow player data
                            elif(team == 1):
                                if(dat[2] == 'num:'): 
                                    number = int(dat[3])
                                if(dat[4] == 'position:'):
                                    x = float(dat[5])
                                    y = float(dat[6])
                                    orientation = float(dat[7])
                                if(dat[8] == 'velocity:'):
                                    velx = float(dat[9])
                                    vely = float(dat[10])
                                    w = float(dat[11])
                                yellowPlayer = Player(team, number, x, y, orientation, velx, vely, w)
                                gameData.yellowData.append(yellowPlayer)
                        # process ball data
                        elif(dat[0] == 'ballpos:'):
                            x = float(dat[1])
                            y = float(dat[2])
                            velx = float(dat[4])
                            vely = float(dat[5])
                            ball = BallData()
                            ball.reset(x, y, velx, vely)
                            gameData.ballData.reset(x, y, velx, vely)
                            dataValid = checkBallDataValid(ball)
                else:
                    while True:
                        line = f.readline()
                        dat = line.split(' ')
                        if(dat[0] == 'time:' or not line):
                            break
                
    print('End of data preprocessing, the size of useful data can be used', len(allData))

    return allData

def processMin_Max_Norm(allData):
    bluevelx = []
    bluevely = []

    yellowvelx = []
    yellowvely = []

    ballx = []
    bally = []
    ballvelx = []
    ballvely = []

    for data in allData:
        # process ball data
        ballx.append(data.ballData.x)
        bally.append(data.ballData.y)
        ballvelx.append(data.ballData.velx)
        ballvely.append(data.ballData.vely)

        #process blue players data
        for blue in data.blueData:
            bluevelx.append(blue.velx)
            bluevely.append(blue.vely)

        # process yellow players data
        for yellow in data.yellowData:
            yellowvelx.append(yellow.velx)
            yellowvely.append(yellow.vely)

    maxballx = max(ballx)
    minballx = min(ballx)
    maxbally = max(bally)
    minbally = min(bally)
    maxballvelx = max(ballvelx)
    minballvelx = min(ballvelx)
    maxballvely = max(ballvely)
    minballvely = min(ballvely)

    maxbluevelx = max(bluevelx)
    minbluevelx = min(bluevelx)
    maxbluevely = max(bluevely)
    minbluevely = min(bluevely)

    maxyellowvelx = max(yellowvelx)
    minyellowvelx = min(yellowvelx)
    maxyellowvely = max(yellowvely)
    minyellowvely = min(yellowvely)


    for data in allData:
        data.ballData.x = (data.ballData.x + fieldLength/2) / fieldLength
        data.ballData.y = (data.ballData.y + fieldWidth/2) / fieldWidth
        data.ballData.velx = (data.ballData.velx - minballvelx) / (maxballvelx - minballvelx)
        data.ballData.vely = (data.ballData.vely - minballvely) / (maxballvely - minballvely)

        for blue in data.blueData:
            blue.x = (blue.x + fieldLength/2) / fieldLength
            blue.y = (blue.y + fieldWidth/2) / fieldWidth
            blue.velx = (blue.velx - minbluevelx) / (maxbluevelx - minbluevelx)
            blue.vely = (blue.vely - minbluevely) / (maxbluevely - minbluevely)

        for yellow in data.yellowData:
            yellow.x = (yellow.x + fieldLength/2) / fieldLength
            yellow.y = (yellow.y + fieldWidth/2) / fieldWidth
            yellow.velx = (yellow.velx - minyellowvelx) / (maxyellowvelx - minyellowvelx)
            yellow.vely = (yellow.vely - minyellowvely) / (maxyellowvely - minyellowvely)

    return allData
