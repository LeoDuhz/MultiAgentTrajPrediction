from dataFormat import TimeSeqGameData
from parameters import timeDiff, playerNum, fieldLength, fieldWidth, penaltyLength, penaltyWidth
from dataFormat import Player, BallData, GameData, TimeSeqGameData
from referee import NORMAL_START, FORCE_START


def generateTimeSeqData(allData):
    allTimeSeqData = []
    for i in range(len(allData)):
        if(i%1 != 0):
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
