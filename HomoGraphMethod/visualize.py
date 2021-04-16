import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from parameters import playerNum, fieldLength, fieldWidth, savePic
from dataFormat import Player, GameData

def plotData(gameData):
    plt.ion()
    plt.xlim(-fieldLength/2-200, fieldLength/2+200)
    plt.ylim(-fieldWidth/2-200, fieldWidth/2+200)

    blue_x = []
    blue_y = []
    yellow_x = []
    yellow_y = []

    for bluePlayer in gameData.blueData:
        x = bluePlayer.x
        y = bluePlayer.y
        blue_x.append(x)
        blue_y.append(y)
    
    for yellowPlayer in gameData.yellowData:
        x = yellowPlayer.x
        y = yellowPlayer.y
        yellow_x.append(x)
        yellow_y.append(y)

    ball_x = gameData.ballData.x
    ball_y = gameData.ballData.y

    ax = plt.subplot(111)
    p1 = ax.scatter(blue_x, blue_y, s=70, c='blue')
    p2 = ax.scatter(yellow_x, yellow_y, s=70, c='orange')
    p3 = ax.scatter(ball_x, ball_y, s=80, c='black')
    # plt.scatter(blue_x, blue_y, s=70, c='blue')
    # plt.scatter(yellow_x, yellow_y, s=70, c='orange')
    plt.legend((p1, p2, p3), ('blue player', 'yellow player', 'ball'), loc=2)
    plt.show()
    plt.pause(0.0001)
    plt.clf()


def plotTimeSeqData(gameData, gameData2):
    plt.ion()
    # plt.xlim(-fieldLength/2-200, fieldLength/2+200)
    # plt.ylim(-fieldWidth/2-200, fieldWidth/2+200)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    blue_x = []
    blue_y = []
    blue_x_pred = []
    blue_y_pred = []

    yellow_x = []
    yellow_y = []
    yellow_x_pred = []
    yellow_y_pred = []
    
    if len(gameData2.blueData) != playerNum:
        print('predict blue data num error!')

    if len(gameData2.yellowData) != playerNum:
        print('predict yellow data num error!')

    for bluePlayer in gameData.blueData:
        x = bluePlayer.x
        y = bluePlayer.y
        blue_x.append(x)
        blue_y.append(y)
    
    for yellowPlayer in gameData.yellowData:
        x = yellowPlayer.x
        y = yellowPlayer.y
        yellow_x.append(x)
        yellow_y.append(y)

    for bluePlayer in gameData2.blueData:
        x = bluePlayer.x
        y = bluePlayer.y
        blue_x_pred.append(x)
        blue_y_pred.append(y)
    
    for yellowPlayer in gameData2.yellowData:
        x = yellowPlayer.x
        y = yellowPlayer.y
        yellow_x_pred.append(x)
        yellow_y_pred.append(y)

    ball_x = gameData.ballData.x
    ball_y = gameData.ballData.y

    ball_x_pred = gameData2.ballData.x
    ball_y_pred = gameData2.ballData.y

    ax = plt.subplot(111)
    p1 = ax.scatter(blue_x, blue_y, s=70, c='blue')
    p2 = ax.scatter(yellow_x, yellow_y, s=70, c='orange')
    p3 = ax.scatter(ball_x, ball_y, s=80, c='green')
    p4 = ax.scatter(blue_x_pred, blue_y_pred, s=70, c='navy')
    p5 = ax.scatter(yellow_x_pred, yellow_y_pred, s=70, c='darkgoldenrod')
    p6 = ax.scatter(ball_x_pred, ball_y_pred, s=80, c='lightseagreen')
    for i in range(len(blue_x)):
        ax.add_line(Line2D([blue_x[i], blue_x_pred[i]], [blue_y[i], blue_y_pred[i]], linewidth=1, color='red'))
        ax.add_line(Line2D([yellow_x[i], yellow_x_pred[i]], [yellow_y[i], yellow_y_pred[i]], linewidth=1, color='red'))
    ax.add_line(Line2D([ball_x, ball_x_pred], [ball_y, ball_y_pred], linewidth=1, color='red'))

    plt.legend((p1, p2, p3, p4, p5, p6), ('blue player', 'yellow player', 'ball', 'blue predict', 'yellow predict', 'ball predcit'), loc=2)
    if savePic:
        global epo
        global picNum
        plt.savefig('./pic/epoch{:03d}pic{:04d}.jpg'.format(epo, picNum))
        picNum += 1
    plt.show()
    plt.pause(0.001)
    plt.clf()


def drawRobots(o, l):
    gameData = GameData()
    for blue in o[0]:  
        # print('blue: ', o[0])          
        team = 0
        number = 1
        x = blue[0]
        y = blue[1]
        orientation = 0
        velx = 0
        vely = 0
        w = 0
        bluePlayer = Player(team, number, x, y, orientation, velx, vely, w)
        gameData.blueData.append(bluePlayer)
    for yellow in o[1]:
        # print('yellow: ', o[1]) 
        team = 0
        number = 1
        x = yellow[0]
        y = yellow[1]
        orientation = 0
        velx = 0
        vely = 0
        w = 0
        yellowPlayer = Player(team, number, x, y, orientation, velx, vely, w)
        gameData.yellowData.append(yellowPlayer)        
    
    for ball in o[2]:
        x = ball[0]
        y = ball[1]
        velx = 0
        vely = 0
        gameData.ballData.reset(x, y, velx, vely)

    gameData2 = GameData()
    for blue in l[0]:            
        team = 0
        number = 1
        x = blue[0]
        y = blue[1]
        orientation = 0
        velx = 0
        vely = 0
        w = 0
        bluePlayer = Player(team, number, x, y, orientation, velx, vely, w)
        gameData2.blueData.append(bluePlayer)

    for yellow in l[1]:
        team = 0
        number = 1
        x = yellow[0]
        y = yellow[1]
        orientation = 0
        velx = 0
        vely = 0
        w = 0
        yellowPlayer = Player(team, number, x, y, orientation, velx, vely, w)
        gameData2.yellowData.append(yellowPlayer)        
    
    for ball in l[2]:
        x = ball[0]
        y = ball[1]
        velx = 0
        vely = 0
        gameData2.ballData.reset(x, y, velx, vely)

    plotTimeSeqData(gameData2, gameData)