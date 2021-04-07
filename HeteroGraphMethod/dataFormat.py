#basic data format in Robocup SSL game
#player, balldata, gamedata, time sequence gamedata are defined

class Player:
    def __init__(self, team, number, x, y, orientation, velx, vely, w):
        super().__init__()
        self.team = team
        self.number = number
        self.x = x
        self.y = y
        self.orientation = orientation
        self.velx = velx
        self.vely = vely
        self.w = w

class BallData:
    def __init__(self):
        super().__init__()
        self.x = 0
        self.y = 0
        self.velx = 0
        self.vely = 0
    
    def reset(self, x, y, velx, vely):
        self.x = x
        self.y = y
        self.velx = velx
        self.vely = vely

class GameData:
    def __init__(self):
        super().__init__()
        self.time = 0
        self.blueData = []
        self.yellowData = []
        self.ballData = BallData()

class TimeSeqGameData:
    def __init__(self):
        super().__init__()
        self.timeDiff = None
        self.currentData = GameData()
        self.predictData = GameData()

    def reset(self, currentData, predictData):
        self.currentData = currentData
        self.predictData = predictData

