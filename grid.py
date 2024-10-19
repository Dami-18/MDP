import numpy as np
from numpy.random import shuffle

def isWall(ch): # takes character as input
    if ch == '0':  # if 0, then it is not a wall or obstacle
        return False
    else:
        return True
    
def get_neighbours(x,y,maze): # find valid grid neighbours
    nbs = []
    if x>0 and (not isWall(maze[x+1][y])):
        nbs.append((x+1,y))
    if y>0 and (not isWall(maze[x][y+1])):
        nbs.append((x,y+1))
    if x>1 and (not isWall(maze[x-1][y])):
        nbs.append((x-1,y))
    if y>1 and (not isWall(maze[x][y-1])):
        nbs.append((x,y-1))
    shuffle(nbs)
    
    return nbs