import numpy as np
from numpy.random import shuffle

def isWall(num):
    if num == 0:  # if 0, then it is not a wall or obstacle
        return False
    else:
        return True
    

    
def get_neighbours(x,y,maze): # find valid grid neighbours
    nbs = []
    rows, cols = maze.shape
    if x>1 and (not isWall(maze[x-1][y])):
        nbs.append((1,(x-1,y))) # puts 1 if up is a valid neighbour
    else:
        nbs.append((0,(-1,-1)))
    if y>1 and (not isWall(maze[x][y-1])):
        nbs.append((1,(x,y-1))) # puts 1 if left is a valid neighbour
    else:
        nbs.append((0,(-1,-1)))
    if x<rows-1 and (not isWall(maze[x+1][y])):
        nbs.append((1,(x+1,y))) # puts 1 if down is a valid neighbour
    else:
        nbs.append((0,(-1,-1)))
    if y<cols-1 and (not isWall(maze[x][y+1])):
        nbs.append((1,(x,y+1))) # puts 1 if right is a valid neighbour
    else:
        nbs.append((0,(-1,-1)))
    
    return nbs # return array of tuples of valid neighbours

def is_cell_inside(x,y,maze): # checks if a cell is inside grid or not
    rows, cols = maze.shape
    return (x>=0 and x<rows) and (y>=0 and y<cols)

def pointInGrid(x,y,direction): # returns the neighbours of a given point
    if(direction==0):
        return (x-1,y)
    elif (direction==1):
        return (x,y-1)
    elif (direction==2):
        return (x+1,y)
    elif (direction==3):
        return (x,y+1)


# Then for probability, p = 1/len(nbs)
# Reward is 0 every where except the goal state, where it is 1.0
# In Bellman equation, action a is the policy
# We will iterate over all four actions up, down, left, right and then take the maximum value
# Vi(s) means it is a whole state of the grid. All grid will have values which will change as we iterate more 
# for transition function , I am assuming, 0.8 probability that it actually ends up at cell where it intends to move to and rest probability equally distributed in other directions
# Also probability of ending up in opposite direction, give that it intends to move in a given direction = 0
# Repeat above process in loop for each action, up, down, left, right
# reward = 0 everywhere except when you are in the terminal state.
# when in the terminal state, we don't do iteration, we keep it +1 only