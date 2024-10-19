# value iteration algo
import grid
import numpy as np
import mdpstate

def value_iteration(maze,gamma=0.9):
    rows, cols = maze.shape
    