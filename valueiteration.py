# value iteration algo
import gridworld as gw
import numpy as np
import mdpstate

def value_iteration(maze,gamma=0.9,transition_prob=0.8):
    rows, cols = maze.shape
    actions = ['up','left','down','right'] # 0 1 2 3
    val = np.zeros((rows, cols)) #initializing a value matrix with zeroes
    last_state = [[0,val]] # a matrix to store iteration number and corresponding state values at that particular iteration. This will be used while calculating the next iteration. Initially store only zeroth iteration
    is_value_changed = True

    iterations = 1
    # until all values are stable
    while is_value_changed:
        is_value_changed = False
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                all_actions = [] # store values from all actions
                neighbors = gw.get_neighbours(i,j,maze)
                for a in range(len(actions)):
                    if neighbors[a][0] != 0: # if for a given action, we have valid neighbour
                        value = transition_prob*(0 + gamma*val[neighbors[a][1][0]][neighbors[a][1][1]])
                    # now with 0.1 & 0.1 probabilities, move in possible other two directions except the opposite direction
                    for dir in range(len(actions)):
                        if(abs(dir-a)!=2 and gw.is_cell_inside(gw.pointInGrid(i,j,dir))):
                            value = value + ((1.0-transition_prob) // 2)*(0 + gamma*val[neighbors[dir][1][0]][neighbors[dir][1][1]])
                    all_actions.append(value)

                v = max(all_actions)

                if v != val[i][j]: # continue until convergence
                    is_value_changed = True
                    val[i][j] = v

        iterations += 1

              
