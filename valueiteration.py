# value iteration algo
import gridworld as gw
import numpy as np

def value_iteration(maze,gamma=0.9,transition_prob=0.8,rewards=None, threshold=1e-6):
    rows = len(maze)
    cols = len(maze[0])
    actions = ['up','left','down','right'] # 0 1 2 3
    val = np.zeros((rows, cols)) #initializing a value matrix with zeroes
    # last_state = [[0,val]] # a matrix to store iteration number and corresponding state values at that particular iteration. This will be used while calculating the next iteration. Initially store only zeroth iteration

    if rewards is None:
        rewards = np.zeros((rows, cols))
    is_value_changed = True

    iterations = 1
    # until all values are stable
    while is_value_changed:
        is_value_changed = False

        temp_matrix = np.copy(val)
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if maze[i][j] == 1:  # skipping obstacles for now
                    continue

                all_actions = [] # store values from all actions
                neighbors = gw.get_neighbours(i,j,maze)
                for a in range(len(actions)):
                    value = 0
                    if neighbors[a][0] != 0:  # If a valid neighbor exists for this action
                        next_i, next_j = gw.neighborPoint(i, j, a)
                        value = transition_prob * (rewards[i][j] + gamma * val[next_i][next_j]) # use values from last iteration only

                    # now with 0.1 & 0.1 probabilities, move in possible other two directions except the opposite direction
                    for dir in range(len(actions)):
                        if abs(dir - a) != 2:  # not in opposite direction
                            next_i, next_j = gw.neighborPoint(i, j, dir)
                            if gw.is_cell_inside(next_i, next_j, maze):
                                value += ((1.0 - transition_prob) / 2) * (rewards[i][j] + gamma * val[next_i][next_j])

                    all_actions.append(value)

                temp_matrix[i][j] = max(all_actions)

        delta = np.max(np.abs(val - temp_matrix))
        if delta > threshold:  
            is_value_changed = True
        
        val = np.copy(temp_matrix)  # copy the updated one
        
                # Update the value matrix
                # if v != val[i][j]: # continue until convergence
                #     is_value_changed = True
                #     val[i][j] = v
        
        if not is_value_changed:
            break

        iterations += 1
    return val, iterations

def main():
    maze = [[0,0,0,1,0,0,1,0,0],
            [0,0,0,1,0,0,1,0,0],
            [0,0,0,1,0,0,0,0,0],
            [0,0,0,0,1,1,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0]
           ]    

    rewards = np.zeros((len(maze), len(maze[0])))
    rewards[0][8] = 1  # Reward for goal state
    
    value_iteration_matrix, iterations = value_iteration(maze, rewards=rewards)
    
    print("Value Iteration Matrix after {} iterations:".format(iterations))
    print(value_iteration_matrix)

if __name__=="__main__":
    main()     
