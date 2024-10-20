import gridworld as gw
import numpy as np

def value_iteration(maze, gamma=0.9, rewards=None, threshold=1e-6):
    rows = len(maze)
    cols = len(maze[0])
    actions = ['up', 'left', 'down', 'right']  # 0 1 2 3
    val = np.zeros((rows, cols))  # Initializing a value matrix with zeroes

    if rewards is None:
        rewards = np.zeros((rows, cols))

    is_value_changed = True
    iterations = 1
    
    # Run until all values are stable (convergence)
    while is_value_changed:
        is_value_changed = False
        temp_matrix = np.copy(val)  # Create a temp matrix for the current iteration
        
        for i in range(rows):
            for j in range(cols):
                if maze[i][j] == 1:  # Skip if it's an obstacle
                    continue

                all_actions = []  # Store values from all actions
                neighbors = gw.get_neighbours(i, j, maze)

                for a in range(len(actions)):
                    value = 0
                    # If the neighbor for the chosen action is valid
                    if neighbors[a][0] != 0:
                        # Deterministic update based on the action
                        next_i, next_j = gw.neighborPoint(i, j, a)
                        if gw.is_cell_inside(next_i, next_j, maze):
                            value = rewards[i][j] + gamma * val[next_i][next_j]

                    all_actions.append(value)

                # Select the best action deterministically
                temp_matrix[i][j] = max(all_actions)

        # Check the maximum change in value between iterations
        delta = np.max(np.abs(val - temp_matrix))
        if delta > threshold:  # If the change is significant, continue
            is_value_changed = True
        
        val = temp_matrix  # Update the value matrix
        iterations += 1

        # Break if the change is smaller than the threshold
        if not is_value_changed:
            break
    
    return val, iterations

def main():
    maze = [[0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0]]    

    # Assign rewards
    rewards = np.zeros((len(maze), len(maze[0])))
    rewards[0][8] = 100  # Reward for goal state
    
    value_iteration_matrix, iterations = value_iteration(maze, rewards=rewards)
    
    print("Value Iteration Matrix after {} iterations:".format(iterations))
    print(value_iteration_matrix)

if __name__ == "__main__":
    main()
