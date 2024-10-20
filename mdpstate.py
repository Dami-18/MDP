# defining MDP state
# here we define a coordinate in grid (x,y) as state and each state has a reward and value associated with it

class MDPState:
     def __init__(self,up,down,left,right,reward=0,value=0): # all have zero reward
          self.up = up
          self.down = down
          self.left = left
          self.right = right
          self.reward = reward
          self.value = value