import numpy as np
import utils
import random

class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
        self.action_list = [3,2,1,0]
        self.reset()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
        
    def discretize(self, state):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state
        #discretize wall directions 
        adjoining_walls = [0,0]
        if snake_head_x == utils.GRID_SIZE:
            adjoining_walls[0] = 1
        if snake_head_x == 12*utils.GRID_SIZE:
            adjoining_walls[0] = 2
        if snake_head_y == utils.GRID_SIZE:
            adjoining_walls[1] = 1
        if snake_head_y == 12*utils.GRID_SIZE:
            adjoining_walls[1] = 2
        #discretize food directions
        food_dirs = [0,0]
        if snake_head_x - food_x > 0:
            food_dirs[0] = 1
        if snake_head_x - food_x < 0:
            food_dirs[0] = 2 
        if snake_head_y - food_y > 0:
            food_dirs[1] = 1
        if snake_head_y - food_y < 0:
            food_dirs[1] = 2     
        #discretize snake body  
        adjoining_bodies = [0,0,0,0]
        for snake_body_x, snake_body_y in snake_body:
            if snake_body_x == snake_head_x and snake_body_y == snake_head_y - utils.GRID_SIZE:
                adjoining_bodies[0] = 1
            if snake_body_x == snake_head_x and snake_body_y == snake_head_y + utils.GRID_SIZE:
                adjoining_bodies[1] = 1
            if snake_body_y == snake_head_y and snake_body_x == snake_head_x - utils.GRID_SIZE:
                adjoining_bodies[2] = 1
            if snake_body_y == snake_head_y and snake_body_x == snake_head_x + utils.GRID_SIZE:
                adjoining_bodies[3] = 1
        return [adjoining_walls[0], adjoining_walls[1], food_dirs[0], food_dirs[1], adjoining_bodies[0], adjoining_bodies[1], adjoining_bodies[2], adjoining_bodies[3]]

    def r(self, points, dead):
        if dead:
            return -1
        if points - self.points > 0:
            return 1
        return -0.1
    
    def f(self,u, n):
        if n < self.Ne:
            return 1
        return u
    
    def maxQ(self, state):
        q = None
        a = None 
        for action in self.action_list:
            sa = tuple(state[:] + [action])
            q_table = self.Q[sa]
            if q == None:
                q = q_table
                a = action
                continue
            if q_table > q:
                q = q_table
                a = action 
        return (q, a)
    
    def update_Qtable(self, next_state, r):
        prev_s = self.discretize(self.s)
        prev_sa = tuple(prev_s[:] + [self.a])
        alpha = self.C / (self.C + self.N[prev_sa])
        maxQ = self.maxQ(next_state)[0]
        self.Q[prev_sa] += alpha*(r + self.gamma*maxQ - self.Q[prev_sa])
    
    def explore(self, state):
        maxA, maxF = (None, None)
        for action in self.action_list:
            next_sa = tuple(state[:] + [action])
            f = self.f(self.Q[next_sa], self.N[next_sa])
            if maxF == None:
                maxA = action
                maxF = f
                continue
            if f > maxF:
                maxA = action
                maxF = f
        return maxA 
    
    def act(self, state, points, dead):
        next_state = self.discretize(state)
        if self.train:
            if self.s != None:
                self.update_Qtable(next_state, self.r(points, dead))
            maxA = self.explore(next_state)
            if not dead:
                next_sa = tuple(next_state[:] + [maxA])
                self.N[next_sa] += 1
            self.s = state[:]
            self.a = maxA 
            self.points = points 
        else:
            maxA = self.maxQ(next_state)[1]
        if dead:
            self.reset()
        return maxA