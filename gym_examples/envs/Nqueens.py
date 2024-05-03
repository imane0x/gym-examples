import numpy as np
import gym
from gym import spaces
import random

class NQueensEnv(gym.Env):
    def __init__(self, n=8):
        self.n = n
        self.board = np.zeros((n, n), dtype=int)  
        self.queens = [] 
        self.action_space = spaces.Discrete(n * n) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(n, n), dtype=np.uint8) 

    def reset(self):
        self.board = np.zeros((self.n, self.n), dtype=int)
        self.queens = []
        return self.board

    def step(self, action):
        row = action // self.n
        col = action % self.n

        if self.is_valid_move(row, col):
            self.board[row, col] = 1
            self.queens.append((row, col))
            done = len(self.queens) == self.n
            #reward = 1 if done else 0
            reward = 5
        else:
            reward = -1
            done = False

        return self.board, reward, done, {}

    def is_valid_move(self, row, col):
        for r, c in self.queens:
            if r == row or c == col or abs(r - row) == abs(c - col):
                return False
        return True

    def render(self, mode='human'):
        return self.board

    def close(self):
        pass

