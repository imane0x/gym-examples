import gym
from gym import spaces
import numpy as np
import pygame

class NQueensEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, n=8):
        super(NQueensEnv, self).__init__()
        self.n = n
        self.board = np.zeros((n, n), dtype=int)
        self.action_space = spaces.Discrete(n * n)
        self.observation_space = spaces.Box(0, 1, shape=(n, n), dtype=int)
        self.screen = None
        self.cell_size = 50
        self.width = self.cell_size * self.n
        self.height = self.cell_size * self.n

    def reset(self):
        self.board = np.zeros((self.n, self.n), dtype=int)
        return self.board

    def step(self, action):
        row, col = divmod(action, self.n)
        if self.board[row, col] == 1:
            return self.board, -1, False, {}
        
        self.board[row, col] = 1
        if self.is_goal_state():
            return self.board, 1, True, {}
        else:
            return self.board, 0, False, {}

    def is_goal_state(self):
        for row in range(self.n):
            for col in range(self.n):
                if self.board[row, col] == 1 and not self.is_valid_position(row, col):
                    return False
        return np.sum(self.board) == self.n

    def is_valid_position(self, row, col):
        for i in range(self.n):
            if i != row and self.board[i, col] == 1:
                return False
            if i != col and self.board[row, i] == 1:
                return False
            if row + i < self.n and col + i < self.n and i != 0 and self.board[row + i, col + i] == 1:
                return False
            if row - i >= 0 and col - i >= 0 and i != 0 and self.board[row - i, col - i] == 1:
                return False
            if row + i < self.n and col - i >= 0 and i != 0 and self.board[row + i, col - i] == 1:
                return False
            if row - i >= 0 and col + i < self.n and i != 0 and self.board[row - i, col + i] == 1:
                return False
        return True

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))

        self.screen.fill((255, 255, 255))
        for row in range(self.n):
            for col in range(self.n):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                if (row + col) % 2 == 0:
                    pygame.draw.rect(self.screen, (240, 240, 240), rect)
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect)
                if self.board[row, col] == 1:
                    pygame.draw.circle(self.screen, (0, 0, 0), rect.center, self.cell_size // 3)
        
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


