import gym
from gym import spaces
import numpy as np
import pygame
from PIL import Image
import io

class NQ(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, n=8):
        super(NQ, self).__init__()
        self.n = n
        self.board = np.zeros((n, n), dtype=int)
        self.action_space = spaces.Discrete(n * n)
        self.observation_space = spaces.Box(0, 1, shape=(n, n), dtype=int)
        self.screen = None
        self.cell_size = 50
        self.width = self.cell_size * self.n
        self.height = self.cell_size * self.n

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.n, self.n), dtype=int)
        return self.board, {}

    def step(self, action):
        row, col = divmod(action, self.n)
        if self.board[row, col] == 1:
            return self.board, -1, False, False, {}
        
        self.board[row, col] = 1
        if self.is_goal_state():
            return self.board, 1, True, False, {}
        else:
            return self.board, 0, False, False, {}

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

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
    
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((255, 255, 255))
        pix_square_size = self.cell_size  # The size of a single grid square in pixels
    
        # Drawing the queens on the board
        for row in range(self.n):
            for col in range(self.n):
                if self.board[row, col] == 1:
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 0),
                        ((col + 0.5) * pix_square_size, (row + 0.5) * pix_square_size),
                        pix_square_size // 3
                    )
    
        # Drawing gridlines
        for x in range(self.n + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, pix_square_size * x),
                (self.width, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.height),
                width=2,
            )
    
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

