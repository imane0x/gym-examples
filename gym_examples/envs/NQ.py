import gym
from gym import spaces
import numpy as np
import pygame

class NQ(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, n=8, render_mode='rgb_array'):
       # super(NQ, self).__init__()
        self.n = n
        self.board = np.zeros((n, n), dtype=int)
        self.action_space = spaces.Discrete(n * n)
        self.observation_space = spaces.Box(0, 1, shape=(n, n), dtype=int)
        self.cell_size = 50
        self.width = self.cell_size * self.n
        self.height = self.cell_size * self.n
        self.render_mode = 'rgb_array'
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.n, self.n), dtype=int)
        if self.render_mode == "human":
            self._render_frame()
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

    def render(self, mode='rgb_array):
        # if mode is not None:
        #     self.render_mode = mode
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        elif self.render_mode == 'human':
            self._render_human()

    def _render_frame(self):
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
    
        return np.transpose(
            pygame.surfarray.array3d(canvas), axes=(1, 0, 2)
        )

    def _render_human(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
    
        canvas = self._render_frame()
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(30)  # Adjust FPS as needed

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
