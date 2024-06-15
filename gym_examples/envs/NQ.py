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
        self.current_step = 0
        self.max_steps = 500
        self.width = self.cell_size * self.n
        self.height = self.cell_size * self.n
        self.render_mode = 'rgb_array'
        self.window = None
        self.clock = None
      
    def reset(self, **kwargs):
        self.state = np.random.choice(self.n, size=self.n, replace=False)  # Random initial state
        self.current_step = 0
        if self.render_mode =="human":
          self._render_frame()
        return self.state, {}
      
    def step(self, action):
        self.current_step= self.current_step +1
        row = action % self.n
        col = action // self.n
        self.state[row] = col
        reward = self.calculate_reward()
        if reward == 0 or self.current_step >= self.max_steps:
            done = True
        else:
          done  = False
        return self.state, reward, done, False, {}


    def calculate_reward(self):
        # Initialize violations count
        violations = 0
        # Create sets to keep track of conflicts
        row_set = set()
        diag1_set = set()
        diag2_set = set()

        # Iterate through each queen
        for i in range(self.n):
            # Calculate positions on diagonals
            diag1 = self.state[i] + i
            diag2 = self.state[i] - i

            # Check for conflicts
            if self.state[i] in row_set or diag1 in diag1_set or diag2 in diag2_set:
                violations += 1

            # Update sets
            row_set.add(self.state[i])
            diag1_set.add(diag1)
            diag2_set.add(diag2)

        # Reward is negative number of violations
        return -violations

    def render(self, mode='rgb_array'):
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
    def is_terminal_state(self):
        return self.calculate_reward() == 0  # Terminal state reached when there are no violations
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
